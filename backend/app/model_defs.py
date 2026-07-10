"""
Model architectures — copied verbatim from the original camera-inference
script so that checkpoint state_dicts load without modification.
"""
import torch
import torch.nn as nn

LH_FLAG_M = 306
RH_FLAG_M = 307


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attn_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)


class ASLv3Classifier(nn.Module):
    """Motion-aware BiGRU/V3 sequence model over a (T, 308) window."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        gru1_hidden: int = 256,
        gru2_hidden: int = 128,
        dropout_gru: float = 0.35,
        dropout_cls1: float = 0.40,
        dropout_cls2: float = 0.30,
    ):
        super().__init__()

        gru1_out = gru1_hidden * 2
        gru2_out = gru2_hidden * 2
        fused_dim = gru2_out + 64

        self.input_norm = nn.LayerNorm(feat_dim, eps=1e-5)

        self.gru1 = nn.GRU(
            feat_dim, gru1_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.drop1 = nn.Dropout(dropout_gru)

        self.gru2 = nn.GRU(
            gru1_out, gru2_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.drop2 = nn.Dropout(dropout_gru)

        self.attention = TemporalAttention(input_dim=gru2_out, attn_hidden=128)

        self.flag_proj = nn.Sequential(nn.Linear(2, 64), nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_cls1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_cls2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flags = x[:, :, LH_FLAG_M:RH_FLAG_M + 1]
        flag_summary = flags.mean(dim=1)
        flag_emb = self.flag_proj(flag_summary)

        x = self.input_norm(x)

        out1, _ = self.gru1(x)
        out1 = self.drop1(out1)

        out2, _ = self.gru2(out1)
        out2 = self.drop2(out2)

        context = self.attention(out2)
        fused = torch.cat([context, flag_emb], dim=1)
        return self.classifier(fused)


class ASLStaticModel(nn.Module):
    """Single-frame, single-hand CNN classifier. Input: (B, 1, 63)."""

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1920, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
