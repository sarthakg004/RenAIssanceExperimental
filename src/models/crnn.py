"""
CRNN (ResNet-CNN + BiLSTM + CTC) model for OCR.

Architecture
------------
Input  : (B, 1, H, W)  — grayscale line image, H = img_height (default 64)
CNN    : Lightweight ResNet backbone → (B, 256, 1, W')
LSTM   : Bidirectional LSTM         → (B, W', lstm_hidden * 2)
Head   : Linear + log_softmax       → (B, W', vocab_size)
"""

import torch
import torch.nn as nn


# ── Residual block ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


# ── Lightweight ResNet CNN backbone ───────────────────────────────────────────

class ResNetCNN(nn.Module):
    """
    Lightweight ResNet-style CNN backbone.

    Input  : (B, 1, H, W)  — H = img_height = 64
    Output : (B, 256, 1, W')

    Height reduction:
        stem + MaxPool(2,2)   → H = 32
        stage2 + MaxPool(2,2) → H = 16
        stage3 + MaxPool(2,1) → H =  8
        stage4 + MaxPool(2,1) → H =  4
        AdaptiveAvgPool(1,*)  → H =  1
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResBlock(64,  64),
            ResBlock(64,  64),
            nn.MaxPool2d(2, 2),
        )

        self.stage2 = nn.Sequential(
            ResBlock(64,  128),
            ResBlock(128, 128),
            nn.MaxPool2d(2, 2),
        )

        self.stage3 = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(256, 256),
            nn.MaxPool2d((2, 1)),
        )

        self.stage4 = nn.Sequential(
            ResBlock(256, 256),
            nn.MaxPool2d((2, 1)),
        )

        self.height_collapse = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.height_collapse(x)   # (B, 256, 1, W')
        return x


# ── CRNN ──────────────────────────────────────────────────────────────────────

class CRNN(nn.Module):
    """
    ResNet CNN → BiLSTM → Linear → CTC log-softmax.

    Parameters
    ----------
    vocab_size   : number of output classes (including CTC blank at index 0)
    lstm_hidden  : hidden units per LSTM direction  (default 256)
    lstm_layers  : number of stacked LSTM layers    (default 2)
    dropout      : dropout probability               (default 0.3)
    """

    def __init__(
        self,
        vocab_size:  int,
        lstm_hidden: int   = 256,
        lstm_layers: int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()

        self.cnn = ResNetCNN()

        self.lstm = nn.LSTM(
            input_size    = 256,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)          # (B, 256, 1, W')
        x = x.squeeze(2)         # (B, 256, W')
        x = x.permute(0, 2, 1)   # (B, W', 256)
        x, _ = self.lstm(x)      # (B, W', lstm_hidden*2)
        x = self.dropout(x)
        x = self.classifier(x)   # (B, W', vocab_size)
        return x.log_softmax(-1)
