"""
FAST-PLATE-OCR-VIETNAMESE: CCT Architecture Adapter & Multi-Head Decoder
Hiện thực hóa kiến trúc Compact Convolutional Transformer (CCT) tương thích cấu hình biển số Việt Nam.
Tích hợp Convolutional Tokenizer, Learnable Positional Embedding và 10 Classification Heads.
"""

import os
import sys
import yaml
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    F = object

from src.grammar.dfa import VietnamPlateDFA


class ConvTokenizer(nn.Module if HAS_TORCH else object):
    """
    Convolutional Tokenizer trong CCT:
    Trích xuất các đặc trưng không gian cục bộ từ ảnh đầu vào (3, 64, 256)
    và chuyển đổi thành chuỗi N tokens với kích thước embedding_dim (D).
    """
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (1, 1),
        pooling_kernel_size: Tuple[int, int] = (3, 3),
        pooling_stride: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            padding=1,
        )

    def forward(self, x):
        # x: [B, 3, 64, 256]
        x = self.conv(x)  # [B, D, 32, 128]
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)  # [B, D, 16, 64]
        # Chuyển đổi thành chuỗi tokens: [B, D, H', W'] -> [B, N, D]
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) # [B, H'*W', D] = [B, 1024, 128]
        return tokens


class PositionalEmbedding(nn.Module if HAS_TORCH else object):
    """
    Tầng nhúng vị trí có thể học được (Learnable 1D Positional Embedding):
    Bổ sung thông tin trật tự không gian từ trái sang phải cho các ký tự.
    """
    def __init__(self, sequence_length: int = 10, embedding_dim: int = 128, dropout_rate: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, embedding_dim) * 0.02)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B, Seq_Len, D]
        x = x + self.pos_embedding
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module if HAS_TORCH else object):
    """Khối Multi-Head Self-Attention tiêu chuẩn."""
    def __init__(self, dim: int = 128, num_heads: int = 4, qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderBlock(nn.Module if HAS_TORCH else object):
    """Khối Transformer Encoder gồm LayerNorm + MHSA + LayerNorm + MLP."""
    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SequencePooling(nn.Module if HAS_TORCH else object):
    """
    Sequence Pooling: Chiếu chuỗi spatial tokens thành số lượng slots tương ứng (10 slots).
    Kết hợp cơ chế Attention Pooling học sâu với Spatial Horizontal Projection nhằm đảm bảo
    định hướng không gian từ trái sang phải khớp với thứ tự ký tự biển số xe.
    """
    def __init__(self, in_features: int = 128, num_slots: int = 10, h_prime: int = 16, w_prime: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.h_prime = h_prime
        self.w_prime = w_prime
        self.attention_pool = nn.Linear(in_features, num_slots)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        # 1. Cơ chế Attention Pooling
        weights = self.softmax(self.attention_pool(x)) # [B, N, num_slots]
        attn_slots = torch.bmm(weights.transpose(1, 2), x) # [B, num_slots, D]

        # 2. Chiếu không gian ngang (Spatial Horizontal Projection)
        if N == self.h_prime * self.w_prime:
            x_grid = x.view(B, self.h_prime, self.w_prime, D).mean(dim=1) # [B, W', D]
            spatial_slots = F.adaptive_avg_pool1d(x_grid.transpose(1, 2), self.num_slots).transpose(1, 2)
            slots = spatial_slots + self.alpha * attn_slots
        else:
            slots = attn_slots

        return slots


class CCTPlateVietnamModel(nn.Module if HAS_TORCH else object):
    """
    Mô hình Compact Convolutional Transformer hoàn chỉnh cho biển số xe Việt Nam:
    - Input: Tensor dải ảnh [B, 3, 64, 256] sau khi qua Module Nắn phối cảnh Bước 2.2
    - Output: Logits [B, 10, 33] tương ứng với 10 slot ký tự và 33 class trong alphabet.
    """

    def __init__(
        self,
        config_path: str = "configs/plate_config_vietnam.yaml",
    ):
        super().__init__()
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Thông số từ config
        self.max_plate_slots = self.config.get("max_plate_slots", 10)
        self.alphabet = self.config.get("alphabet", "0123456789ABCDEFGHKLMNPQRSTUVXYZ_")
        self.pad_char = self.config.get("pad_char", "_")
        self.num_classes = len(self.alphabet)

        cct_cfg = self.config.get("cct_model", {})
        self.embedding_dim = cct_cfg.get("embedding_dim", 128)

        # 1. Convolutional Tokenizer
        self.tokenizer = ConvTokenizer(
            in_channels=self.config.get("channels", 3),
            embedding_dim=self.embedding_dim,
        )

        # 2. Sequence Pooling chiếu thành max_plate_slots
        self.seq_pool = SequencePooling(
            in_features=self.embedding_dim,
            num_slots=self.max_plate_slots,
        )

        # 3. Learnable Positional Embedding cho 10 vị trí
        self.pos_embed = PositionalEmbedding(
            sequence_length=self.max_plate_slots,
            embedding_dim=self.embedding_dim,
            dropout_rate=0.1,
        )

        # 4. Transformer Encoder Blocks (4 tầng)
        enc_cfg = cct_cfg.get("transformer_encoder", {})
        num_layers = enc_cfg.get("num_layers", 4)
        num_heads = enc_cfg.get("num_heads", 4)
        mlp_ratio = enc_cfg.get("mlp_ratio", 2.0)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=self.embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(self.embedding_dim)

        # 5. Multi-Head Classifier: Chiếu D -> 33 classes trên từng slot
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def forward(self, x):
        # x: [B, 3, 64, 256]
        # 1. Tách patch tokens: [B, 1024, D]
        tokens = self.tokenizer(x)

        # 2. Sequence pooling về đúng 10 slot: [B, 10, D]
        slots = self.seq_pool(tokens)

        # 3. Cộng Positional Embedding để giữ thứ tự vị trí trái qua phải
        x_seq = self.pos_embed(slots)

        # 4. Truyền qua các khối Transformer Encoder
        for block in self.encoder_blocks:
            x_seq = block(x_seq)
        x_seq = self.norm(x_seq)

        # 5. Chiếu qua Classifier: [B, 10, 33]
        logits = self.classifier(x_seq)
        return logits

    def decode(
        self,
        logits,
        use_grammar_beam_search: bool = False,
        beam_width: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Giải mã tensor logits [B, 10, 33] thành chuỗi biển số text hoàn chỉnh:
        - Greedy Decoding (nếu use_grammar_beam_search=False)
        - Grammar-Guided Beam Search (nếu use_grammar_beam_search=True, Bước 4.1):
          Áp dụng Probability Masking và DFA để triệt tiêu các lỗi nhầm lẫn quang học.
        """
        if not HAS_TORCH:
            return []

        if use_grammar_beam_search:
            from src.decoder.beam_search import GrammarGuidedBeamSearchDecoder
            decoder = GrammarGuidedBeamSearchDecoder(
                alphabet=self.alphabet,
                pad_char=self.pad_char,
                max_slots=self.max_plate_slots,
            )
            return decoder.decode_batch(logits, beam_width=beam_width, apply_grammar=True)

        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        batch_results = []
        batch_size = logits.shape[0]

        for b in range(batch_size):
            pred_indices = predictions[b].cpu().numpy()
            slot_confs = confidences[b].cpu().numpy()

            raw_chars = []
            valid_confs = []

            for idx, conf in zip(pred_indices, slot_confs):
                ch = self.alphabet[idx]
                if ch != self.pad_char:
                    raw_chars.append(ch)
                    valid_confs.append(float(conf))

            clean_text = "".join(raw_chars)
            mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

            batch_results.append({
                "text": clean_text,
                "confidence": round(mean_conf, 4),
                "raw_slots": [self.alphabet[i] for i in pred_indices],
                "slot_confidences": [round(float(c), 3) for c in slot_confs],
            })

        return batch_results
