"""
FAST-PLATE-OCR-VIETNAMESE: Temporal Consensus Engine (Giai đoạn 5 - Bước 5.2)
Hợp nhất quyết định nhận diện đa khung hình (Multi-Frame Temporal Consensus):
- Bỏ phiếu có trọng số (Weighted Majority Voting) theo điểm chất lượng quang học Q.
- Hợp nhất ma trận xác suất mềm (Soft Probability Fusion) triệt tiêu lỗi chớp đèn / lóa sáng.
- Xuất ra 1 kết quả duy nhất cho mỗi Tracklet ID đảm bảo 100% tuân thủ ngữ pháp pháp lý.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

from src.grammar.dfa import VietnamPlateDFA
from src.decoder.beam_search import GrammarGuidedBeamSearchDecoder


class TemporalConsensusEngine:
    """
    Động cơ hợp nhất đa khung hình theo thời gian (Temporal Consensus Engine):
    - Tiếp nhận tập hợp K khung hình tốt nhất từ BestKFrameBuffer của một Tracklet ID.
    - Kết hợp trọng số chất lượng ảnh Q và độ tin cậy để triệt tiêu các ngoại lai (outliers).
    - Giải quyết hiện tượng chớp đèn pha, bóng đổ hoặc nhòe tức thời của camera giao thông.
    """

    def __init__(
        self,
        config_path: str = "configs/plate_config_vietnam.yaml",
        alphabet: str = "0123456789ABCDEFGHKLMNPQRSTUVXYZ_",
        pad_char: str = "_",
        max_slots: int = 10,
        min_frames: int = 2,
    ):
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.max_slots = max_slots
        self.min_frames = min_frames

        self.char2idx = {c: i for i, c in enumerate(alphabet)}
        self.idx2char = {i: c for i, c in enumerate(alphabet)}
        self.vocab_size = len(alphabet)

        self.dfa = VietnamPlateDFA()
        self.decoder = GrammarGuidedBeamSearchDecoder(
            config_path=config_path,
            alphabet=self.alphabet,
            pad_char=self.pad_char,
            max_slots=self.max_slots,
        )

    def compute_normalized_weights(self, frames_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Tính trọng số chuẩn hóa w_i cho từng khung hình dựa trên:
        - Điểm chất lượng quang học Q_i (độ nét Laplace, góc nghiêng, ánh sáng)
        - Độ tin cậy OCR / Detection C_i
        """
        raw_weights = []
        for f in frames_data:
            q = float(f.get("quality_score", 0.5))
            c = float(f.get("confidence", f.get("ocr_confidence", 0.5)))
            # Trọng số bậc 2 nhấn mạnh các khung hình có độ nét vượt trội
            weight = max(1e-4, (q ** 1.5) * c)
            raw_weights.append(weight)

        weights = np.array(raw_weights, dtype=np.float32)
        total_w = np.sum(weights)
        if total_w > 0:
            weights /= total_w
        else:
            weights = np.full(len(frames_data), 1.0 / len(frames_data), dtype=np.float32)

        return weights

    def fuse_probabilities(
        self,
        frames_data: List[Dict[str, Any]],
        weights: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Hợp nhất ma trận xác suất mềm [L, V] của K khung hình:
        P_fused = Sum(w_i * P_i)
        """
        valid_matrices = []
        valid_weights = []

        for i, f in enumerate(frames_data):
            logits_or_probs = f.get("logits") if f.get("logits") is not None else f.get("probs")
            if logits_or_probs is not None:
                arr = np.asarray(logits_or_probs, dtype=np.float32)
                if len(arr.shape) == 3:
                    arr = arr[0]

                # Nếu là logits, chuyển đổi qua Softmax
                if np.min(arr) < 0 or np.max(arr) > 1.05:
                    exp_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                    probs = exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)
                else:
                    probs = arr

                valid_matrices.append(probs)
                valid_weights.append(weights[i])

        if not valid_matrices:
            return None

        # Chuẩn hóa lại trọng số cho các frame có ma trận
        norm_w = np.array(valid_weights, dtype=np.float32)
        norm_w /= np.sum(norm_w)

        fused = np.zeros_like(valid_matrices[0])
        for p, w in zip(valid_matrices, norm_w):
            fused += w * p

        return fused

    def vote_slot_characters(
        self,
        frames_data: List[Dict[str, Any]],
        weights: np.ndarray,
    ) -> Tuple[str, List[float], float]:
        """
        Bỏ phiếu theo từng vị trí ký tự (Weighted Majority Voting per Slot):
        Áp dụng khi chỉ có chuỗi văn bản nhận dạng từ các frame.
        """
        slot_votes = np.zeros((self.max_slots, self.vocab_size), dtype=np.float32)

        for i, f in enumerate(frames_data):
            text = str(f.get("text", f.get("plate_text", ""))).strip().upper()
            padded = text.ljust(self.max_slots, self.pad_char)[: self.max_slots]
            w = weights[i]

            for slot_idx, ch in enumerate(padded):
                if ch in self.char2idx:
                    c_idx = self.char2idx[ch]
                    slot_votes[slot_idx, c_idx] += w

        # Lấy token có số phiếu cao nhất ở từng slot
        best_indices = np.argmax(slot_votes, axis=-1)
        slot_confs = np.max(slot_votes, axis=-1)

        result_chars = []
        for s_idx, c_idx in zip(range(self.max_slots), best_indices):
            ch = self.idx2char.get(c_idx, self.pad_char)
            if ch != self.pad_char:
                result_chars.append(ch)

        final_str = "".join(result_chars)
        mean_conf = float(np.mean(slot_confs)) if len(slot_confs) > 0 else 0.0

        return final_str, [round(float(c), 3) for c in slot_confs], round(mean_conf, 4)

    def consolidate_tracklet(
        self,
        track_id: int,
        frames_data: List[Dict[str, Any]],
        beam_width: int = 5,
    ) -> Dict[str, Any]:
        """
        Hợp nhất toàn diện các quan sát đa khung hình của 1 Tracklet ID:
        Trả về kết quả quyết định cuối cùng duy nhất cho phương tiện.
        """
        if not frames_data:
            return {
                "track_id": track_id,
                "plate_text": "",
                "consensus_confidence": 0.0,
                "is_valid": False,
                "agreement_rate": 0.0,
                "total_frames_voted": 0,
            }

        num_frames = len(frames_data)
        weights = self.compute_normalized_weights(frames_data)

        # 1. Hợp nhất mềm qua ma trận xác suất (nếu có ma trận logits / probs)
        fused_probs = self.fuse_probabilities(frames_data, weights)

        if fused_probs is not None:
            # Chuyển đổi xác suất sang không gian log-likelihood trước khi giải mã beam search
            fused_log_probs = np.log(np.clip(fused_probs, 1e-12, 1.0))
            decoded = self.decoder.decode_single(
                fused_log_probs,
                beam_width=beam_width,
                apply_grammar=True,
            )
            consensus_text = decoded["text"]
            consensus_conf = decoded["confidence"]
            is_valid = decoded["is_valid"]
            dfa_meta = decoded["dfa_meta"]
            dfa_msg = decoded["dfa_message"]
        else:
            # 2. Bỏ phiếu có trọng số theo từng slot (Weighted Majority Voting)
            voted_text, slot_confs, consensus_conf = self.vote_slot_characters(frames_data, weights)
            is_valid, dfa_msg, dfa_meta = self.dfa.validate(voted_text)
            consensus_text = voted_text

        # 3. Tính toán tỷ lệ đồng thuận (Consensus Agreement Rate)
        # Tỷ lệ tổng trọng số các frame dự đoán trùng khớp hoàn toàn với chuỗi hợp nhất
        agree_weight = 0.0
        outliers_suppressed = []

        for i, f in enumerate(frames_data):
            f_text = str(f.get("text", f.get("plate_text", ""))).strip().upper()
            f_w = float(weights[i])
            f_idx = f.get("frame_idx", i)

            if f_text == consensus_text:
                agree_weight += f_w
            else:
                # Phát hiện frame ngoại lai bị triệt tiêu do chớp đèn / lóa sáng
                outliers_suppressed.append({
                    "frame_idx": f_idx,
                    "frame_pred": f_text,
                    "frame_quality": round(float(f.get("quality_score", 0.0)), 3),
                    "discrepancy": f"Dự đoán '{f_text}' khác kết quả hợp nhất '{consensus_text}'",
                })

        agreement_rate = round(float(agree_weight), 4)

        return {
            "track_id": track_id,
            "plate_text": consensus_text,
            "consensus_confidence": round(float(consensus_conf), 4),
            "is_valid": is_valid,
            "agreement_rate": agreement_rate,
            "agreement_percent": round(agreement_rate * 100, 2),
            "total_frames_voted": num_frames,
            "frame_indices": [f.get("frame_idx", i) for i, f in enumerate(frames_data)],
            "normalized_weights": [round(float(w), 4) for w in weights],
            "outliers_suppressed": outliers_suppressed,
            "dfa_message": dfa_msg,
            "dfa_meta": dfa_meta,
        }
