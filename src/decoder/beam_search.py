"""
FAST-PLATE-OCR-VIETNAMESE: Grammar-Guided Beam Search Decoder (Giai đoạn 4 - Bước 4.1)
Thuật toán giải mã Beam Search có tích hợp mặt nạ xác suất ngữ pháp (Probability Masking).
Khử triệt để các lỗi nhầm lẫn quang học (8 <-> B, 0 <-> D, 5 <-> S, 2 <-> Z, 1 <-> T)
dựa trên Máy trạng thái hữu hạn DFA quy chuẩn Việt Nam.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from src.decoder.masking import DFAPositionMasker, OPTICAL_CONFUSION_PAIRS
from src.grammar.dfa import VietnamPlateDFA


class GrammarGuidedBeamSearchDecoder:
    """
    Bộ giải mã Beam Search ràng buộc ngữ pháp (Grammar-Guided Beam Search):
    - Khởi tạo beam search với độ rộng chùm k (beam_width).
    - Tại mỗi bước t trong chuỗi ký tự, áp dụng mặt nạ xác suất (Log-Probability Masking)
      triệt tiêu xác suất của các ký tự vi phạm cấu trúc cú pháp của biển số xe Việt Nam.
    - So sánh với Greedy Decoder để phát hiện và báo cáo các xung đột quang học đã được nắn chỉnh.
    - Kiểm định 100% kết quả xuất ra qua máy trạng thái DFA.
    """

    def __init__(
        self,
        config_path: str = "configs/plate_config_vietnam.yaml",
        alphabet: str = "0123456789ABCDEFGHKLMNPQRSTUVXYZ_",
        pad_char: str = "_",
        max_slots: int = 10,
    ):
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.max_slots = max_slots
        self.char2idx = {c: i for i, c in enumerate(alphabet)}
        self.idx2char = {i: c for i, c in enumerate(alphabet)}

        self.masker = DFAPositionMasker(
            alphabet=self.alphabet,
            pad_char=self.pad_char,
            max_slots=self.max_slots,
        )
        self.dfa = VietnamPlateDFA()

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        """Tính log-softmax số học ổn định."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return np.log(e_x / np.sum(e_x, axis=-1, keepdims=True) + 1e-12)

    def decode_single(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        beam_width: int = 5,
        apply_grammar: bool = True,
    ) -> Dict[str, Any]:
        """
        Giải mã 1 ma trận logits kích thước [L, V] (10 slots x 33 classes).
        Trả về kết quả giải mã tối ưu, độ tin cậy và báo cáo nắn chỉnh quang học.
        """
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().float().numpy()
        else:
            logits_np = np.asarray(logits, dtype=np.float32)

        # Đảm bảo shape [L, V]
        if len(logits_np.shape) == 3:
            logits_np = logits_np[0]

        seq_len, vocab_size = logits_np.shape

        # 1. Giải mã Greedy thô (không ràng buộc) để làm đối chứng
        raw_argmax_indices = np.argmax(logits_np, axis=-1)
        raw_greedy_chars = [self.idx2char.get(i, "") for i in raw_argmax_indices]
        raw_greedy_str = "".join(raw_greedy_chars).replace(self.pad_char, "")
        raw_is_valid, raw_msg, _ = self.dfa.validate(raw_greedy_str)

        # 2. Khởi tạo chùm Beam: danh sách tuple (chuỗi_prefix, điểm_log_prob_tích_lũy)
        beams: List[Tuple[str, float]] = [("", 0.0)]

        for step in range(seq_len):
            step_logits = logits_np[step]
            step_log_probs = self._log_softmax(step_logits)

            candidates: List[Tuple[str, float]] = []

            for prefix, cum_score in beams:
                # Áp dụng mặt nạ ngữ pháp nếu được kích hoạt
                if apply_grammar:
                    mask = self.masker.compute_log_mask(prefix, step)
                    masked_probs = step_log_probs + mask
                else:
                    masked_probs = step_log_probs

                # Lấy top k ký tự có xác suất cao nhất trong tập hợp hợp lệ
                # Lọc bỏ các token có điểm log_prob cực âm do bị mask
                valid_indices = np.where(masked_probs > -1e8)[0]
                if len(valid_indices) == 0:
                    # Fallback dự phòng nếu toàn bộ bị mask
                    valid_indices = np.argsort(step_log_probs)[-beam_width:]

                # Sắp xếp các chỉ số hợp lệ theo điểm số giảm dần
                sorted_valid = valid_indices[np.argsort(-masked_probs[valid_indices])][:beam_width]

                for token_idx in sorted_valid:
                    char = self.idx2char[token_idx]
                    token_score = masked_probs[token_idx]
                    candidates.append((prefix + char, cum_score + float(token_score)))

            # Sắp xếp toàn bộ candidate và cắt tỉa giữ lại top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        # 3. Hậu xử lý & Đánh giá DFA trên các giả thuyết tốt nhất
        hypotheses: List[Dict[str, Any]] = []
        corrections_detected: List[Dict[str, Any]] = []

        for final_seq, final_score in beams:
            cleaned_text = final_seq.replace(self.pad_char, "")
            is_valid, msg, meta = self.dfa.validate(cleaned_text)
            
            # Tính điểm tin cậy trung bình [0.0, 1.0]
            # normalized exp(score / seq_len)
            conf = float(np.exp(final_score / float(seq_len)))
            conf = min(1.0, max(0.0, conf))

            hypotheses.append({
                "text": cleaned_text,
                "full_padded": final_seq,
                "log_score": round(final_score, 4),
                "confidence": round(conf, 4),
                "is_valid": is_valid,
                "dfa_message": msg,
                "dfa_meta": meta,
            })

        # Ưu tiên chọn giả thuyết có điểm cao nhất và hợp lệ DFA nếu apply_grammar=True
        if apply_grammar:
            valid_hyps = [h for h in hypotheses if h["is_valid"]]
            best_hyp = valid_hyps[0] if len(valid_hyps) > 0 else hypotheses[0]
        else:
            best_hyp = hypotheses[0]

        # 4. Phát hiện các xung đột quang học đã được nắn chỉnh thành công
        best_padded = best_hyp["full_padded"]
        for slot_idx in range(seq_len):
            greedy_char = raw_greedy_chars[slot_idx]
            guided_char = best_padded[slot_idx]
            if greedy_char != guided_char:
                is_optical_pair = (
                    greedy_char in OPTICAL_CONFUSION_PAIRS
                    and OPTICAL_CONFUSION_PAIRS.get(greedy_char) == guided_char
                )
                corrections_detected.append({
                    "slot": slot_idx,
                    "greedy_raw": greedy_char,
                    "grammar_guided": guided_char,
                    "is_optical_confusion_pair": is_optical_pair,
                    "confusion_type": f"{greedy_char} -> {guided_char}" if is_optical_pair else "GRAMMAR_PRUNED",
                })

        return {
            "text": best_hyp["text"],
            "confidence": best_hyp["confidence"],
            "is_valid": best_hyp["is_valid"],
            "dfa_message": best_hyp["dfa_message"],
            "dfa_meta": best_hyp["dfa_meta"],
            "raw_greedy_text": raw_greedy_str,
            "raw_greedy_is_valid": raw_is_valid,
            "raw_greedy_msg": raw_msg,
            "corrections_made": corrections_detected,
            "all_hypotheses": hypotheses,
        }

    def decode_batch(
        self,
        batch_logits: Union[np.ndarray, torch.Tensor],
        beam_width: int = 5,
        apply_grammar: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Giải mã hàng loạt theo batch [B, L, V].
        """
        if isinstance(batch_logits, torch.Tensor):
            batch_np = batch_logits.detach().cpu().float().numpy()
        else:
            batch_np = np.asarray(batch_logits, dtype=np.float32)

        results: List[Dict[str, Any]] = []
        for i in range(batch_np.shape[0]):
            res = self.decode_single(
                batch_np[i],
                beam_width=beam_width,
                apply_grammar=apply_grammar,
            )
            results.append(res)

        return results
