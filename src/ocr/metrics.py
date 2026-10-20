"""
FAST-PLATE-OCR-VIETNAMESE: Evaluation Metrics
Hàm tính toán khoảng cách Levenshtein, Character Error Rate (CER),
và Full Match Accuracy (FMA) cho nhận dạng ký tự biển số xe.
"""

from typing import List, Dict, Any


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Tính khoảng cách hiệu chỉnh Levenshtein giữa 2 chuỗi ký tự.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Xóa (deletion)
                    dp[i][j - 1],      # Chèn (insertion)
                    dp[i - 1][j - 1]   # Thay thế (substitution)
                )

    return dp[m][n]


def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Tính Character Error Rate (CER):
    CER = Tổng khoảng cách Levenshtein / Tổng số ký tự tham chiếu thực tế.
    """
    if not ground_truths or len(ground_truths) == 0:
        return 0.0

    total_edit_distance = 0
    total_gt_chars = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.strip()
        gt_clean = gt.strip()
        dist = levenshtein_distance(pred_clean, gt_clean)
        total_edit_distance += dist
        total_gt_chars += max(1, len(gt_clean))

    return total_edit_distance / max(1, total_gt_chars)


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Tính Full Match Accuracy (FMA) / Exact Match (EM):
    Tỷ lệ biển số nhận dạng chính xác hoàn toàn 100% từng ký tự.
    """
    if not ground_truths or len(ground_truths) == 0:
        return 0.0

    matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred.strip() == gt.strip())
    return matches / len(ground_truths)


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Đánh giá toàn diện bộ dự đoán.
    """
    cer = compute_cer(predictions, ground_truths)
    acc = compute_accuracy(predictions, ground_truths)
    total_samples = len(ground_truths)
    matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())

    return {
        "cer": round(cer, 5),
        "cer_percent": round(cer * 100, 3),
        "accuracy": round(acc, 5),
        "accuracy_percent": round(acc * 100, 3),
        "total_samples": total_samples,
        "exact_matches": matches,
    }
