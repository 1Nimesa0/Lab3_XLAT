"""
Module: eval_utils.py
Chức năng: Đánh giá mô hình và thực hiện các thí nghiệm mở rộng:
- Tính ma trận nhầm lẫn (Confusion Matrix) và Accuracy
- Vẽ biểu đồ ma trận nhầm lẫn chuẩn trực quan
- Thí nghiệm Robustness: thử nghiệm với môi trường nhiễu âm (Gaussian Noise / SNR)
- Phân tích phân bố điểm số và thiết lập ngưỡng từ chối (Rejection Threshold) cho từ ngoài từ điển (OOV/Unknown)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from hmmlearn.hmm import GaussianHMM

from speech_features import extract_features
from hmm_model import recognize_single_utterance

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def evaluate_dataset(
    test_dir: str,
    models: Dict[str, GaussianHMM],
    labels: List[str]
) -> Tuple[float, List[str], List[str], List[float]]:
    """
    Đánh giá mô hình HMM trên toàn bộ tập kiểm thử.
    Trả về: (accuracy, y_true, y_pred, margins)
    """
    y_true = []
    y_pred = []
    margins = []

    for label in labels:
        label_dir = Path(test_dir) / label
        test_files = sorted(list(label_dir.glob("*.wav")))

        for wav in test_files:
            feat = extract_features(str(wav))
            pred, _, margin = recognize_single_utterance(feat, models)
            y_true.append(label)
            y_pred.append(pred)
            margins.append(margin)

    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
    accuracy = (correct / len(y_true)) * 100.0 if y_true else 0.0
    return accuracy, y_true, y_pred, margins


def plot_confusion_matrix_figure(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    title: str = "Confusion Matrix - Gaussian HMM",
    save_path: str = None
):
    """Vẽ và lưu ma trận nhầm lẫn sử dụng sklearn."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[ĐÃ LƯU HÌNH ẢNH] Confusion matrix lưu tại: '{save_path}'")
    plt.close()


def add_noise_to_audio(
    features: np.ndarray,
    noise_level: float = 0.2,
    random_state: int = 42
) -> np.ndarray:
    """Mô phỏng nhiễu môi trường bằng cách cộng thêm Gaussian Noise vào vector đặc trưng."""
    rng = np.random.RandomState(random_state)
    noise = rng.normal(0, noise_level, size=features.shape)
    return features + noise


def evaluate_robustness_with_noise(
    test_dir: str,
    models: Dict[str, GaussianHMM],
    labels: List[str],
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.4, 0.6]
) -> Dict[float, float]:
    """
    Đánh giá độ bền vững của hệ thống khi độ lớn của nhiễu tăng dần.
    """
    results = {}
    for nl in noise_levels:
        y_true = []
        y_pred = []
        for label in labels:
            test_files = sorted(list((Path(test_dir) / label).glob("*.wav")))
            for wav in test_files:
                feat = extract_features(str(wav))
                if nl > 0:
                    feat = add_noise_to_audio(feat, noise_level=nl)
                pred, _, _ = recognize_single_utterance(feat, models)
                y_true.append(label)
                y_pred.append(pred)

        acc = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp]) / len(y_true) * 100.0
        results[nl] = acc

    return results


def analyze_rejection_threshold(
    test_dir: str,
    models: Dict[str, GaussianHMM],
    labels: List[str],
    save_path: str = None
) -> float:
    """
    Phân tích phân bố điểm log-likelihood chuẩn hóa (score / T).
    Đề xuất ngưỡng từ chối (Reject Threshold) để phát hiện từ nằm ngoài tập từ vựng (Unknown/Out-of-Vocabulary).
    """
    correct_scores = []
    all_scores = []

    for label in labels:
        test_files = sorted(list((Path(test_dir) / label).glob("*.wav")))
        for wav in test_files:
            feat = extract_features(str(wav))
            T = len(feat)
            pred, scores, _ = recognize_single_utterance(feat, models)
            best_score_per_frame = scores[pred] / T
            all_scores.append(best_score_per_frame)
            if pred == label:
                correct_scores.append(best_score_per_frame)

    # Đề xuất ngưỡng: mean - 2 * std của các mẫu đúng
    if correct_scores:
        mu = np.mean(correct_scores)
        sigma = np.std(correct_scores)
        recommended_threshold = mu - 2.0 * sigma
    else:
        recommended_threshold = -150.0

    # Vẽ biểu đồ phân bố điểm số
    plt.figure(figsize=(8, 4.5))
    plt.hist(all_scores, bins=12, color="skyblue", edgecolor="black", alpha=0.7, label="Điểm số Test utterances")
    plt.axvline(recommended_threshold, color="crimson", linestyle="--", linewidth=2,
                label=f"Đề xuất ngưỡng Reject = {recommended_threshold:.2f}")

    plt.title("Phân bố điểm Acoustic Score / Frame và Ngưỡng Rejection", fontsize=11, fontweight="bold")
    plt.xlabel("Log-likelihood trung bình mỗi Frame (Score / T)", fontsize=10)
    plt.ylabel("Tần số (Số mẫu)", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[ĐÃ LƯU HÌNH ẢNH] Biểu đồ Rejection Threshold lưu tại: '{save_path}'")
    plt.close()

    return recommended_threshold
