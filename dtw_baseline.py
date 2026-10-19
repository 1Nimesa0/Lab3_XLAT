"""
Module: dtw_baseline.py
Chức năng: Cài đặt hệ nhận dạng từ tách rời bằng đối sánh mẫu Dynamic Time Warping (DTW)
để làm baseline so sánh với HMM theo yêu cầu Mục 4.5 & 5.8 của Lab 3.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from speech_features import extract_features

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def compute_dtw_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Tính khoảng cách DTW chuẩn hóa theo độ dài đường đi giữa 2 chuỗi đặc trưng.
    - Ma trận khoảng cách cục bộ dùng Euclidean distance.
    - Quy hoạch động tìm đường chi phí nhỏ nhất:
      D(i, j) = dist(i, j) + min(D(i-1, j), D(i, j-1), D(i-1, j-1))
    """
    N, M = len(feat1), len(feat2)
    dist_matrix = cdist(feat1, feat2, metric='euclidean')

    # Khởi tạo ma trận chi phí tích lũy
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = dist_matrix[i - 1, j - 1]
            D[i, j] = cost + min(D[i - 1, j],      # insertion
                                 D[i, j - 1],      # deletion
                                 D[i - 1, j - 1])  # match

    # Chuẩn hóa khoảng cách theo tổng số frame (N + M) để công bằng giữa các độ dài
    normalized_dist = D[N, M] / (N + M)
    return normalized_dist


class DTWRecognizer:
    """
    Hệ nhận dạng dựa trên mẫu (Template-based Recognizer) bằng DTW.
    Lưu các vector đặc trưng mẫu từ tập huấn luyện (training templates).
    """
    def __init__(self):
        self.templates: Dict[str, List[np.ndarray]] = {}

    def fit(self, train_dir: str, labels: List[str]):
        """Nạp các mẫu huấn luyện cho từng lớp."""
        self.templates = {label: [] for label in labels}
        for label in labels:
            label_dir = Path(train_dir) / label
            for wav in sorted(label_dir.glob("*.wav")):
                feat = extract_features(str(wav))
                self.templates[label].append(feat)

    def predict(self, test_feat: np.ndarray) -> Tuple[str, float]:
        """
        Nhận dạng 1 mẫu thử: tìm mẫu template trong kho có khoảng cách DTW nhỏ nhất.
        """
        best_label = None
        min_dist = float("inf")

        for label, template_list in self.templates.items():
            for tmpl in template_list:
                d = compute_dtw_distance(test_feat, tmpl)
                if d < min_dist:
                    min_dist = d
                    best_label = label

        return best_label, min_dist


def evaluate_dtw(
    test_dir: str,
    dtw_model: DTWRecognizer,
    labels: List[str]
) -> Tuple[float, float]:
    """
    Đánh giá độ chính xác và thời gian suy diễn trung bình của bộ nhận dạng DTW.
    """
    total = 0
    correct = 0
    total_time = 0.0

    for label in labels:
        test_files = sorted(list((Path(test_dir) / label).glob("*.wav")))
        for wav in test_files:
            feat = extract_features(str(wav))
            t0 = time.time()
            pred, _ = dtw_model.predict(feat)
            t1 = time.time()

            total_time += (t1 - t0)
            total += 1
            if pred == label:
                correct += 1

    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    avg_time_per_trial = (total_time / total) if total > 0 else 0.0
    return accuracy, avg_time_per_trial
