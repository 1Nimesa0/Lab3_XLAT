"""
Module: hmm_model.py
Chức năng: Quản lý mô hình Gaussian HMM cho nhận dạng tiếng nói:
- Khởi tạo HMM trái-sang-phải (Left-to-Right / Bakis topology)
- Phân đoạn thời gian đều (Uniform temporal segmental initialization) chuẩn âm thanh
- Huấn luyện Whole-Word HMM bằng thuật toán Baum-Welch (EM)
- Tính toán Acoustic Likelihood và giải mã nhận dạng Maximum Likelihood
- Giải mã chuỗi trạng thái tối ưu bằng thuật toán Viterbi (State alignment)
- Lưu và tải mô hình đã huấn luyện (.pkl)
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from speech_features import extract_features

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def make_left_to_right_hmm(
    n_states: int = 5,
    n_iter: int = 80,
    tol: float = 1e-3,
    min_covar: float = 1e-3,
    random_state: int = 42,
    params: str = "mc"
) -> GaussianHMM:
    """
    Tạo cấu trúc HMM trái-sang-phải (Left-to-Right topology).
    - π = [1, 0, ..., 0]: Luôn bắt đầu ở trạng thái đầu tiên q1.
    - Ma trận chuyển A chỉ cho phép self-loop (ở lại) hoặc tiến sang trạng thái kế tiếp.
    - State cuối cùng chỉ có self-loop (absorbing state / a_{N,N} = 1.0).
    - params='mc': Cố định cấu trúc chuyển trạng thái Left-to-Right theo lưu ý Mục 5.3 & 8 của Lab.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        tol=tol,
        min_covar=min_covar,
        init_params="",      # Tự khởi tạo tham số chuyên dụng cho âm thanh
        params=params,       # Mặc định 'mc' để bảo toàn tuyệt đối Left-to-Right topology
        random_state=random_state
    )

    # Khởi tạo xác suất bắt đầu π: luôn bắt đầu ở state 0
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    model.startprob_ = startprob

    # Khởi tạo ma trận chuyển trạng thái A:
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transmat[i, i] = 0.70        # Xác suất ở lại trạng thái hiện tại (self-loop)
        transmat[i, i + 1] = 0.30    # Xác suất chuyển sang trạng thái kế tiếp
    transmat[-1, -1] = 1.0          # Trạng thái cuối cùng
    model.transmat_ = transmat

    return model


def init_speech_segmental_means(
    features_list: List[np.ndarray],
    n_states: int = 5,
    n_features: int = 39
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Khởi tạo kỳ vọng (means) và hiệp phương sai (covars) bằng phân đoạn thời gian đều
    (Uniform temporal segmentation / Segmental K-means) - chuẩn kinh điển cho Speech HMM.
    Mỗi utterance được chia làm n_states đoạn bằng nhau theo tiến trình thời gian.
    """
    means = []
    for s in range(n_states):
        state_frames = []
        for feat in features_list:
            T = len(feat)
            t_start = int(s * T / n_states)
            t_end = int((s + 1) * T / n_states)
            if t_end <= t_start:
                t_end = t_start + 1
            state_frames.append(feat[t_start:t_end])
        stacked_frames = np.vstack(state_frames)
        means.append(stacked_frames.mean(axis=0))

    means = np.array(means)
    covars = np.ones((n_states, n_features))  # Sau CMVN, phương sai chuẩn xấp xỉ 1.0
    return means, covars


def train_word_hmms(
    train_dir: str,
    labels: List[str],
    n_states: int = 5,
    n_iter: int = 80
) -> Dict[str, GaussianHMM]:
    """
    Huấn luyện một Whole-word Gaussian HMM cho mỗi từ lệnh trong danh sách.
    Nối các utterance của cùng một lớp thành ma trận X_concat và truyền lengths.
    """
    models = {}
    print(f"\n[BẮT ĐẦU HUẤN LUYỆN HMM] Số trạng thái: {n_states} | Lặp tối đa: {n_iter}")
    print("=" * 65)

    for label in labels:
        label_dir = Path(train_dir) / label
        wav_files = sorted(list(label_dir.glob("*.wav")))

        if len(wav_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy file WAV nào trong thư mục: {label_dir}")

        features_list = []
        lengths = []

        for wav in wav_files:
            feat = extract_features(str(wav))
            features_list.append(feat)
            lengths.append(len(feat))

        # Nối tất cả các vector đặc trưng của các utterance thuộc cùng 1 nhãn
        X_concat = np.vstack(features_list)

        # Khởi tạo mô hình Left-to-Right
        model = make_left_to_right_hmm(n_states=n_states, n_iter=n_iter)

        # Gán giá trị khởi tạo means và covars phân đoạn theo thời gian
        means, covars = init_speech_segmental_means(features_list, n_states=n_states, n_features=X_concat.shape[1])
        model.means_ = means
        model.covars_ = covars

        # Huấn luyện bằng thuật toán Baum-Welch (EM)
        model.fit(X_concat, lengths)

        models[label] = model
        converged = getattr(model.monitor_, 'converged', False)
        print(f" • Từ: '{label:6s}' | File train: {len(wav_files):2d} | Tổng frames: {sum(lengths):4d} | Hội tụ: {converged}")

    print("=" * 65)
    print("[HOÀN THÀNH HUẤN LUYỆN] Tất cả các mô hình đã sẵn sàng.")
    return models


def recognize_single_utterance(
    features: np.ndarray,
    models: Dict[str, GaussianHMM]
) -> Tuple[str, Dict[str, float], float]:
    """
    Nhận dạng 1 mẫu âm thanh bằng quy tắc Maximum Likelihood:
    ŵ = argmax_w log P(X | λ_w)
    Trả về:
    - Nhãn dự đoán tốt nhất (Top-1)
    - Dictionary điểm log-likelihood của từng model
    - Khoảng cách điểm giữa Top-1 và Top-2 (Confidence Margin)
    """
    scores = {}
    for label, model in models.items():
        try:
            score = model.score(features)
        except Exception:
            score = -1e9
        scores[label] = score

    # Sắp xếp các giả thuyết theo log-likelihood giảm dần
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    pred_label = sorted_scores[0][0]

    # Tính khoảng cách giữa top 1 và top 2
    margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    return pred_label, scores, margin


def get_viterbi_alignment(
    features: np.ndarray,
    model: GaussianHMM
) -> Tuple[float, np.ndarray]:
    """
    Thực hiện thuật toán Viterbi tìm chuỗi trạng thái tốt nhất (Viterbi state path).
    """
    log_prob, state_sequence = model.decode(features, algorithm="viterbi")
    return log_prob, state_sequence


def plot_viterbi_trellis(
    state_sequence: np.ndarray,
    title: str = "Đường trạng thái Viterbi (State Alignment)",
    save_path: str = None
):
    """
    Vẽ đồ thị bước (step plot) thể hiện tiến trình trạng thái theo từng Frame.
    """
    plt.figure(figsize=(9, 4.5))
    frames = np.arange(len(state_sequence))
    plt.step(frames, state_sequence, where="post", color="navy", linewidth=2.2, label="Viterbi Path")
    plt.scatter(frames, state_sequence, color="crimson", s=18, zorder=3)

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Chỉ số Frame (10ms/frame)", fontsize=10)
    plt.ylabel("Chỉ số Trạng thái HMM (State Index)", fontsize=10)
    plt.yticks(np.arange(np.max(state_sequence) + 1), [f"q{i+1}" for i in range(np.max(state_sequence) + 1)])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[ĐÃ LƯU HÌNH ẢNH] Viterbi path lưu tại: '{save_path}'")
    plt.close()


def save_models(models: Dict[str, GaussianHMM], save_path: str):
    """Lưu dict các mô hình HMM ra file pickle."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(models, f)
    print(f"[ĐÃ LƯU MÔ HÌNH] Danh sách model HMM lưu tại: '{save_path}'")


def load_models(load_path: str) -> Dict[str, GaussianHMM]:
    """Tải dict các mô hình HMM từ file pickle."""
    with open(load_path, "rb") as f:
        models = pickle.load(f)
    print(f"[ĐÃ TẢI MÔ HÌNH] Đã tải {len(models)} mô hình từ: '{load_path}'")
    return models
