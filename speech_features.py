"""
Module: speech_features.py
Chức năng: Trích xuất đặc trưng âm thanh theo chuẩn Lab 3:
- Đọc file WAV, mono, 16 kHz
- Cắt silence bằng VAD / librosa.effects.trim (top_db=30)
- Pre-emphasis lọc tăng tần số cao (coef=0.97)
- 13 MFCC tĩnh + 13 Delta (d1) + 13 Delta-Delta (d2) -> 39 chiều (T x 39)
- Chuẩn hóa trung bình và phương sai phổ CMVN (Utterance-level)
- Trực quan hóa Heatmap so sánh đặc trưng giữa các từ
"""

import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def extract_features(
    path: str,
    sr: int = 16000,
    top_db: float = 30.0,
    preemph_coef: float = 0.97,
    n_mfcc: int = 13,
    n_mels: int = 26,
    n_fft: int = 512,
    win_sec: float = 0.025,
    hop_sec: float = 0.010,
    use_cmvn: bool = True
) -> np.ndarray:
    """
    Trích xuất ma trận đặc trưng 39 chiều (T x 39) từ file âm thanh:
    [MFCC(13), Delta(13), Delta-Delta(13)] kết hợp CMVN.
    """
    # 1. Đọc tín hiệu âm thanh mono với tần số lấy mẫu sr=16000
    y, sr = librosa.load(path, sr=sr, mono=True)

    # 2. Loại bỏ khoảng lặng ở đầu và cuối (VAD/Trim)
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trimmed) >= int(0.05 * sr):
        y = y_trimmed

    # 3. Pre-emphasis: nâng cao các dải tần số cao
    y = librosa.effects.preemphasis(y, coef=preemph_coef)

    # 4. Trích xuất 13 hệ số MFCC tĩnh
    win_length = int(win_sec * sr)   # 25ms = 400 mẫu
    hop_length = int(hop_sec * sr)   # 10ms = 160 mẫu
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window='hamming',
        n_mels=n_mels,
        fmin=20,
        fmax=sr / 2
    )

    # 5. Tính đạo hàm bậc 1 (Delta) và bậc 2 (Delta-Delta)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Ghép thành ma trận T x 39: [c_t, Delta c_t, Delta^2 c_t]
    features = np.vstack([mfcc, delta1, delta2]).T  # Shape: (T, 39)

    # 6. Utterance-level CMVN (Cepstral Mean and Variance Normalization)
    if use_cmvn:
        mu = features.mean(axis=0, keepdims=True)
        sd = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mu) / sd

    return features


def plot_feature_heatmaps(
    file_path1: str,
    label1: str,
    file_path2: str,
    label2: str,
    save_path: str = None
):
    """
    Vẽ heatmap so sánh đặc trưng MFCC + Delta + Delta^2 giữa 2 từ lệnh khác nhau.
    """
    feat1 = extract_features(file_path1)
    feat2 = extract_features(file_path2)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)

    im1 = axes[0].imshow(feat1.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"Ma trận đặc trưng 39-D (CMVN): Từ '{label1}' (Frames: {feat1.shape[0]})")
    axes[0].set_ylabel("Chỉ số đặc trưng (0-12: MFCC, 13-25: Δ, 26-38: Δ²)")
    fig.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)

    im2 = axes[1].imshow(feat2.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f"Ma trận đặc trưng 39-D (CMVN): Từ '{label2}' (Frames: {feat2.shape[0]})")
    axes[1].set_xlabel("Chỉ số Frame thời gian (10ms / frame)")
    axes[1].set_ylabel("Chỉ số đặc trưng (0-38)")
    fig.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[ĐÃ LƯU HÌNH ẢNH] Heatmap đặc trưng lưu tại: '{save_path}'")
    plt.close()
