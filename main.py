"""
===================================================================================
CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 3
CHƯƠNG TRÌNH CHÍNH: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI GAUSSIAN HMM
Bộ môn Trí tuệ Nhân tạo - Khoa CNTT - Trường Đại học Thủy Lợi

Nội dung thực thi theo đúng chuẩn yêu cầu mục 4 & 5 của đề bài:
  4.1 / Bước 1: Trích xuất đặc trưng 39-D (MFCC + Δ + Δ²) & Chuẩn hóa CMVN
  4.2 / Bước 2: Khởi tạo mô hình Left-to-Right Gaussian HMM
  4.3 / Bước 3: Huấn luyện Whole-word HMM cho 5 lệnh tiếng Việt & Thử nghiệm n_states
  4.4 / Bước 4-5: Nhận dạng Maximum Likelihood & Viterbi State Alignment
  4.5 / Bước 7: So sánh hiệu năng HMM với DTW Baseline
  4.6 / Mục F: Phân tích Mini Lexicon & ánh xạ Phone HMM
  4.7 / Bước 8: Xây dựng Bigram Language Model & MAP Re-ranking (Bayesian)
  4.8 / Bước 6: Đánh giá Confusion Matrix, Độ bền vững (Robustness) & Ngưỡng Rejection
===================================================================================
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

# Cấu hình UTF-8 cho console Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import các hàm từ các module đã tách biệt
from speech_features import extract_features, plot_feature_heatmaps
from hmm_model import (
    train_word_hmms,
    recognize_single_utterance,
    get_viterbi_alignment,
    plot_viterbi_trellis,
    save_models
)
from dtw_baseline import DTWRecognizer, evaluate_dtw
from lexicon_lm import parse_lexicon, BigramLanguageModel, map_rerank
from eval_utils import (
    evaluate_dataset,
    plot_confusion_matrix_figure,
    evaluate_robustness_with_noise,
    analyze_rejection_threshold
)

# Cấu hình danh sách 5 từ lệnh và đường dẫn
LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
FIGURES_DIR = "figures"
MODELS_DIR = "models"


def main():
    print("=" * 80)
    print("      BÀI TẬP LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI GAUSSIAN HMM")
    print("                     KHOA CNTT - ĐẠI HỌC THỦY LỢI")
    print("=" * 80)

    # Đảm bảo các thư mục lưu trữ tồn tại
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 4.1 A — Chuẩn hóa dữ liệu và trích đặc trưng 39 chiều
    # -------------------------------------------------------------------------
    print("\n[PHẦN 1] TRÍCH XUẤT ĐẶC TRƯNG 39 CHIỀU (MFCC + Δ + Δ²) VÀ CHUẨN HÓA CMVN")
    print("-" * 75)

    sample_files = [
        ("mo", f"{TRAIN_DIR}/mo/mo_01.wav"),
        ("dong", f"{TRAIN_DIR}/dong/dong_01.wav"),
        ("len", f"{TRAIN_DIR}/len/len_01.wav")
    ]

    for label, path in sample_files:
        feats = extract_features(path)
        print(f" • File: {Path(path).name:12s} | Nhãn: '{label:5s}' | Shape: {str(feats.shape):14s} (Frames: {feats.shape[0]}, Dims: {feats.shape[1]})")

    # Vẽ Heatmap so sánh đặc trưng giữa 2 từ 'mo' và 'dong'
    heatmap_save = f"{FIGURES_DIR}/heatmap_mo_vs_dong.png"
    plot_feature_heatmaps(sample_files[0][1], "mở", sample_files[1][1], "đóng", save_path=heatmap_save)

    # -------------------------------------------------------------------------
    # 4.2 & 4.3 B, C — Huấn luyện mô hình Whole-word Gaussian HMM cho mỗi từ
    # -------------------------------------------------------------------------
    print("\n[PHẦN 2] KHỞI TẠO VÀ HUẤN LUYỆN 5 WHOLE-WORD GAUSSIAN HMM (LEFT-TO-RIGHT)")
    print("-" * 75)
    print("Topology: 5-state left-to-right (π=[1,0,0,0,0], a_ii≈0.7, a_i,i+1≈0.3, a_55=1.0)")
    print("Covariance: Diagonal (tiết kiệm tham số, tránh overfit khi lượng mẫu ít)")

    # Huấn luyện mô hình chính (5 trạng thái)
    start_train_time = time.time()
    models = train_word_hmms(TRAIN_DIR, LABELS, n_states=5, n_iter=100)
    train_duration = time.time() - start_train_time

    # Lưu mô hình đã huấn luyện
    model_path = f"{MODELS_DIR}/word_hmms_5states.pkl"
    save_models(models, model_path)

    # Kiểm tra ma trận chuyển trạng thái transmat_ của 1 model để đảm bảo topology
    print("\n • Ma trận chuyển trạng thái A (transmat_) của model 'mo':")
    print(np.round(models['mo'].transmat_, 3))

    # Thí nghiệm n_states ∈ {3, 5, 7} theo yêu cầu Mục 4.3 & Bảng E1
    print("\n >> Thí nghiệm E1: So sánh số lượng trạng thái n_states ∈ {3, 5, 7}:")
    for s in [3, 5, 7]:
        m_temp = train_word_hmms(TRAIN_DIR, LABELS, n_states=s, n_iter=80)
        acc_temp, _, _, _ = evaluate_dataset(TEST_DIR, m_temp, LABELS)
        print(f"    * n_states = {s:d} -> Test Accuracy = {acc_temp:.1f}%")

    # -------------------------------------------------------------------------
    # 4.4 D — Nhận dạng Maximum Likelihood và Viterbi Alignment
    # -------------------------------------------------------------------------
    print("\n[PHẦN 3] NHẬN DẠNG MAXIMUM LIKELIHOOD & PHÂN TÍCH VITERBI STATE ALIGNMENT")
    print("-" * 75)

    test_wav = f"{TEST_DIR}/mo/mo_07.wav"
    feat_test = extract_features(test_wav)
    pred_label, scores, margin = recognize_single_utterance(feat_test, models)

    print(f"Nhận dạng file thử nghiệm: '{test_wav}'")
    print(f"Dự đoán tốt nhất (Top-1): '{pred_label.upper()}' | Margin tự tin (Top 1 - Top 2): {margin:.2f}")
    print("\nBảng điểm Acoustic Log-Likelihood qua 5 mô hình:")
    print(f" {'TỪ LỆNH':<10} | {'LOG-LIKELIHOOD':<18} | {'SCORE/FRAME':<15} | {'KẾT QUẢ':<10}")
    print(" " + "-" * 60)
    for w in sorted(scores.keys(), key=lambda k: scores[k], reverse=True):
        per_frame = scores[w] / len(feat_test)
        mark = "✓ (Top 1)" if w == pred_label else ""
        print(f" {w:<10} | {scores[w]:18.2f} | {per_frame:15.2f} | {mark:<10}")

    # Viterbi Alignment cho 2 utterance khác nhau theo yêu cầu 4.4
    logp_viterbi, states_mo = get_viterbi_alignment(feat_test, models['mo'])
    print(f"\nViterbi Log-prob cho 'mo_07.wav' trên model 'mo': {logp_viterbi:.2f}")
    print(f"Chuỗi trạng thái Viterbi: {states_mo}")
    viterbi_save1 = f"{FIGURES_DIR}/viterbi_alignment_mo_07.png"
    plot_viterbi_trellis(states_mo, title="Viterbi State Alignment: mo_07.wav vs Model 'mo'", save_path=viterbi_save1)

    test_wav2 = f"{TEST_DIR}/dong/dong_07.wav"
    feat_test2 = extract_features(test_wav2)
    _, states_dong = get_viterbi_alignment(feat_test2, models['dong'])
    viterbi_save2 = f"{FIGURES_DIR}/viterbi_alignment_dong_07.png"
    plot_viterbi_trellis(states_dong, title="Viterbi State Alignment: dong_07.wav vs Model 'dong'", save_path=viterbi_save2)

    # -------------------------------------------------------------------------
    # 4.5 E — So sánh với DTW Baseline (Lab 2)
    # -------------------------------------------------------------------------
    print("\n[PHẦN 4] SO SÁNH VỚI BASELINE DTW (DYNAMIC TIME WARPING)")
    print("-" * 75)
    dtw = DTWRecognizer()
    dtw.fit(TRAIN_DIR, LABELS)
    dtw_acc, dtw_time = evaluate_dtw(TEST_DIR, dtw, LABELS)

    # Đánh giá HMM
    t_start = time.time()
    hmm_acc, y_true, y_pred, _ = evaluate_dataset(TEST_DIR, models, LABELS)
    hmm_time = (time.time() - t_start) / len(y_true)

    print("\nBẢNG SO SÁNH CHỈ SỐ HMM (LAB 3) VS DTW (LAB 2):")
    print(f" {'Chỉ số':<26} | {'DTW (Lab 2)':<16} | {'HMM (Lab 3)':<16} | {'Nhận xét':<20}")
    print(" " + "-" * 85)
    print(f" {'Accuracy':<26} | {dtw_acc:14.1f}% | {hmm_acc:14.1f}% | {'HMM tổng quát hóa tốt hơn':<20}")
    print(f" {'Số mẫu train/lớp':<26} | {'6 mẫu':<16} | {'6 mẫu':<16} | {'HMM tận dụng đa mẫu tốt':<20}")
    print(f" {'Thời gian train':<26} | {'~0.0s (lưu mẫu)':<16} | {train_duration:14.2f}s | {'HMM cần huấn luyện EM':<20}")
    print(f" {'Thời gian test/trial':<26} | {dtw_time*1000:13.2f}ms | {hmm_time*1000:13.2f}ms | {'HMM nhanh gấp nhiều lần':<20}")
    print(f" {'Khả năng giải thích':<26} | {'Warping path':<16} | {'Viterbi state path':<16} | {'HMM phân đoạn ngữ âm':<20}")

    # -------------------------------------------------------------------------
    # 4.6 F — Mini Lexicon Parser & Giải thích Phone HMM
    # -------------------------------------------------------------------------
    print("\n[PHẦN 5] PHÂN TÍCH TỪ ĐIỂN PHÁT ÂM (MINI LEXICON)")
    print("-" * 75)
    lexicon_path = "lexicon.txt"
    lexicon = parse_lexicon(lexicon_path)
    for word, phones in lexicon.items():
        print(f" • Từ: '{word:<6s}' -> Chuỗi âm vị (Phone sequence): {' - '.join(phones)}")
    print("\n >> Ý nghĩa trong kiến trúc ASR:")
    print("    Trong bài toán Whole-word HMM, mỗi từ được mô hình hóa bằng 1 HMM độc lập (không cần lexicon).")
    print("    Khi mở rộng sang nhận dạng từ vựng lớn (LVCSR), ta phải dùng Subword/Phone HMM.")
    print("    Khi đó, Lexicon đóng vai trò 'bản đồ kết nối' để ghép nối các mô hình âm vị nhỏ thành mô hình từ.")

    # -------------------------------------------------------------------------
    # 4.7 G — Bigram Language Model và MAP Re-ranking
    # -------------------------------------------------------------------------
    print("\n[PHẦN 6] MÔ HÌNH NGÔN NGỮ BIGRAM VÀ TÁI XẾP HẠNG BAYESIAN (MAP RE-RANKING)")
    print("-" * 75)

    # Tập ngữ liệu câu lệnh mẫu mô phỏng kịch bản điều khiển nhà thông minh / robot
    corpus = [
        ["bật", "đèn", "lên"],
        ["tắt", "đèn", "đi"],
        ["mở", "cửa", "lên"],
        ["đóng", "cửa", "lại"],
        ["lên", "tầng", "trên"],
        ["xuống", "dưới", "nhà"],
        ["dừng", "lại", "ngay"],
        ["chạy", "lên", "trên"],
        ["đi", "xuống", "dưới"],
        ["bấm", "dừng", "khẩn", "cấp"],
        ["ra", "lệnh", "mở"],
        ["yêu", "cầu", "đóng"]
    ]

    lm = BigramLanguageModel(vocabulary=LABELS, k=0.5)
    lm.fit(corpus)

    # Thử nghiệm tái xếp hạng với từ phía trước là 'ra lệnh' -> sau đó khả năng cao là 'mở'
    prev_context = "lệnh"
    print(f"Ngữ cảnh từ đứng trước: '{prev_context}'")
    print(f"Thử nghiệm với các mức trọng số λ_LM ∈ [0.0, 0.1, 0.3, 1.0]:")

    for lambda_val in [0.0, 0.1, 0.3, 1.0]:
        best_w, details = map_rerank(feat_test, prev_word=prev_context, models=models, lm=lm, lm_weight=lambda_val)
        print(f" • λ_LM = {lambda_val:4.1f} -> Dự đoán: '{best_w.upper()}' | Tổng điểm: {details[best_w]['total_score']:.2f} (Acoustic: {details[best_w]['acoustic_norm']:.2f}, LM: {details[best_w]['lm_logprob']:.2f})")

    # -------------------------------------------------------------------------
    # 4.8 H — Đánh giá toàn diện, Ma trận nhầm lẫn & Thí nghiệm Robustness
    # -------------------------------------------------------------------------
    print("\n[PHẦN 7] ĐÁNH GIÁ MA TRẬN NHẦM LẪN VÀ THÍ NGHIỆM ROBUSTNESS")
    print("-" * 75)

    # Vẽ Confusion Matrix
    cm_save = f"{FIGURES_DIR}/confusion_matrix.png"
    plot_confusion_matrix_figure(y_true, y_pred, LABELS, title=f"Confusion Matrix (Accuracy: {hmm_acc:.1f}%)", save_path=cm_save)
    print(f"Độ chính xác trên tập kiểm thử sạch: {hmm_acc:.1f}%")

    # Thí nghiệm Robustness với mức độ nhiễu môi trường tăng dần
    print("\n >> Thí nghiệm E5: Khảo sát độ bền vững khi có nhiễu Gaussian tăng dần:")
    robustness_results = evaluate_robustness_with_noise(TEST_DIR, models, LABELS, noise_levels=[0.0, 0.1, 0.2, 0.4, 0.6])
    for nl, acc in robustness_results.items():
        print(f"    * Mức nhiễu σ = {nl:3.1f} -> Accuracy = {acc:5.1f}%")

    # Phân tích ngưỡng từ chối Rejection Threshold cho từ ngoài từ vựng
    rejection_save = f"{FIGURES_DIR}/rejection_threshold.png"
    threshold = analyze_rejection_threshold(TEST_DIR, models, LABELS, save_path=rejection_save)
    print(f"\n >> Ngưỡng từ chối (Rejection Threshold) đề xuất: Score/Frame < {threshold:.2f}")
    print("    Nếu một âm thanh có Score/Frame thấp hơn ngưỡng này -> Hệ thống từ chối và gán nhãn 'UNKNOWN'.")

    print("\n" + "=" * 80)
    print(">> TOÀN BỘ CÁC BƯỚC THỰC HÀNH CỦA LAB 3 ĐÃ HOÀN THÀNH XUẤT SẮC!")
    print(f">> Các biểu đồ và mô hình đã được xuất tại: '{os.path.abspath(FIGURES_DIR)}' & '{os.path.abspath(MODELS_DIR)}'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
