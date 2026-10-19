# BÀI THỰC HÀNH LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
## Học phần: Xử lý Âm thanh và Tiếng nói (CSE457) - Đại học Thủy Lợi

Hệ thống nhận dạng tiếng nói cho 5 lệnh tiếng Việt tách rời (*"mở", "đóng", "lên", "xuống", "dừng"*) sử dụng vector đặc trưng 39 chiều (MFCC + $\Delta$ + $\Delta^2$), mô hình Left-to-Right Gaussian HMM, giải thuật Forward-Backward (Baum-Welch), giải mã Viterbi, so sánh đối sánh mẫu DTW, từ điển phát âm Lexicon và mô hình ngôn ngữ Bigram với MAP Re-ranking.

---

## 1. Cấu trúc thư mục mã nguồn

Dự án được thiết kế theo cấu trúc module hóa chuẩn công nghiệp, tách biệt rõ ràng các nhiệm vụ để file chính ngắn gọn, dễ bảo trì:

```text
lab3_speech_recognition/
│
├── Lab03.py                     # File chạy nộp bài chính thức theo đề bài
├── main.py                      # Pipeline chạy toàn diện các bước thực hành từ A -> H
├── speech_features.py           # Module trích xuất đặc trưng 39-D (MFCC + Delta + Delta2 + CMVN)
├── hmm_model.py                 # Module khởi tạo, huấn luyện HMM, ML scoring và Viterbi Alignment
├── dtw_baseline.py              # Module baseline DTW (Dynamic Time Warping) từ Lab 2
├── lexicon_lm.py                # Module từ điển phát âm Lexicon & Mô hình ngôn ngữ Bigram (Add-k)
├── eval_utils.py                # Module đánh giá ma trận nhầm lẫn, robustness và ngưỡng Rejection
│
├── text_to_speech.py            # [YÊU CẦU ĐẶC BIỆT] Bộ chuyển văn bản sang WAV với giọng tùy biến
├── generate_dataset.py          # Script tự động sinh tập dữ liệu 5 từ (30 train + 10 test) đa dạng giọng
│
├── lexicon.txt                  # File từ điển phát âm mini cho 5 lệnh tiếng Việt
├── REPORT_LAB03_ANSWERS.md      # Báo cáo học thuật trả lời đầy đủ 10 câu hỏi lý thuyết Mục 6.1
│
├── dataset/                     # Thư mục chứa dữ liệu âm thanh (WAV mono, 16kHz, 16-bit PCM)
│   ├── train/                   # 6 file huấn luyện cho mỗi từ (mo, dong, len, xuong, dung)
│   └── test/                    # 2 file kiểm thử cho mỗi từ
│
├── figures/                     # Lưu các biểu đồ kết quả (Confusion matrix, Viterbi path, Heatmap,...)
└── models/                      # Lưu các mô hình HMM đã huấn luyện (.pkl)
```

---

## 2. Hướng dẫn cài đặt môi trường

Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn gtts
```

---

## 3. Hướng dẫn sử dụng

### 3.1. Chạy bài thực hành Lab 3 (End-to-End Pipeline)
Chỉ cần thực thi lệnh sau:
```bash
python Lab03.py
# hoặc: python main.py
```
Chương trình sẽ tự động thực hiện tuần tự:
1. Trích xuất đặc trưng 39 chiều và vẽ Heatmap so sánh.
2. Huấn luyện 5 mô hình Left-to-Right Gaussian HMM và thử nghiệm số trạng thái $n\_states \in \{3, 5, 7\}$.
3. Nhận dạng Maximum Likelihood, hiển thị bảng Likelihood score và vẽ đường căn chỉnh trạng thái Viterbi.
4. Chạy mô hình DTW baseline và in bảng so sánh hiệu năng chi tiết (Accuracy, Tốc độ, Bộ nhớ).
5. Phân tích file từ điển `lexicon.txt` và giải thích cơ chế ghép nối Phone HMM.
6. Xây dựng mô hình ngôn ngữ Bigram (Add-k smoothing) và thực hiện tái xếp hạng MAP Bayesian với các giá trị $\lambda_{LM} \in \{0.0, 0.1, 0.3, 1.0\}$.
7. Đánh giá Confusion Matrix, thử nghiệm độ bền vững khi có nhiễu Gaussian và phân tích ngưỡng Rejection cho từ ngoài từ vựng.

---

### 3.2. Sử dụng file Chuyển văn bản sang âm thanh WAV có đổi giọng (`text_to_speech.py`)

Module này hỗ trợ:
- Chuyển văn bản tiếng Việt sang file WAV **chuẩn 16 kHz, 16-bit PCM mono**, tự động chèn 0.3s khoảng lặng (silence pad) ở hai đầu đúng yêu cầu đề bài.
- **Thay đổi giọng nói linh hoạt**: Giọng nữ chuẩn, nữ cao, nam trầm, nam tự nhiên, trẻ em, nói chậm, nói nhanh,...
- Có thể dùng trực tiếp qua giao diện dòng lệnh (CLI) hoặc gọi hàm trong Python.

#### Dùng qua dòng lệnh (CLI):
```bash
# 1. Xem danh sách các giọng mẫu có sẵn:
python text_to_speech.py --list-voices

# 2. Sinh giọng nam trầm:
python text_to_speech.py --text "mở" --voice male_deep --output mo_nam_tram.wav

# 3. Sinh giọng nữ nói chậm:
python text_to_speech.py --text "dừng" --voice slow_female --output dung_nu_cham.wav

# 4. Tùy chỉnh cao độ (pitch) và tốc độ (rate) tùy ý:
python text_to_speech.py --text "lên" --pitch -4.0 --rate 1.1 --output len_tuy_chinh.wav
```

#### Dùng trong code Python:
```python
from text_to_speech import synthesize_text_to_wav

synthesize_text_to_wav(
    text="đóng",
    output_path="dong.wav",
    voice="male",          # 'female', 'male', 'male_deep', 'child', 'slow_female', 'fast_male'
    pitch_shift=-3.5,      # Giảm bán cung để trầm giọng (nam), tăng để cao giọng (nữ)
    rate=1.0,              # Tốc độ nói: 1.0 bình thường, 1.2 nhanh, 0.8 chậm
    silence_pad_sec=0.3    # Khoảng lặng đầu/cuối (giây)
)
```

---

## 4. Tóm tắt kết quả so sánh HMM vs DTW

| Tiêu chí | DTW (Lab 2) | HMM (Lab 3) | Ưu thế & Nhận xét |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 100.0% | 100.0% | HMM tổng quát hóa phân bố xác suất tốt hơn |
| **Thời gian suy diễn/mẫu** | **~40.8 ms** | **~2.5 ms** | **HMM nhanh hơn gấp 16 lần** |
| **Cách thức lưu trữ** | Phải lưu tất cả template mẫu | Chỉ lưu ma trận $A$, $\mu$, $\Sigma$ nén | HMM tiết kiệm bộ nhớ khi tập dữ liệu lớn |
| **Khả năng mở rộng** | Rất khó ghép với Language Model | Dễ dàng tích hợp với Lexicon, N-gram LM, WFST | HMM là nền tảng cho nhận dạng tiếng nói liên tục |
| **Giải thích căn chỉnh** | Warping path giữa 2 tín hiệu | Viterbi path (đoạn ngữ âm & thời lượng) | Viterbi phân đoạn âm học theo trạng thái |
