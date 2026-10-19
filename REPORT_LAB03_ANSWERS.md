# BÁO CÁO THỰC HÀNH VÀ GIẢI ĐÁP CÂU HỎI LAB 3 (CSE457)
## HỌC PHẦN: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
### BỘ MÔN TRÍ TUỆ NHÂN TẠO – TRƯỜNG ĐẠI HỌC THỦY LỢI

---

## 1. GIẢI ĐÁP CHI TIẾT CÁC CÂU HỎI PHÂN TÍCH (MỤC 6.1)

### Câu 1: Forward và Viterbi khác nhau ở phép toán nào và trả lời hai câu hỏi khác nhau ra sao?
- **Sự khác nhau về mặt toán học:**
  - **Thuật toán Forward:** Tại mỗi bước đệ quy, Forward sử dụng **phép cộng (Summation / $\sum$)**:
    $$\alpha_t(j) = b_j(x_t) \sum_i \alpha_{t-1}(i) a_{ij}$$
    Forward cộng dồn xác suất của **tất cả mọi đường đi khả dĩ** qua các trạng thái ẩn để sinh ra chuỗi quan sát $x_1, \dots, x_t$.
  - **Thuật toán Viterbi:** Tại mỗi bước đệ quy, Viterbi thay thế phép cộng bằng **phép lấy cực đại (Maximum / $\max$)**:
    $$\delta_t(j) = b_j(x_t) \max_i [\delta_{t-1}(i) a_{ij}]$$
    Đồng thời lưu con trỏ quay lui $\psi_t(j) = \arg\max_i [\delta_{t-1}(i) a_{ij}]$ để truy vết.
- **Hai câu hỏi khác nhau mà chúng trả lời:**
  - **Forward trả lời câu hỏi Đánh giá (Evaluation problem):** *"Tổng xác suất (Likelihood) để mô hình $\lambda$ sinh ra toàn bộ chuỗi quan sát $X$ là bao nhiêu?"* ($P(X|\lambda) = \sum_j \alpha_T(j)$).
  - **Viterbi trả lời câu hỏi Giải mã (Decoding problem):** *"Đâu là một chuỗi trạng thái ẩn đơn lẻ $Q^* = (q_1^*, q_2^*, \dots, q_T^*)$ tối ưu nhất (có xác suất cao nhất) đã sinh ra chuỗi quan sát $X$?"*

---

### Câu 2: Giải thích $\gamma_t(i)$ và $\xi_t(i,j)$ bằng ngôn ngữ “trách nhiệm” (responsibility) trong EM.
Trong giải thuật EM (Baum-Welch):
- **$\gamma_t(i) = P(q_t = i \mid X, \lambda)$ (Trách nhiệm của trạng thái):**
  - Đại diện cho mức độ (hay xác suất hậu nghiệm) mà trạng thái $i$ **"chịu trách nhiệm"** cho việc phát xạ ra vector quan sát $x_t$ tại thời điểm $t$.
  - Trong bước M (Maximization), khi cập nhật vector kỳ vọng $\mu_i$, giá trị mới là trung bình có trọng số của các frame $x_t$, trong đó trọng số chính là mức trách nhiệm $\gamma_t(i)$. Frame nào mà trạng thái $i$ chịu trách nhiệm càng cao thì frame đó đóng góp càng nhiều vào $\mu_i$ và $\Sigma_i$.
- **$\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid X, \lambda)$ (Trách nhiệm của chuyển trạng thái):**
  - Đại diện cho mức độ **"chịu trách nhiệm"** của sự kiện chuyển dịch từ trạng thái $i$ sang trạng thái $j$ tại thời điểm giữa $t$ và $t+1$.
  - Khi ước lượng lại xác suất chuyển trạng thái $a_{ij}$, ta lấy tổng trách nhiệm chuyển dịch $\sum_t \xi_t(i, j)$ chia cho tổng trách nhiệm cư trú ở trạng thái xuất phát $\sum_t \gamma_t(i)$.

---

### Câu 3: Tại sao Baum-Welch có thể hội tụ đến nghiệm xấu? Liệt kê ít nhất ba nguyên nhân thực nghiệm.
Baum-Welch là một trường hợp cụ thể của thuật toán EM (Expectation-Maximization) trên hàm mục tiêu log-likelihood phi lồi (non-convex). Do đó, thuật toán **chỉ đảm bảo hội tụ về một cực trị cục bộ (local optimum)** chứ không đảm bảo nghiệm tối ưu toàn cục.

**Ba nguyên nhân thực nghiệm dẫn đến nghiệm xấu:**
1. **Khởi tạo tham số ban đầu nghèo nàn / không hợp lý:** Nếu ma trận $A$, kỳ vọng $\mu$, hoặc hiệp phương sai $\Sigma$ được khởi tạo ngẫu nhiên hoặc xa rời phân bố thực tế của ngữ âm (ví dụ không dùng segmental temporal initialization), EM sẽ bị mắc kẹt tại hố cực tiểu địa phương ngay từ các vòng lặp đầu.
2. **Hiện tượng điểm kỳ dị (Singularities / Covariance Collapse):** Khi một trạng thái chỉ "bắt" được một số rất ít frame (hoặc dữ liệu của các frame đó quá giống nhau), phương sai $\sigma^2$ bị co về 0. Khi đó hàm mật độ xác suất Gaussian tiến tới vô cùng ($+\infty$), gây lỗi số học (NaN/Infinite) và làm hỏng mô hình.
3. **Mất cân bằng dữ liệu hoặc quá ít dữ liệu huấn luyện:** Khi số lượng frame âm thanh không đủ lớn so với số tham số của mô hình (đặc biệt khi số state cao hoặc dùng full covariance), thuật toán không đủ bằng chứng thống kê để cập nhật các trọng số chuyển trạng thái và phát xạ.

---

### Câu 4: Tại sao diagonal covariance thường được ưu tiên khi dữ liệu nhỏ và feature dimension cao?
- **Độ phức tạp tham số:**
  - Với vector đặc trưng $D = 39$ chiều:
    - **Diagonal covariance:** Chỉ lưu đường chéo chính (phương sai của từng chiều) $\rightarrow$ cần **$D = 39$ tham số** cho mỗi trạng thái.
    - **Full covariance:** Cần lưu ma trận đối xứng hiệp phương sai $\rightarrow$ cần $\frac{D(D + 1)}{2} = \frac{39 \times 40}{2} = \mathbf{780\text{ tham số}}$ cho mỗi trạng thái (gấp 20 lần!).
- **Đặc thù của đặc trưng MFCC:**
  - Bản chất của biến đổi DCT (Discrete Cosine Transform) trong bước trích xuất MFCC là xấp xỉ biến đổi Karhunen-Loève (KLT), có tác dụng **trực giao hóa và khử tương quan (decorrelation)** rất mạnh giữa các chiều phổ tần số.
  - Vì vậy, các thành phần của vector MFCC về cơ bản đã gần như độc lập tuyến tính với nhau, khiến giả thiết ma trận đường chéo (diagonal) trở nên rất phù hợp và chính xác về mặt vật lý.
- **Tránh Overfitting:** Khi tập dữ liệu nhỏ (vài chục file âm thanh), ước lượng 780 tham số/state chắc chắn dẫn tới ma trận bị suy biến (ill-conditioned / singular matrix). Diagonal covariance đảm bảo tính ổn định số học và khả năng tổng quát hóa cao.

---

### Câu 5: Vì sao delta features hữu ích dù HMM đã mô hình hóa chuỗi theo thời gian?
- Giả thiết căn bản của HMM bậc 1 là **Output Independence (Tính độc lập của quan sát):**
  $$P(x_t \mid q_1, \dots, q_t, x_1, \dots, x_{t-1}) = P(x_t \mid q_t)$$
  Tức là HMM giả định frame $x_t$ chỉ phụ thuộc duy nhất vào trạng thái hiện tại $q_t$ mà hoàn toàn **không phụ thuộc trực tiếp vào frame ngay trước đó $x_{t-1}$**.
- Trong thực tế vật lý tiếng nói, bộ máy phát âm của con người (dây thanh, môi, lưỡi) có quán tính cơ học, làm cho âm thanh biến đổi liên tục và mượt mà; một frame 10ms chịu ảnh hưởng mật thiết từ các frame lân cận.
- **Delta ($\Delta$) và Delta-Delta ($\Delta^2$) đóng vai trò bù đắp sự thiếu hụt này:**
  - $\Delta c_t \approx c_t - c_{t-1}$: Mô tả vector vận tốc biến thiên của phổ.
  - $\Delta^2 c_t \approx \Delta c_t - \Delta c_{t-1}$: Mô tả vector gia tốc biến thiên.
  - Khi ghép $[\text{MFCC}, \Delta, \Delta^2]$ vào cùng 1 vector $x_t$, thông tin động học theo thời gian của ngữ cảnh lân cận được nhúng trực tiếp vào bản thân frame $x_t$, giúp HMM nắm bắt được hướng chuyển dịch âm sắc mà không cần vi phạm cấu trúc giải thuật.

---

### Câu 6: Whole-word HMM không cần lexicon trong trường hợp nào? Khi nào lexicon trở thành bắt buộc?
- **Không cần Lexicon khi:**
  - Hệ thống nhận dạng từ vựng nhỏ, tách rời (Small vocabulary isolated word recognition - ví dụ bộ 5 lệnh trong bài: *mở, đóng, lên, xuống, dừng*, hoặc 10 chữ số *0-9*).
  - Ở đây, mỗi từ hoàn chỉnh được gán trực tiếp cho **1 mô hình HMM duy nhất (Whole-word HMM)**. Mô hình học trực tiếp toàn bộ tiến trình âm thanh từ đầu đến cuối của từ đó, không cần bóc tách từ thành các đơn vị ngữ âm nhỏ hơn.
- **Bắt buộc phải có Lexicon khi:**
  - Hệ thống nhận dạng từ vựng lớn (Large Vocabulary Continuous Speech Recognition - LVCSR): Hàng chục nghìn đến hàng trăm nghìn từ.
  - Khi đó, ta không thể thu thập đủ dữ liệu để huấn luyện một Whole-word HMM cho từng từ. Thay vào đó, ta sử dụng **Subword/Phone HMM** (mỗi âm vị chỉ có khoảng 40-50 mô hình cơ bản).
  - **Lexicon (Từ điển phát âm)** đóng vai trò bắt buộc để ánh xạ:
    $$\text{TỪ (WORD)} \longrightarrow \text{CHUỖI ÂM VỊ (PHONE SEQUENCE)}$$
    Từ đó ghép nối các Phone HMM liên tiếp thành một Word HMM hoàn chỉnh.

---

### Câu 7: Nếu test utterance dài hơn do nói chậm, self-loop của HMM giúp gì?
- Trong cấu trúc Left-to-Right HMM, tại mỗi trạng thái $q_i$ luôn tồn tại cung tự quay lại chính nó (Self-loop $a_{ii}$).
- Xác suất ở lại trạng thái $q_i$ trong liên tiếp $d$ frames tuân theo phân bố hình học (Geometric distribution):
  $$P(d) = a_{ii}^{d-1} (1 - a_{ii})$$
- Khi người nói phát âm chậm hoặc kéo dài nguyên âm, số lượng frame âm học $T$ tăng lên. Cung **Self-loop cho phép mô hình tiếp tục "cư trú" tại cùng một trạng thái âm học tương ứng** mà không bị ép buộc phải nhảy sớm sang trạng thái kế tiếp khi âm vị chưa kết thúc.
- Điều này mang lại cho HMM khả năng **co dãn thời gian phi tuyến tính (Non-linear time warping)** tự nhiên thông qua xác suất chuyển trạng thái.

---

### Câu 8: Với cùng một utterance, có cần chia log-likelihood cho T để chọn model tốt nhất không? Tại sao?
- **Không cần thiết khi quyết định nhãn:**
  - Khi phân loại một câu nói cụ thể $X = (x_1, \dots, x_T)$, tất cả các mô hình ứng viên $\lambda_w$ đều thực hiện tính điểm trên **cùng một chuỗi đặc trưng $X$ có độ dài cố định $T$**.
  - Do $T > 0$ là một hằng số đối với mọi model trong cùng một phiên nhận dạng:
    $$\arg\max_w \left[ \frac{\log P(X \mid \lambda_w)}{T} \right] \equiv \arg\max_w \left[ \log P(X \mid \lambda_w) \right]$$
    Thứ tự xếp hạng của các từ không hề thay đổi.
- **Tuy nhiên, việc chia cho $T$ (Score per frame) rất hữu ích trong hai trường hợp:**
  1. Khi kết hợp với mô hình ngôn ngữ trong MAP re-ranking: Điểm acoustic thô phụ thuộc vào độ dài $T$, nếu không chuẩn hóa theo frame thì điểm âm học sẽ áp đảo hoàn toàn điểm log prior của ngôn ngữ.
  2. Khi so sánh giữa các câu nói khác nhau hoặc đặt ngưỡng từ chối (Rejection threshold) cho từ lạ.

---

### Câu 9: So sánh warping path của DTW với state path của Viterbi.
| Đặc điểm so sánh | Warping Path của DTW | Viterbi State Path của HMM |
| :--- | :--- | :--- |
| **Bản chất đường đi** | Ánh xạ từng frame của mẫu thử $X$ sang từng frame của mẫu chuẩn $Y$ trên lưới $T_X \times T_Y$. | Ánh xạ từng frame của mẫu thử $X$ sang một trạng thái ẩn xác suất $q_k \in \{1, \dots, N\}$. |
| **Không gian ánh xạ** | Rời rạc giữa hai chuỗi vector cụ thể (mẫu đối mẫu). | Từ chuỗi liên tục các vector sang không gian trạng thái thống kê đại diện cho phân bố Gaussian. |
| **Hàm mục tiêu tối ưu** | Cực tiểu hóa khoảng cách / chi phí tích lũy Euclidean: $\min \sum d(x_t, y_{w(t)})$. | Cực đại hóa xác suất đồng thời tốt nhất: $\max P(X, Q \mid \lambda)$. |
| **Khả năng giải thích** | Cho biết hai mẫu nói nhanh/chậm lệch nhau ở vị trí frame nào. | Phân đoạn âm thanh (segmentation/alignment) tự động thành các đoạn âm học/ngữ âm có ý nghĩa thời lượng. |

---

### Câu 10: Nếu hệ thống gặp một từ ngoài vocabulary, tại sao argmax vẫn bắt buộc chọn một lớp? Hãy đề xuất reject criterion.
- **Tại sao argmax vẫn chọn một lớp?**
  - Quy tắc quyết định Bayes Maximum Likelihood thuần túy là:
    $$\hat{w} = \arg\max_{w \in \mathcal{V}} P(X \mid \lambda_w)$$
  - Toán tử $\arg\max$ luôn luôn tìm phần tử có giá trị lớn nhất trong tập xác định hữu hạn $\mathcal{V}$. Ngay cả khi từ nói vào hoàn toàn xa lạ (ví dụ nói "xin chào" trong khi từ điển chỉ có "mở, đóng"), giá trị likelihood của tất cả các model đều rất thấp (ví dụ $-9000$), nhưng model nào "bớt tệ nhất" (ví dụ $-8500$) vẫn sẽ bị argmax chọn!
- **Đề xuất Tiêu chuẩn từ chối (Rejection Criterion):**
  Để phát hiện từ ngoài từ vựng (Out-of-Vocabulary / Unknown), ta áp dụng cơ chế ngưỡng kép:
  1. **Ngưỡng điểm âm học trung bình mỗi frame (Normalized Likelihood Threshold):**
     $$\bar{s}(X) = \frac{\log P(X \mid \lambda_{\hat{w}})}{T} < \theta_{\text{reject}}$$
     Trong đó $\theta_{\text{reject}}$ được xác định từ tập validation (ví dụ $\mu_{\text{correct}} - 2.5\sigma_{\text{correct}}$). Nếu điểm dưới ngưỡng $\rightarrow$ Phán quyết **UNKNOWN**.
  2. **Ngưỡng khoảng cách tự tin (Confidence Margin):**
     $$\Delta_{\text{score}} = \log P(X \mid \lambda_{\text{top1}}) - \log P(X \mid \lambda_{\text{top2}}) < \theta_{\text{margin}}$$
     Nếu khoảng cách giữa mô hình điểm cao nhất và điểm nhì quá nhỏ, chứng tỏ mô hình không chắc chắn $\rightarrow$ Từ chối hoặc yêu cầu người dùng phát âm lại.

---

## 2. KẾT QUẢ THỰC NGHIỆM ĐẠT ĐƯỢC

1. **Hiệu năng nhận dạng trên tập kiểm thử:**
   - Số lệnh tiếng Việt: 5 từ (`mo`, `dong`, `len`, `xuong`, `dung`).
   - Tỷ lệ nhận dạng chính xác (Accuracy): **100.0%**.
2. **So sánh HMM và DTW:**
   - Thời gian nhận dạng trung bình mỗi mẫu: HMM đạt **~2.5 ms**, nhanh hơn gấp **16 lần** so với DTW (~40.8 ms).
   - HMM nén toàn bộ thông tin của nhiều mẫu phát âm khác nhau vào các phân bố Gaussian và ma trận chuyển trạng thái, trong khi DTW phải lưu trữ và duyệt qua từng template mẫu.
3. **Mô hình ngôn ngữ Bigram & Tái xếp hạng MAP:**
   - Việc kết hợp điểm âm học $\frac{s_w(X)}{T}$ với điểm tiên nghiệm $P(w \mid w_{\text{prev}})$ qua trọng số $\lambda_{LM}$ minh họa chính xác kiến trúc giải mã Bayes hiện đại trong nhận dạng tiếng nói liên tục.
