"""
Module: generate_dataset.py
Chức năng: Tự động khởi tạo bộ dữ liệu 5 lệnh tiếng Việt mẫu cho Lab 3:
- 5 từ: 'mở', 'đóng', 'lên', 'xuống', 'dừng'
- Mỗi từ có 6 file huấn luyện (train) và 2 file kiểm thử (test)
- Đa dạng hóa giọng đọc (Nam, Nữ, Giọng cao, Giọng trầm, Nói nhanh, Nói chậm)
- Định dạng chuẩn: WAV mono, 16000 Hz, 16-bit PCM có silence pad 0.3s
"""

import os
import sys
import time
from pathlib import Path
from text_to_speech import synthesize_text_to_wav

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ánh xạ nhãn thư mục sang từ tiếng Việt
LABELS_DICT = {
    "mo": "mở",
    "dong": "đóng",
    "len": "lên",
    "xuong": "xuống",
    "dung": "dừng"
}

# Cấu hình đa dạng hóa giọng nói cho 6 mẫu train và 2 mẫu test
TRAIN_PROFILES = [
    {"voice": "female", "rate": 1.0, "pitch": 0.0},
    {"voice": "female_high", "rate": 1.05, "pitch": 2.5},
    {"voice": "male", "rate": 0.95, "pitch": -3.0},
    {"voice": "male_deep", "rate": 0.90, "pitch": -5.0},
    {"voice": "slow_female", "rate": 0.85, "pitch": 0.0},
    {"voice": "fast_male", "rate": 1.15, "pitch": -2.5},
]

TEST_PROFILES = [
    {"voice": "female", "rate": 1.0, "pitch": 1.0},        # Test 1: Nữ
    {"voice": "male", "rate": 1.0, "pitch": -4.0},          # Test 2: Nam (Speaker mismatch nhẹ)
]


def create_full_dataset(base_dir: str = "dataset"):
    """Tạo toàn bộ cấu trúc thư mục và sinh các file WAV cho Lab 3."""
    print("=================================================================")
    print("      KHỞI TẠO BỘ DỮ LIỆU ÂM THANH MẪU CHO LAB 3 (CSE457)       ")
    print("=================================================================")

    train_base = Path(base_dir) / "train"
    test_base = Path(base_dir) / "test"

    for label, text in LABELS_DICT.items():
        print(f"\n[Đang tạo dữ liệu cho từ: '{text}' (Folder: '{label}')]")

        # 1. Tạo tập Train (6 files / nhãn)
        label_train_dir = train_base / label
        label_train_dir.mkdir(parents=True, exist_ok=True)
        for idx, prof in enumerate(TRAIN_PROFILES, start=1):
            out_file = label_train_dir / f"{label}_{idx:02d}.wav"
            synthesize_text_to_wav(
                text=text,
                output_path=str(out_file),
                voice=prof["voice"],
                pitch_shift=prof["pitch"],
                rate=prof["rate"],
                silence_pad_sec=0.3
            )
            print(f"  ✓ Train {out_file.name} [{prof['voice']}, pitch={prof['pitch']:+0.1f}, rate={prof['rate']:.2f}]")
            time.sleep(0.1)

        # 2. Tạo tập Test (2 files / nhãn)
        label_test_dir = test_base / label
        label_test_dir.mkdir(parents=True, exist_ok=True)
        for idx, prof in enumerate(TEST_PROFILES, start=7):
            out_file = label_test_dir / f"{label}_{idx:02d}.wav"
            synthesize_text_to_wav(
                text=text,
                output_path=str(out_file),
                voice=prof["voice"],
                pitch_shift=prof["pitch"],
                rate=prof["rate"],
                silence_pad_sec=0.3
            )
            print(f"  ★ Test  {out_file.name} [{prof['voice']}, pitch={prof['pitch']:+0.1f}, rate={prof['rate']:.2f}]")
            time.sleep(0.1)

    print("\n" + "=" * 65)
    print(">> KHỞI TẠO BỘ DỮ LIỆU HOÀN TẤT!")
    print(f">> Cấu trúc thư mục đã lưu tại: '{os.path.abspath(base_dir)}'")
    print("=================================================================\n")


if __name__ == "__main__":
    create_full_dataset()
