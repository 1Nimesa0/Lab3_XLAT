"""
Module: text_to_speech.py
Chức năng: Chuyển đổi văn bản tiếng Việt sang file âm thanh WAV chuẩn mono 16kHz, 16-bit PCM.
Hỗ trợ thay đổi giọng nói linh hoạt:
- Giới tính/Cao độ (Pitch): Nữ chuẩn, Nữ cao, Nam trầm, Nam vừa, Giọng trẻ em,...
- Tốc độ nói (Rate/Speed): Nhanh, bình thường, chậm.
- Thêm khoảng lặng silence (0.2s - 0.5s) ở đầu và cuối theo chuẩn yêu cầu Lab 3.
- Hỗ trợ cả gTTS (Google Speech Engine) và pyttsx3 (SAPI5 offline trên Windows).
"""

import os
import io
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS

# Đảm bảo in tiếng Việt trên console Windows không lỗi mã hóa
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thiết lập các cấu hình giọng nói mẫu
VOICE_PROFILES = {
    "female": {"pitch_shift": 0.0, "rate": 1.0, "desc": "Giọng nữ tiêu chuẩn"},
    "female_high": {"pitch_shift": 3.0, "rate": 1.05, "desc": "Giọng nữ cao / trẻ trung"},
    "male": {"pitch_shift": -3.5, "rate": 0.95, "desc": "Giọng nam tự nhiên"},
    "male_deep": {"pitch_shift": -5.5, "rate": 0.90, "desc": "Giọng nam trầm, dày"},
    "child": {"pitch_shift": 4.5, "rate": 1.10, "desc": "Giọng trẻ em"},
    "slow_female": {"pitch_shift": 0.0, "rate": 0.80, "desc": "Giọng nữ nói chậm, rõ ràng"},
    "fast_male": {"pitch_shift": -3.0, "rate": 1.25, "desc": "Giọng nam nói nhanh"},
}


def synthesize_text_to_wav(
    text: str,
    output_path: str,
    voice: str = "female",
    pitch_shift: float = None,
    rate: float = None,
    target_sr: int = 16000,
    silence_pad_sec: float = 0.3,
    gain_db: float = 0.0,
    lang: str = "vi"
) -> str:
    """
    Chuyển văn bản thành file WAV 16kHz 16-bit PCM, điều chỉnh giọng nói và thêm silence.

    Parameters:
    -----------
    text : str
        Văn bản cần đọc (VD: 'mở', 'đóng', 'lên', 'xuống', 'dừng').
    output_path : str
        Đường dẫn file .wav xuất ra.
    voice : str
        Tên profile giọng trong VOICE_PROFILES ('female', 'male', 'male_deep', v.v.)
    pitch_shift : float, optional
        Số bán cung (semitones) muốn dịch: âm là trầm hơn (nam), dương là cao hơn (nữ).
        Nếu truyền vào sẽ ghi đè giá trị của voice profile.
    rate : float, optional
        Tỷ lệ tốc độ nói (>1 là nhanh hơn, <1 là chậm hơn).
        Nếu truyền vào sẽ ghi đè giá trị của voice profile.
    target_sr : int
        Tần số lấy mẫu mục tiêu (chuẩn ASR Lab 3 là 16000 Hz).
    silence_pad_sec : float
        Độ dài khoảng lặng (giây) chèn vào đầu và cuối file (chuẩn Lab 3: 0.2 - 0.5s).
    gain_db : float
        Độ khuếch đại âm lượng (dB).
    lang : str
        Mã ngôn ngữ (mặc định 'vi' cho tiếng Việt).

    Returns:
    --------
    output_path : str
        Đường dẫn file WAV đã được lưu thành công.
    """
    profile = VOICE_PROFILES.get(voice, VOICE_PROFILES["female"])
    actual_pitch = pitch_shift if pitch_shift is not None else profile["pitch_shift"]
    actual_rate = rate if rate is not None else profile["rate"]

    # 1. Sinh âm thanh gốc bằng gTTS
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    # 2. Đọc vào mảng NumPy và resample về đúng 16kHz mono
    y, sr = librosa.load(fp, sr=target_sr, mono=True)

    # 3. Thay đổi tốc độ nói (Time Stretch) nếu rate khác 1.0
    if abs(actual_rate - 1.0) > 1e-3:
        # librosa.effects.time_stretch thay đổi độ dài thời gian mà giữ nguyên cao độ cơ bản
        y = librosa.effects.time_stretch(y, rate=actual_rate)

    # 4. Thay đổi cao độ (Pitch Shift) để biến đổi giọng (Nam <-> Nữ <-> Trẻ em)
    if abs(actual_pitch) > 1e-3:
        y = librosa.effects.pitch_shift(y, sr=target_sr, n_steps=actual_pitch)

    # 5. Cắt bỏ khoảng lặng thừa gốc trong file sinh ra (nếu có)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > 0:
        y = y_trimmed

    # 6. Chuẩn hóa biên độ âm thanh (Normalize peak amplitude)
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-6:
        y = y / max_amp * 0.90

    # Áp dụng khuếch đại gain_db nếu có
    if abs(gain_db) > 1e-3:
        scale = 10.0 ** (gain_db / 20.0)
        y = np.clip(y * scale, -1.0, 1.0)

    # 7. Chèn khoảng lặng chuẩn ở đầu và cuối (0.2s - 0.5s để kiểm tra VAD/Endpoint)
    if silence_pad_sec > 0:
        pad_len = int(silence_pad_sec * target_sr)
        pad = np.zeros(pad_len, dtype=np.float32)
        y = np.concatenate([pad, y, pad])

    # 8. Lưu dưới định dạng chuẩn WAV 16-bit PCM mono 16000 Hz
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, y, target_sr, subtype="PCM_16")

    return output_path


def list_available_voices():
    """Liệt kê các kiểu giọng mẫu có sẵn."""
    print("\n--- Danh sách các giọng nói mẫu (Voice Profiles) ---")
    for key, p in VOICE_PROFILES.items():
        print(f" • '{key:12s}': Pitch={p['pitch_shift']:+4.1f} st, Rate={p['rate']:4.2f}x | {p['desc']}")
    print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển văn bản tiếng Việt sang file WAV với giọng nói tùy chỉnh.")
    parser.add_argument("--text", type=str, default="mở", help="Văn bản cần đọc (VD: 'mở', 'đóng', 'dừng')")
    parser.add_argument("--output", type=str, default="output.wav", help="Đường dẫn file WAV xuất ra")
    parser.add_argument("--voice", type=str, default="male", choices=list(VOICE_PROFILES.keys()), help="Chọn loại giọng")
    parser.add_argument("--pitch", type=float, default=None, help="Độ dịch cao độ tùy chỉnh (semitones: -5 -> +5)")
    parser.add_argument("--rate", type=float, default=None, help="Hệ số tốc độ nói tùy chỉnh (0.7 -> 1.4)")
    parser.add_argument("--pad", type=float, default=0.3, help="Khoảng lặng đầu/cuối (giây, mặc định 0.3s)")
    parser.add_argument("--list-voices", action="store_true", help="Hiển thị danh sách giọng mẫu")

    args = parser.parse_args()

    if args.list_voices:
        list_available_voices()
    else:
        out = synthesize_text_to_wav(
            text=args.text,
            output_path=args.output,
            voice=args.voice,
            pitch_shift=args.pitch,
            rate=args.rate,
            silence_pad_sec=args.pad
        )
        print(f"[THÀNH CÔNG] Đã tạo file: '{out}' | Văn bản: '{args.text}' | Kiểu giọng: {args.voice}")
