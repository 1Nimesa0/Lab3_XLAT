"""
Module: lexicon_lm.py
Chức năng: Quản lý từ điển phát âm (Lexicon) và Mô hình ngôn ngữ (Language Model):
- Đọc và phân tích file lexicon.txt (ánh xạ từ sang chuỗi âm vị/phone)
- Xây dựng mô hình Unigram/Bigram có làm mịn Add-k (Add-k smoothing)
- Thực hiện tái xếp hạng Bayesian MAP (Acoustic Likelihood + LM Prior)
- Thử nghiệm các giá trị trọng số lambda_LM theo yêu cầu Mục 4.7 & 5.9
"""

import sys
import math
from typing import Dict, List, Tuple
from collections import defaultdict
from hmmlearn.hmm import GaussianHMM

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def parse_lexicon(lexicon_file: str) -> Dict[str, List[str]]:
    """
    Đọc file từ điển phát âm định dạng: WORD PHONE_1 PHONE_2 ... PHONE_K
    Trả về dictionary: {word: [phones]}
    """
    lexicon = {}
    with open(lexicon_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            word = parts[0]
            phones = parts[1:]
            lexicon[word] = phones
    return lexicon


class BigramLanguageModel:
    """
    Mô hình ngôn ngữ Bigram với Add-k Smoothing:
    P(w_cur | w_prev) = (C(w_prev, w_cur) + k) / (C(w_prev) + k * V)
    """
    def __init__(self, vocabulary: List[str], k: float = 0.5):
        self.vocab = sorted(list(set(vocabulary)))
        self.V = len(self.vocab)
        self.k = k
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.total_words = 0

    def fit(self, sentences: List[List[str]]):
        """Đếm tần suất xuất hiện unigram và bigram từ tập ngữ liệu huấn luyện."""
        for sent in sentences:
            for i, word in enumerate(sent):
                self.unigram_counts[word] += 1
                self.total_words += 1
                if i > 0:
                    prev_word = sent[i - 1]
                    self.bigram_counts[(prev_word, word)] += 1

    def prob(self, prev_word: str, cur_word: str) -> float:
        """Tính xác suất chuyển đổi từ prev_word sang cur_word có làm mịn."""
        c_prev = self.unigram_counts.get(prev_word, 0)
        c_bigram = self.bigram_counts.get((prev_word, cur_word), 0)
        prob = (c_bigram + self.k) / (c_prev + self.k * self.V)
        return prob

    def log_prob(self, prev_word: str, cur_word: str) -> float:
        """Tính log xác suất của bigram."""
        p = self.prob(prev_word, cur_word)
        return math.log(p) if p > 0 else -100.0


def map_rerank(
    features,
    prev_word: str,
    models: Dict[str, GaussianHMM],
    lm: BigramLanguageModel,
    lm_weight: float = 0.3
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Tái xếp hạng dự đoán theo nguyên lý Bayes:
    Score(W) = log P_A(X|W) / T + lambda_LM * log P_L(W | W_prev)
    - log P_A(X|W) / T: Điểm âm học chuẩn hóa theo số lượng frame T.
    - log P_L(W | W_prev): Điểm mô hình ngôn ngữ (ngữ cảnh từ phía trước).
    - lambda_LM: Trọng số cân bằng giữa âm học và ngôn ngữ.
    """
    T = len(features)
    details = {}

    for word, model in models.items():
        try:
            raw_acoustic = model.score(features)
        except Exception:
            raw_acoustic = -1e9

        norm_acoustic = raw_acoustic / T
        log_lm = lm.log_prob(prev_word, word)
        combined_score = norm_acoustic + lm_weight * log_lm

        details[word] = {
            "acoustic_norm": norm_acoustic,
            "lm_logprob": log_lm,
            "total_score": combined_score
        }

    best_word = max(details.keys(), key=lambda w: details[w]["total_score"])
    return best_word, details
