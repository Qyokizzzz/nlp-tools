import sys
import math
from typing import Dict, Any


class WordInfo(object):
    def __init__(self, text: str) -> None:
        self.left = dict()  # Dict[str, int]
        self.right = dict()  # Dict[str, int]
        self.text = text
        self.word_len = len(text)
        self.fq = 0  # 频数
        self.freq = 0.0  # 频率
        self.left_entropy = 0.0
        self.right_entropy = 0.0
        self.pmi = sys.float_info.max
        self.entropy = 0.0
        self.score = 0.0

    def __str__(self) -> str:
        return self.text + '\t' + str(self.word_len) + '\t' + str(self.fq) + '\t' + \
               str(self.left_entropy) + '\t' + str(self.right_entropy) + '\t' + str(self.pmi) + '\t' + \
               str(self.entropy) + '\t' + str(self.score)

    @staticmethod
    def _update_fq(c: str, storage: Dict[str, int]) -> None:
        if storage.get(c) is None:
            storage[c] = 1
        else:
            storage[c] += 1

    def update(self, left: str, right: str) -> None:
        self.fq += 1
        self._update_fq(left, self.left)
        self._update_fq(right, self.right)

    def _compute_entropy(self, storage: Dict[str, int]) -> float:
        sum_of_entropy = 0.0
        for k in storage:
            p = storage[k] / self.fq
            sum_of_entropy -= p * math.log(p, 2)
        return sum_of_entropy

    def compute_freq_entropy(self, length: int) -> None:
        self.freq = self.fq / length
        self.left_entropy = self._compute_entropy(self.left)
        self.right_entropy = self._compute_entropy(self.right)
        self.entropy = min(self.left_entropy, self.right_entropy)

    def compute_pmi(self, candidates: Dict[str, Any]) -> None:  # Dict[str, WordInfo]
        if len(self.text) == 1:
            self.pmi = math.sqrt(self.freq)
        else:
            for i in range(1, len(self.text)):
                self.pmi = min(self.pmi,
                               self.freq / candidates.get(self.text[0: i]).freq / candidates.get(self.text[i:]).freq)

    def compute_score(self):
        try:
            self.score = 2 * self.pmi * self.entropy / (self.pmi + self.entropy)
        except ZeroDivisionError:
            self.score = 0
