import re
import os
from typing import Tuple, Iterable
import pandas as pd
from WordInfo import WordInfo
from utils.util import load_words_from_dir


class NewWordDiscover(object):
    def __init__(
            self,
            max_word_len: int = 6,
            min_freq: float = 0.0,
            min_entropy: float = 1.0,
            min_pmi: float = 2.0,
            min_score: float = 3.0,
            dic_dir: str = None,
            dic_encoding: str = 'utf-8'
    ):
        self._max_word_len = max_word_len
        self._min_freq = min_freq
        self._min_entropy = min_entropy
        self._min_pmi = min_pmi
        self._min_score = min_score
        self.dic_dir = dic_dir if dic_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dictionary')
        self.dic_encoding = dic_encoding

    def discover(self, file: Iterable, replaced: str = None) -> pd.DataFrame:
        candidates = dict()
        total_length = 0
        if replaced is None:
            replaced = "[\\s\\d,.<>/?:;'\"\\[\\]{}()\\|~!@#$%^&*\\-_=+，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+"
            
        for line in file:
            doc = re.sub(replaced, '\0', line)
            doc_length = len(doc)
            for i in range(doc_length):
                end = min(i + 1 + self._max_word_len, doc_length + 1)
                for j in range(i + 1, end):
                    word = doc[i: j]
                    if '\0' in word:
                        continue
                    info = candidates.get(word)
                    if info is None:
                        info = WordInfo(word)
                        candidates[word] = info
                    left = '\0' if i == 0 else doc[i-1]
                    right = doc[j] if j < doc_length else '\0'
                    info.update(left, right)
            total_length += doc_length

        for info in candidates.values():
            info.compute_freq_entropy(total_length)
        for info in candidates.values():
            info.compute_pmi(candidates)
            info.compute_score()

        rmtrie = load_words_from_dir(self.dic_dir, self.dic_encoding)
        res_iter = map(
            lambda info: (info.text, info.word_len, info.fq, info.left_entropy, info.right_entropy, info.pmi, info.score),
            filter(
                lambda info: not rmtrie.get(info.text) 
                and 1 < info.word_len <= self._max_word_len and info.freq >= self._min_freq 
                and info.entropy >= self._min_entropy and info.pmi >= self._min_pmi and info.score >= self._min_score, 
                candidates.values()
                )
            )
        return pd.DataFrame(
            res_iter, 
            columns=['word', 'word_len', 'freq', 'left_entropy', 'right_entropy', 'pmi', 'score']
            ).sort_values('score', ascending=0)
