import os
import re
from typing import Iterable
import pandas as pd
from pyhanlp import HanLP
from CorrelativeWordInfo import KeyWordInfo, CorrelativeWordInfo
from utils.util import load_words_from_dir


class CorrelativeWordDiscover(object):
    def __init__(self, min_pmi: float,
                 min_degree: float,
                 num: int = None,
                 replaced: str = None,
                 sep: str = None,
                 dic_dir: str = None
                 ) -> None:
        self._min_pmi = min_pmi
        self._min_degree = min_degree
        self._num = num
        self.replaced = replaced if replaced else \
        "[A-Z\\x20\\t\\f\\d<>/:'\"\\[\\]{}()\\|~@#$%^&*\\-_=+《》、：“”‘’｛｝【】（）￥—┄－]+"
        self.sep = sep if sep else '[，。？！；,.?!;…\n]'
        self.dic_dir = dic_dir if dic_dir else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dictionary')

    def discover(self, keyword: str, corpus: Iterable)-> pd.DataFrame:
        raw = ''.join(list(map(lambda line: line, corpus)))
        raw = re.split(self.sep, re.sub(self.replaced, '\0', raw))
        trie = load_words_from_dir(self.dic_dir)
        processed = list(map(
            lambda s: list(filter(
                lambda word: word != '\0' and not trie.get(word), 
                map(lambda term: term.word, s)
                )), 
            map(lambda sentence: HanLP.segment(sentence), raw)
            ))

        keyword_info = KeyWordInfo(keyword)
        candidates = dict()
        total_length = 0
        for sentence in processed:
            if keyword in sentence:
                keyword_info.update_single_fq()
                sentence.remove(keyword)
                for word in sentence:
                    info = candidates.get(word)
                    if info is None:
                        info = CorrelativeWordInfo(word)
                        candidates[word] = info
                    candidates[word].update_union_fq()
            total_length += len(sentence)

        for sentence in processed:
            for word in sentence:
                if word in candidates.keys():
                    candidates[word].update_single_fq()

        sum_of_pmi = 0
        for info in candidates.values():
            info.compute_pmi(keyword_info.single_fq, total_length)
            if info.pmi > self._min_pmi:
                sum_of_pmi += info.pmi

        for info in candidates.values():
            if info.pmi > self._min_pmi:
                info.compute_degree(sum_of_pmi)

        output = map(
            lambda w: (w.text, w.degree),
            filter(lambda w: w.degree > self._min_degree and w.word_len > 1, candidates.values())
        )
        
        return pd.DataFrame(output, columns=['word', 'degree']).sort_values('degree', ascending=0).iloc[0: self._num]
