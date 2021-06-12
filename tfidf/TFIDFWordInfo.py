import operator
from functools import reduce
from math import log
from typing import List


class Dictionary(object):
    def __init__(self, docs: List[List[List[str]]]) -> None:
        # 输入分词后的文档
        self.word_set = set(reduce(operator.add, sum(docs, [])))
        self.word_map = list(self.word_set)

    def doc2bow(self, doc: List[List[str]]) -> List[int]:
        # 返回一篇文档的词袋向量
        word_freq = dict(zip(self.word_set, [0 for _ in range(len(self.word_set))]))   
        for word in reduce(operator.add, doc):
            try:
                word_freq[word] += 1
            except KeyError:
                raise Exception('No such word in dictionary')
        return list(map(lambda word: word_freq[word], self.word_map))
    
    def __getitem__(self, bow_idx: int) -> str:
        return self.word_map[bow_idx]
    
    def __len__(self) -> int:
        return len(self.word_set)
    
    
class WordInfo(object):
    def __init__(self, word_idx):
        self.word_idx = word_idx
        self.fq = 0
        self.docs_n = 0
        self.tf = 0.00
        self.idf = 0.00
        self.tf_idf = 0.00
    
    def update_fq(self, n: int) -> None:
        self.fq += n
    
    def update_docs_n(self) -> None:
        self.docs_n += 1
    
    def compute_tf(self, total: int) -> None:
        self.tf = self.fq / total
    
    def compute_idf(self, docs_total) -> None:
        self.idf = log(docs_total / (self.docs_n + 1), 2)
    
    def compute_tfidf(self) -> None:
        self.tf_idf = self.tf * self.idf
