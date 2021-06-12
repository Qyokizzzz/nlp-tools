import pandas as pd
from typing import List
from TFIDFWordInfo import WordInfo, Dictionary


class TfidfModel(object):
    def __init__(self, dictionary: Dictionary, min_tf: float = 0.00, min_tfidf: float = 0.001, top_n: int = None) -> None:
        self.dictionary = dictionary
        self._min_tf = min_tf
        self._min_tfidf = min_tfidf
        self._top_n = top_n
    
    def discover(self, doc_bow: List[int], common_docs_bow: List[List[int]]):
        words = dict()
        for doc in common_docs_bow:
            for i in range(len(doc_bow)):
                if doc_bow[i] > 0:
                    info = words.get(i)
                    if info is None:
                        info = WordInfo(i)
                        words[i] = info
                    if doc[i] > 0:
                        info.update_docs_n()
                    
        total = sum(doc_bow)    
        for info in words.values():
            info.update_fq(doc_bow[info.word_idx])
            info.compute_tf(total)
            info.compute_idf(len(common_docs_bow)+1)
            info.compute_tfidf()
        
        res_iter = map(
            lambda info: (self.dictionary[info.word_idx], info.fq, info.docs_n, info.tf_idf), 
            filter(
                lambda info: len(self.dictionary[info.word_idx]) > 1 
                and info.tf >= self._min_tf and info.tf_idf >= self._min_tfidf,
                words.values()
                )
            )
        return pd.DataFrame(
            res_iter,
            columns=['word', 'fq', 'docs_n', 'tf-idf'] 
            ).sort_values('tf-idf', ascending=0).iloc[0: self._top_n]
      