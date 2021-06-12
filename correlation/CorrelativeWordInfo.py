class KeyWordInfo(object):
    def __init__(self, text: str) -> None:
        self.text = text
        self.word_len = len(text)
        self.single_fq = 0

    def update_single_fq(self) -> None:
        self.single_fq += 1


class CorrelativeWordInfo(KeyWordInfo):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.union_fq = 0
        self.pmi = 0.0
        self.degree = 0.0

    def update_union_fq(self) -> None:
        self.union_fq += 1

    def compute_pmi(self, core_word_fq: int, doc_length: int) -> None:
        # 计算两个词之间的互信息只需要用频次算，然后在结果上除以文档长度即可
        self.pmi = self.union_fq/self.single_fq/core_word_fq/doc_length

    def compute_degree(self, sum_of_pmi: float) -> None:
        self.degree = self.pmi/sum_of_pmi
