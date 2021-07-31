# 调用方式

```Python
HanLP.Config.ShowTermNature = False
stopwords_trie = load_words('./dictionary/stopwords.txt')
replaced = "【.*?】|\\(.*?\\)|（.*?）|[a-zA-Z0-8\\s,<>/?:;'\"\\[\\]{}()\\|~!@#$%^&*\\-_=+，。《》、？：；“”‘’｛｝【】（）…￥！—┄－「」→]+"    
with open('./corpus/test.txt', encoding='utf-8') as f:
    target_doc = list(map(
        lambda s: list(filter(
            lambda word: not stopwords_trie.get(word), 
            map(lambda term: term.word, s)
            )), 
        map(lambda s: HanLP.segment(re.sub(replaced, '', s)), f)
        ))

with open('./corpus/general-corpus.txt', encoding='utf-8') as f:
    common_doc = list(map(
        lambda s: list(filter(
            lambda word: not stopwords_trie.get(word), 
            map(lambda term: term.word, s))), 
        map(lambda s: HanLP.segment(re.sub(replaced, '', s)), f)
        ))

dic = Dictionary([target_doc, common_doc])

target_bow = dic.doc2bow(target_doc)
common_bow = dic.doc2bow(common_doc)
tf_idf = TfidfModel(dic)
print(tf_idf.discover(target_bow, [common_bow]))
```
