# 该算法用于查找给定关键字的相关词

## 简介

### WordInfo.py中包含两个类, 分别是KeyWordInfo和它的超类CorrelativeWordInfo  

#### KeyWordInfo用于保存关键字的相关信息  

属性包括: KeyWordInfo.text, KeyWordInfo.single_fq  
方法包括: KeyWordInfo.update_single_fq()  

#### CorrelativeWordInfo对KeyWordInfo进行了扩展，用于保存与其相关的词的信息  

扩展的属性包括: CorrelativeWordInfo.union_fq, CorrelativeWordInfo.pmi, CorrelativeWordInfo.degree  
扩展的方法包括: CorrelativeWordInfo.update_union_fq, CorrelativeWordInfo.compute_pmi, CorrelativeWordInfo.compute_degree  

### CorrelativeWordDiscover.py中只有CorrelativeWordDiscover类  

#### CorrelativeWordDiscover类的构造器接收三个参数

    min_pmi(float): 小于此阈值的词被认为与关键字无关, 用于归一化相关度, 否则即使是最相关的那个词, 相关度也很低  
    min_degree(float): 小于此相关度的词不输出  
    num(int): 只输出前num个词, 当输入语料足够大时, num为无用的参数, 否则可以选择使用此参数, 默认为None, 即全部输出  

#### CorrelativeWordDiscover.discover(keyword: str, corpus: str, replaced: str, sep: str, dictionary: str)方法用于完成相关词的查找

    keyword: 给定的关键字  
    corpus: 输入语料的路径  
    replaced: 为需要删除的无用字符的正则表达式, 默认为"[\\s\\d<>/:'\"\\[\\]{}()\\|~@#$%^&*\\-_=+《》、：“”‘’｛｝【】（）￥—┄－]+"  
    sep: 分隔符的正则表达式, 默认为'[，。,.；？！!?;…]'. 分隔后的前后两段被认为是相互独立的两句话, 算法效果依赖于此项  

## 调用方式

    ```Python
    cwd = CorrelativeWordDiscover(1.014528e-05, 0)
    with open('./corpus/test.txt', encoding='utf-8') as f:
        words = cwd.discover('巩膜', f)
    print(words)
    ```
