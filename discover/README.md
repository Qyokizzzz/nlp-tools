# 调用方式

```Python
nwd = NewWordDiscover(dic_dir='./dictionary')
with open('./corpus/medical-corpus.txt', encoding='utf-8') as f:
    words = nwd.discover(f)
print(words)
```
