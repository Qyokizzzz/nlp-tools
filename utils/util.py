import re
import os
import operator
import time
from functools import reduce
from typing import List, Tuple
from pyhanlp import AhoCorasickDoubleArrayTrie, JClass


def load_text_by_line(load_path: str, encoding: str = 'utf-8') -> List[str]:
    with open(load_path, encoding=encoding) as f:
        return list(map(lambda text: text.strip(), f))


def read_filepaths_from_dir(dir_name: str) ->List[str]:
    walk = os.walk(dir_name)
    for root, _, files in walk:
        return list(map(lambda name: os.path.join(root, name), files))


def load_text_from_dir(dir_name: str, encoding: str = 'utf-8') -> List[List[str]]:
    text_set = []
    for filepath in read_filepaths_from_dir(dir_name):
        text_set.append(load_text_by_line(filepath, encoding=encoding))  
    return text_set


def load_words(filepath: str, encoding: str = 'utf-8') -> AhoCorasickDoubleArrayTrie:
    tree_map = JClass('java.util.TreeMap')()
    with open(filepath, encoding=encoding) as f:
        for word in f:
            tree_map[word.strip()] = True
    return AhoCorasickDoubleArrayTrie(tree_map)


def load_words_from_dir(dir_name: str, encoding: str = 'utf-8') -> AhoCorasickDoubleArrayTrie:
    tree_map = JClass('java.util.TreeMap')()
    for word in reduce(operator.add, load_text_from_dir(dir_name, encoding)):
        tree_map[word] = True
    return AhoCorasickDoubleArrayTrie(tree_map)


def timer(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('The %s function run time is %s second(s)' % (func.__name__, (stop_time - start_time)))
        return res
    return inner
 