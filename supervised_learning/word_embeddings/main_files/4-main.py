#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gensim.test.utils import common_texts
fasttext_model = __import__('4-fasttext').fasttext_model

print(common_texts[:2])
ft = fasttext_model(common_texts, min_count=1)
print(ft.wv["computer"])
