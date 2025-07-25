#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from_file = __import__('2-from_file').from_file
prune = __import__('8-prune').prune

df = from_file('../datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = prune(df)

print(df.head())
