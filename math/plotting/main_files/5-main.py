#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

all_in_one = __import__('5-all_in_one').all_in_one

all_in_one()