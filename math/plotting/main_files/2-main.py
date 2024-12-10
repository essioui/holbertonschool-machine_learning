#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

change_scale = __import__('2-change_scale').change_scale

change_scale()
