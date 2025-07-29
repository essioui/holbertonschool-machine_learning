#!/usr/bin/env python3
"""
Test file
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

availableShips = __import__('0-passengers').availableShips
ships = availableShips(4)
for ship in ships:
    print(ship)
