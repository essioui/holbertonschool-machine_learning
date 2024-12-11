#!/usr/bin/env python3
"""plot a histogram of student scores for a project:
"""
import matplotlib.pyplot as plt
import numpy as np


def frequency():
    """
    plot a histogram of student scores for a project:
    xticks: devices x-axices by bins array
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    bins = np.arange(0, 101, 10)
    plt.xticks(bins)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.show()
