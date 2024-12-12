#!/usr/bin/env python3
""" plot all 5 previous graphs in one figure"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(8, 6))


    # figure 1
    plt.subplot(3, 2, 1)
    plt.plot(y0, 'r')
    plt.xlim(0, 10)

    # figure 2
    plt.subplot(3, 2, 2)
    plt.title("Men's Height vs Weight")
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.xticks(np.arange(60, 81, 10))
    plt.scatter(x1, y1, c='magenta')

    # figure 3
    plt.subplot(3, 2, 3)
    plt.title('Exponential Decay of C-14')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction remaining')
    plt.yscale('log')
    plt.xticks(np.arange(0, 30000, 10000))
    plt.xlim(0, 28650)
    plt.plot(x2, y2)

    # figure 4
    plt.subplot(3, 2, 4)
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.legend(loc='upper right')
    plt.plot(x3, y32, 'g-', label='Ra-226')
    plt.legend(loc='upper right')

    # figure 5
    plt.subplot(3, 2, (5, 6))
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 10))
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.tight_layout()
    plt.suptitle('All in One')
    plt.show()
