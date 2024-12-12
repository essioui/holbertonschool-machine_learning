#!/usr/bin/env python3
"""plot a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    peoples: columns of persons
    fruit[]: has name of fruits by rows
    plt.bar(x, height, width, bottom, color, label):
        x: len(peoples)
        height: length of rows
        width: 0.5 (by default 0.8)
        bottom: order of columns at y-axis
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Data from tasks
    peoples = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    bar_width = 0.5

    plt.bar(peoples, apples, width=bar_width, color='red', label='apples')
    plt.bar(peoples, bananas, width=bar_width,
            bottom=apples, color='yellow', label='bananas')
    plt.bar(peoples, oranges, width=bar_width, bottom=apples+bananas,
            color='#ff8000', label='oranges')
    plt.bar(peoples, peaches, width=bar_width, bottom=apples+bananas+oranges,
            color='#ffe5b4', label='peaches')

    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.title('Number of Fruit per Person')

    plt.ylim(0, 80)
    plt.show()
