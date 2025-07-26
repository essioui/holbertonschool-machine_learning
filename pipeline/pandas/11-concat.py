#!/usr/bin/env python3
"""
Concat
"""
index = __import__('10-index').index
import pandas as pd


def concat(df1, df2):
    """
    Takes two pd.DataFrame objects and:
        Indexes both dataframes on their Timestamp columns.
        Includes all timestamps from df2 to and including timestamp 1417411920
        Concatenates the selected rows from df2 to the top of df1 (coinbase).
        Adds keys to the concatenated data, labeling the rows from df2 as
            bitstamp and the rows from df1 as coinbase.
    Returns
        the concatenated pd.DataFrame.
    """
    df2 = df2[df2['Timestamp'] <= 1417411920]

    df1 = index(df1)
    df2 = index(df2)

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return df
