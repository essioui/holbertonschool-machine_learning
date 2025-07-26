#!/usr/bin/env python3
"""
Analyze
"""
import pandas as pd


def analyze(df):
    """
    Takes a pd.DataFrame and:
        Computes statistics for all columns except the Timestamp column.
    Returns:
        new pd.DataFrame containing these statistics.
    """
    df = df.drop(columns=['Timestamp'])

    df_stats = pd.DataFrame({
        'count': df.count(),
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        '25%': df.quantile(0.25),
        '50%': df.median(),
        '75%': df.quantile(0.75),
        'max': df.max()
    })

    return df_stats.transpose()
