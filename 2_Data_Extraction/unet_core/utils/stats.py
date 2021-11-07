import numpy as np
import scipy
import pandas as pd


def holm_bonferroni(p_values, alpha):

    array = np.array(p_values)
    order = array.argsort()
    idx = order.argsort()

    adjusted = [alpha/(len(p_values) - y) for y in idx]

    h = [x < y for (x,y) in zip(p_values,adjusted)]

    return h, adjusted


def multiple_wilcoxon(df, names, comparison_metric):

    p_values = []
    n_1 = []
    n_2 = []

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):

            d1 = df[df.method == n1][comparison_metric].values
            d2 = df[df.method == n2][comparison_metric].values

            results = scipy.stats.wilcoxon(d1, d2)

            p_values.append(results.pvalue)
            n_1.append(n1)
            n_2.append(n2)

    h, adjusted = holm_bonferroni(p_values, 0.05)

    df_out = pd.DataFrame({'method_1': n_1, 'method_2': n_2, 'p_value': p_values, 'h': h})

    return df_out