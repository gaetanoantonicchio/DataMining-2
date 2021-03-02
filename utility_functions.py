import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def report(results, n_top=3):
    """
    Return the results n_top results of a grid search
    :param results:
    :param n_top:
    :return:
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

