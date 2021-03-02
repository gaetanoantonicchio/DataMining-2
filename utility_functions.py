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


def scatterPlot(x, y, algoName):
    """
    :param x: (Dataframe) containing features to plot
    :param y: (Dataframe) labels
    :param algoName: name of the algorithm (used for scatter plot title)
    :return: scatter plot
    """
    tempDF = pd.DataFrame(data=x.loc[:, 0:1], index=x.index)
    tempDF = pd.concat((tempDF, y), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", \
               data=tempDF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using " + algoName)
