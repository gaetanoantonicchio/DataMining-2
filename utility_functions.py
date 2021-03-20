import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def report(results, n_top=3):
    """
    Return the results n_top results of a grid search
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

def summarize(results): 
    name = []
    accuracy = []
    f1_score1 = [] 
    precision = []
    recall = []

    for result in results: 
        name.append(result['name'])
        accuracy.append(result['accuracy'])
        f1_score1.append(result['f1_score'])
        precision.append(result['precision'])
        recall.append(result['recall'])

    fig = go.Figure()

    fig.add_trace(go.Bar(x=name, y=accuracy, name='accuracy', marker_color='rgb(55, 83, 109)'))
    fig.add_trace(go.Bar(x=name, y=f1_score1, name='f1_score', marker_color='rgb(26, 118, 255)'))
    fig.add_trace(go.Bar(x=name, y=precision, name='precision', marker_color='rgb(0,255,0)'))
    fig.add_trace(go.Bar(x=name, y=recall, name='recall', marker_color='rgb(255,255,0)'))


    fig.show()