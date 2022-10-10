import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy import trapz
import plotly

football_players = np.random.randn(500) * 20 + 160  # футболисты, класс negative - 0
basketball_players = np.random.randn(500) * 10 + 190  # баскетболисты, класс positive - 1

max_height = 230
height = 180


def height_separator(height, players):
    return [0 if pl < height else 1 for pl in players]


def Accuracy(negative, positive):
    n = len(negative) + len(positive)
    TP = positive.count(1)
    TN = negative.count(0)
    A = (TP + TN) / n
    return A


def Precision(negative, positive):
    TP = positive.count(1)
    if (TP == 0):
        return 1
    FP = negative.count(1)
    P = TP / (TP + FP)
    return P


def Recall(positive):
    TP = positive.count(1)
    FN = positive.count(0)
    R = TP / (TP + FN)
    return R


def alpha(negative):
    FP = negative.count(1)
    TN = negative.count(0)
    a = FP / (TN + FP)
    return a


def betta(positive):
    TP = positive.count(1)
    FN = positive.count(0)
    b = FN / (FN + TP)
    return b


def F1_score(presicion, recall):
    f = 2 * (recall * presicion) / (recall + presicion)
    return f


precision, recall, accuracy_list, threshold = [], [], [], []
I, J = [], []

fig = px.line(x=Recall, y=Precision,
              title=f"График Presicion-Recall, AUC={calcArea(recall, precision)}",
              labels=dict(x="Recall", y="Presicion", hover_data_0="Accuracy", hover_data_1="Threshold"),
              hover_data=[accuracy_list, threshold])
plotly.offline.plot(fig, filename=f'C:/plotly/PR.html')

fig2 = px.line(x=I, y=J,
               title=f"График кривой, AUC = {calcArea(I, J)}",
               labels=dict(x="I", y="J"))
plotly.offline.plot(fig2, filename=f'C:/plotly/1.html')

