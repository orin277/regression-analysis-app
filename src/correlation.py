import numpy as np
import matplotlib.pyplot as plt
from pearson_coefficient import *
from kendall_coefficient import *
from spearman_coefficient import *
from correlation_relation import *

class Correlation:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.pearson_coefficient = PearsonCoefficient(np.copy(data1), np.copy(data2))
        self.spearman_coefficient = SpearmanCoefficient(np.copy(data1), np.copy(data2))
        self.kendall_coefficient = KendallCoefficient(np.copy(data1), np.copy(data2))
        self.correlation_relation = CorrelationRelation(np.copy(data1), np.copy(data2))
        
    def draw_scatterplot(self):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot()
        ax.scatter(self.data1, self.data2)
        ax.grid()
        return fig