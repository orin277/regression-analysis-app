import numpy as np
from scipy import stats
from rank import *
from pearson_coefficient import *


class SpearmanCoefficient:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.quantile_student = stats.t(df=self.data1.size-2).ppf((0.975))
        self.calc_coefficient()
        self.calc_stats()
        
    def calc_coefficient(self):
        self.ranks1 = Rank.calc_ranks(self.data1)
        self.ranks2 = Rank.calc_ranks(self.data2)
        self.value = PearsonCoefficient(self.ranks1, self.ranks2).value
        
    def calc_stats(self):
        self.stats = (self.value * np.sqrt(self.data1.size - 2)) / np.sqrt(1 - self.value**2)
        
    def determine_presence_of_connection(self):
        return np.abs(self.stats) > self.quantile_student