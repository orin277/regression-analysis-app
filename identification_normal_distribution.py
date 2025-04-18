import numpy as np
from scipy import stats
from selection import *


class IdentificationNormalDistribution:
    def __init__(self, data):
        self.data = data
        self.quantile_normal = stats.norm.ppf((0.975))

        self.calc_skewness_statistics()
        self.calc_kurtosis_statistics()

    def calc_skewness_statistics(self):
        self.skewness_statistics = (Selection.calc_skewness_coefficient(self.data) - 0.0) / Selection.calc_standard_deviation_of_skewness_coefficient(self.data) 

    def calc_kurtosis_statistics(self):
        self.kurtosis_statistics = (Selection.calc_kurtosis_coefficient(self.data) - 0.0) / Selection.calc_standard_deviation_of_kurtosis_coefficient(self.data) 

    def identify_distribution(self):
        return np.abs(self.skewness_statistics) <= self.quantile_normal and np.abs(self.kurtosis_statistics) <= self.quantile_normal