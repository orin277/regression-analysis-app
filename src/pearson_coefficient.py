import numpy as np
from scipy import stats
from selection import *


class PearsonCoefficient:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.quantile_student = stats.t(df=self.data1.size-2).ppf((0.975))
        self.calc_coefficient()
        self.calc_stats()
        self.calc_confidence_interval()
        
    def calc_coefficient(self):
        x_average = Selection.calc_average(self.data1)
        y_average = Selection.calc_average(self.data2)
        xy_average = Selection.calc_average(self.data1 * self.data2)
        
        x_std = Selection.calc_standard_deviation(self.data1, 0)
        y_std= Selection.calc_standard_deviation(self.data2, 0)
        
        self.value = (xy_average - x_average * y_average) / (x_std * y_std)
        
    def calc_stats(self):
        self.stats = (self.value * np.sqrt(self.data1.size - 2)) / np.sqrt(1 - self.value**2)
        
    def determine_presence_of_connection(self):
        return np.abs(self.stats) > self.quantile_student
    
    def calc_confidence_interval(self):
        quantile_normal = stats.norm.ppf((0.975))
        interval1 = self.value + (self.value * (1 - self.value**2)) / (2 * self.data1.size)
        interval2 = quantile_normal * ((1 - self.value**2) / np.sqrt(self.data1.size - 1))
        
        self.confidence_interval = np.array([interval1 - interval2, interval1 + interval2])