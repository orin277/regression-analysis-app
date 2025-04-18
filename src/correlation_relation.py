import numpy as np
from scipy import stats
from selection import *
from pearson_coefficient import *


class CorrelationRelation:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.reformat_array()
        self.calc_coefficient()
        self.calc_stats()
        
    def reformat_array(self):
        self.number_of_classes = round(1 + 1.44 * np.log(self.data1.size))
        self.quantile_fisher = stats.f.ppf(0.95, self.number_of_classes - 1, self.data1.size - self.number_of_classes)

        data_min = np.amin(self.data1)
        data_max = np.amax(self.data1)
        class_width = (data_max - data_min) / self.number_of_classes
        
        for i in range(0, self.number_of_classes):
            class1 = data_min + i * class_width
            class2 = data_min + (i + 1) * class_width
            x = 0.5 * (class1 + class2)
            if i != self.number_of_classes - 1:
                self.data1[(self.data1 >= class1) & (self.data1 < class2)] = x
            else:
                self.data1[(self.data1 >= class1) & (self.data1 <= class2)] = x
                
    def calc_coefficient(self):
        data2_average = Selection.calc_average(self.data2)
        
        data1_unique = np.unique(self.data1)
        sum1 = 0
        sum2 = 0
        for i in data1_unique:
            idx = np.where(self.data1 == i)[0]
            group = self.data2[idx]
            group_average = Selection.calc_average(group)
            sum1 += group.size * (group_average - data2_average)**2
            sum2 += np.sum([(y - data2_average)**2 for y in group])
        
        self.value = np.sqrt(sum1 / sum2)

    def test_for_equality_of_pearson_coefficient(self):
        self.pearson_coefficient = PearsonCoefficient(self.data1, self.data2)
        value_squared = self.value**2
        pearson_coefficient_squared = self.pearson_coefficient.value**2
        self.stats2 = ((value_squared - pearson_coefficient_squared) / (self.number_of_classes - 2)) / ((1 - value_squared) / (self.data1.size - self.number_of_classes))
        
    def calc_stats(self):
        self.stats = (self.value**2 / (self.number_of_classes - 1)) / ((1 - self.value**2) / (self.data1.size - self.number_of_classes))
        
    def determine_presence_of_connection(self):
        return np.abs(self.stats) > self.quantile_fisher

    def determine_linear_relationship(self):
        return np.abs(self.stats2) <= self.quantile_fisher