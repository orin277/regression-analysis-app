import numpy as np
from scipy import stats
from rank import *


class KendallCoefficient:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.quantile_normal = stats.norm.ppf((0.975))
        self.calc_coefficient()
        self.calc_stats()
    
    def calc_coefficient(self):
        self.ranks1 = Rank.calc_ranks(self.data1)
        self.ranks2 = Rank.calc_ranks(self.data2)

        quantity_in_bundle1 = np.unique(self.ranks1, return_counts=True)[1]
        quantity_in_bundle2 = np.unique(self.ranks2, return_counts=True)[1]
        quantity_in_bundle1 = quantity_in_bundle1[quantity_in_bundle1 > 1]
        quantity_in_bundle2 = quantity_in_bundle2[quantity_in_bundle2 > 1]

        idx = np.argsort(self.ranks1)
        self.ranks1, self.ranks2 = self.ranks1[idx], self.ranks2[idx]
        
        v = 0
        if (quantity_in_bundle1.size == 0 and quantity_in_bundle2 == 0):
            for i in range(self.ranks2.size):
                for j in range(i, self.ranks2.size):
                    if self.ranks2[i] < self.ranks2[j]:
                        v += 1
                    elif self.ranks2[i] > self.ranks2[j]:
                        v -= 1
            self.value = (2 * v) / (self.data1.size * (self.data1.size - 1))
        else:
            for i in range(self.ranks2.size):
                for j in range(i, self.ranks2.size):
                    if self.ranks2[i] < self.ranks2[j] and self.ranks1[i] != self.ranks1[j]:
                        v += 1
                    elif self.ranks2[i] > self.ranks2[j] and self.ranks1[i] != self.ranks1[j]:
                        v -= 1
                        
            c = 0.5 * np.sum([x * (x - 1) for x in quantity_in_bundle1])
            d = 0.5 * np.sum([x * (x - 1) for x in quantity_in_bundle2])
            self.value = v / np.sqrt((0.5 * self.data1.size * (self.data1.size - 1) - c) 
                     * (0.5 * self.data1.size * (self.data1.size - 1) - d))

    def calc_stats(self):
        self.stats = (3 * self.value * np.sqrt(self.data1.size * (self.data1.size - 1))) / np.sqrt(2 * (2 * self.data1.size + 5))
        
    def determine_presence_of_connection(self):
        return np.abs(self.stats) > self.quantile_normal