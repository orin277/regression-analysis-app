import numpy as np


class Rank:
    @staticmethod
    def calc_ranks(data):
        data_sorted = np.sort(data)
        ranks = np.zeros((data_sorted.size))
        
        for i in range(ranks.size):
            number_of_occurrences = np.count_nonzero(data == data[i])
            rank = np.where(data_sorted == data[i])[0] + 1
            
            if number_of_occurrences == 1:
                ranks[i] = rank[0]
            else:
                ranks[i] = np.sum(rank) / rank.size
        return ranks