from scipy import stats


class FTest:
    def __init__(self, r, s, n):
        self.quantile_fisher = stats.f.ppf(0.95, 1, n - s)
        self.f = (r / (1 - r)) * ((n - s) / (s - 1))
        
    def check_insignificance(self):
        return self.f <= self.quantile_fisher