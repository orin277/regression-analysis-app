import numpy as np
from scipy import stats
from univariate_regression import *
from selection import *
from pearson_coefficient import *


class UnivariateNonlinearRegression(UnivariateRegression):
    def __init__(self, data1, data2, a, b, restored_regression):
        self.x = np.copy(data1)
        self.y = np.copy(data2)
        self.quantile_student = stats.t(df=self.x.size-2).ppf((0.975))
        
        pearson_coefficient = PearsonCoefficient(self.x, self.y)
        x_std = Selection.calc_standard_deviation(self.x, 0)
        x_average = Selection.calc_average(self.x)
        
        self.a = {"value": a}
        self.b = {"value": b}
        self.restored_regression = restored_regression

        self.residuals = self.y - self.restored_regression
        self.calc_residuals_variance()
        
        self.a["std"] = np.sqrt(self.residuals_variance / self.x.size 
                    * (1 + (x_average ** 2) / x_std ** 2))
        self.b["std"] = np.sqrt(self.residuals_variance / (self.x.size * x_std ** 2))
        
        self.calc_confidence_interval()
        self.calc_stats()
        self.calc_regression_confidence_interval()
        self.calc_prediction_value_confidence_interval()
        self.determination_coefficient = pearson_coefficient.value ** 2