import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from selection import *


class MultivariateRegression:
    def __init__(self, data1, data2):
        self.x = np.copy(data1)
        self.x = np.insert(self.x, 0, 1, axis=1)
        self.y = np.copy(data2)
        
        self.calc_params()
        self.quantile_student = stats.t(df=self.y.size-self.params.size).ppf((0.975))
        self.calc_restored_regression()
        self.calc_residuals()
        self.calc_residuals_variance()
        self.calc_param_std()
        self.calc_param_confidence_interval()
        self.calc_param_stats()
        self.calc_regression_confidence_interval()
        self.calc_prediction_value_confidence_interval()
        self.calc_determination_coefficient()
        
    def calc_params(self):
        self.params = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y
        
    def calc_restored_regression(self):
        self.restored_regression = (self.x @ self.params.reshape(-1, 1)).flatten()
        
    def calc_residuals(self):
        self.residuals = self.y - self.restored_regression
        
    def calc_residuals_variance(self):
        sse = np.sum(self.residuals ** 2)
        self.residuals_variance = sse / (self.y.size - self.params.size)
        
    def calc_param_std(self):
        param_variances = self.residuals_variance * np.linalg.inv(self.x.T @ self.x)
        self.param_std = np.sqrt(param_variances.diagonal())
        
    def calc_param_confidence_interval(self):
        self.param_confidence_intervals = np.array([self.params - self.quantile_student 
            * self.param_std, self.params + self.quantile_student * self.param_std]).T
        
    def calc_param_stats(self):
        self.param_stats = self.params / self.param_std
        
    def check_insignificance(self):
        return np.abs(self.param_stats) <= self.quantile_student
    
    def calc_regression_confidence_interval(self):
        self.regression_confidence_interval = np.empty((2, self.y.size))

        for i in range(self.y.size):
            regression_variance = self.residuals_variance * self.x[i] \
                @ np.linalg.inv(self.x.T @ self.x) @ self.x[i].T
            
            self.regression_confidence_interval[0][i] = self.restored_regression[i] \
                - self.quantile_student * np.sqrt(regression_variance)
            self.regression_confidence_interval[1][i] = self.restored_regression[i] \
                + self.quantile_student * np.sqrt(regression_variance)
        
    def calc_prediction_value_confidence_interval(self):
        self.prediction_value_confidence_interval = np.empty((2, self.y.size))

        for i in range(self.y.size):
            regression_variance = self.residuals_variance * self.x[i] \
                @ np.linalg.inv(self.x.T @ self.x) @ self.x[i].T
            
            variance = regression_variance + self.residuals_variance
            
            self.prediction_value_confidence_interval[0][i] = self.restored_regression[i] \
                - self.quantile_student * np.sqrt(variance)
            self.prediction_value_confidence_interval[1][i] = self.restored_regression[i] \
                + self.quantile_student * np.sqrt(variance)
            
    def calc_determination_coefficient(self):
        y_std = Selection.calc_variance(self.y)
        
        self.determination_coefficient = 1 - ((self.y.size - self.params.size) \
          * self.residuals_variance) / ((self.y.size - 1) * y_std)
    
    def draw_residual_diagram(self):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot()
        ax.scatter(self.restored_regression, self.residuals)
        ax.grid()
        return fig