import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from selection import *
from pearson_coefficient import *


class UnivariateRegression:
    def __init__(self, data1, data2):
        self.x = np.copy(data1)
        self.y = np.copy(data2)
        self.quantile_student = stats.t(df=self.x.size-2).ppf((0.975))
        
        pearson_coefficient = PearsonCoefficient(self.x, self.y)
        x_std = Selection.calc_standard_deviation(self.x, 0)
        y_std = Selection.calc_standard_deviation(self.y, 0)
        x_average = Selection.calc_average(self.x)
        y_average = Selection.calc_average(self.y)
        
        self.a = {}
        self.b = {}
        self.b["value"] = pearson_coefficient.value * (y_std / x_std)
        self.a["value"] = y_average - self.b["value"] * x_average

        self.restored_regression = self.a["value"] + self.b["value"] * self.x
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
        
    def calc_residuals_variance(self):
        self.residuals_variance = np.sum(self.residuals ** 2) / (self.residuals.size - 2)
        
    def calc_confidence_interval(self):
        self.a["confidence_interval"] = np.array([self.a["value"] - self.quantile_student 
              * self.a["std"], self.a["value"] + self.quantile_student * self.a["std"]])
        self.b["confidence_interval"] = np.array([self.b["value"] - self.quantile_student 
              * self.b["std"], self.b["value"] + self.quantile_student * self.b["std"]]) 

    def calc_stats(self):
        self.a["stats"] = self.a["value"] / self.a["std"]
        self.b["stats"] = self.b["value"] / self.b["std"]
        
    def check_insignificance_a(self):
        return np.abs(self.a["stats"]) <= self.quantile_student
    
    def check_insignificance_b(self):
        return np.abs(self.b["stats"]) <= self.quantile_student
    
    def calc_regression_confidence_interval(self):
        x_average = Selection.calc_average(self.x)
        regression_std = np.sqrt(self.residuals_variance / self.y.size 
            + (self.b["std"] * (self.x - x_average)) ** 2)
        
        self.regression_confidence_interval = np.array([self.restored_regression 
            - self.quantile_student * regression_std, self.restored_regression 
            + self.quantile_student * regression_std])
        
    def calc_prediction_value_confidence_interval(self):
        x_average = Selection.calc_average(self.x)
        regression_std = np.sqrt(self.residuals_variance / self.y.size 
            + (self.b["std"] * (self.x - x_average)) ** 2)
        std = np.sqrt((regression_std ** 2) + self.residuals_variance)
        
        self.prediction_value_confidence_interval = np.array([self.restored_regression 
            - self.quantile_student * std, self.restored_regression + self.quantile_student * std])
        
    def draw_regression_line(self):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot()
        ax.scatter(self.x, self.y)
        
        sorted_indices = np.argsort(self.x)
        ax.plot(self.x[sorted_indices], self.restored_regression[sorted_indices], c = "C1")

        ax.plot(self.x[sorted_indices], self.regression_confidence_interval[0][sorted_indices], c = "C2")
        ax.plot(self.x[sorted_indices], self.regression_confidence_interval[1][sorted_indices], c = "C2")
        
        ax.plot(self.x[sorted_indices], self.prediction_value_confidence_interval[0][sorted_indices], c = "C3")
        ax.plot(self.x[sorted_indices], self.prediction_value_confidence_interval[1][sorted_indices], c = "C3")
        ax.grid()
        return fig