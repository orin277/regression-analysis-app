import numpy as np
from scipy import stats


class Selection:
    def __init__(self, data):
        self.data = data
        quantile_student = stats.t(df=data.size-1).ppf((0.975))
        quantile_normal = stats.norm.ppf((0.975))
        
        standard_deviation = Selection.calc_standard_deviation(data, 1)
        average = Selection.calc_average(data)
        standard_deviation_of_average = Selection.calc_standard_deviation_of_average(data, standard_deviation)
        confidence_interval_of_average = Selection.calc_confidence_interval_of_average(average, standard_deviation_of_average, quantile_student)                                   
        self.average = {"value": average, 
                        "standard_deviation": standard_deviation_of_average, 
                         "confidence_interval": confidence_interval_of_average}
                                                  
        median = Selection.calc_median(data)
        confidence_interval_of_median = Selection.calc_confidence_interval_of_median(data, quantile_normal)                                   
        self.median = {"value": median, "confidence_interval": confidence_interval_of_median}
        
        standard_deviation_of_standard_deviation = Selection.calc_standard_deviation_of_standard_deviation(data, standard_deviation)
        confidence_interval_of_standard_deviation = Selection.calc_confidence_interval_of_standard_deviation(standard_deviation, standard_deviation_of_standard_deviation, quantile_student)                                   
        self.standard_deviation = {"value": standard_deviation, 
                        "standard_deviation": standard_deviation_of_standard_deviation, 
                         "confidence_interval": confidence_interval_of_standard_deviation}
        
        skewness_coefficient = Selection.calc_skewness_coefficient(data)
        standard_deviation_of_skewness_coefficient = Selection.calc_standard_deviation_of_skewness_coefficient(data)
        confidence_interval_of_skewness_coefficient = Selection.calc_confidence_interval_of_skewness_coefficient(skewness_coefficient, standard_deviation_of_skewness_coefficient, quantile_student)                                   
        self.skewness_coefficient = {"value": skewness_coefficient, 
                        "standard_deviation": standard_deviation_of_skewness_coefficient, 
                         "confidence_interval": confidence_interval_of_skewness_coefficient}
        
        kurtosis_coefficient = Selection.calc_kurtosis_coefficient(data)
        standard_deviation_of_kurtosis_coefficient = Selection.calc_standard_deviation_of_kurtosis_coefficient(data)
        confidence_interval_of_kurtosis_coefficient = Selection.calc_confidence_interval_of_kurtosis_coefficient(kurtosis_coefficient, standard_deviation_of_kurtosis_coefficient, quantile_student)                                   
        self.kurtosis_coefficient = {"value": kurtosis_coefficient, 
                        "standard_deviation": standard_deviation_of_kurtosis_coefficient, 
                         "confidence_interval": confidence_interval_of_kurtosis_coefficient}
        
        counterkurtosis_coefficient = Selection.calc_counterkurtosis_coefficient(data)
        standard_deviation_of_counterkurtosis_coefficient = Selection.calc_standard_deviation_of_counterkurtosis_coefficient(data, kurtosis_coefficient)
        confidence_interval_of_counterkurtosis_coefficient = Selection.calc_confidence_interval_of_counterkurtosis_coefficient(counterkurtosis_coefficient, standard_deviation_of_counterkurtosis_coefficient, quantile_student)                                   
        self.counterkurtosis_coefficient = {"value": counterkurtosis_coefficient, 
                        "standard_deviation": standard_deviation_of_counterkurtosis_coefficient, 
                         "confidence_interval": confidence_interval_of_counterkurtosis_coefficient}
        
        self.minimum = np.amin(data)
        self.maximum = np.amax(data)
    
    @staticmethod
    def calc_average(data):
        return np.sum(data) / data.size

    @staticmethod
    def calc_median(data):
        median = 0
        sorted_data = np.sort(data)
        if data.size % 2 == 0:
            median = (sorted_data[int((data.size / 2) - 1)] + sorted_data[int(data.size / 2)]) / 2
        else:
            median = sorted_data[int(data.size / 2)]
        return median
    
    @staticmethod
    def calc_standard_deviation(data, ddof=1):
        return np.sqrt(Selection.calc_variance(data, ddof))
    
    @staticmethod
    def calc_variance(data, ddof=1):
        # ddof = 0 - s^
        variance = 0
        average = Selection.calc_average(data)
        for i in data:
            variance += (i - average)**2
        if ddof == 1:
            variance /= (data.size - 1)
        else:
            variance /= data.size
        return variance

    @staticmethod
    def calc_skewness_coefficient(data):
        average = Selection.calc_average(data)
        s = 0
        for i in data:
            s += (i - average)**3
        return s / (data.size * Selection.calc_standard_deviation(data, 0)**3)
    
    @staticmethod
    def calc_kurtosis_coefficient(data):
        average = Selection.calc_average(data)
        s = 0
        for i in data:
            s += ((i - average)**4)
        return s / (data.size * Selection.calc_standard_deviation(data, 0)**4) - 3
    
    @staticmethod
    def calc_counterkurtosis_coefficient(data):
        return 1 / (np.sqrt(np.abs(Selection.calc_kurtosis_coefficient(data) + 3)))

    @staticmethod
    def calc_standard_deviation_of_average(data, standard_deviation):
        return standard_deviation / np.sqrt(data.size)
    
    @staticmethod
    def calc_standard_deviation_of_standard_deviation(data, standard_deviation):
        return standard_deviation / np.sqrt(data.size * 2)

    @staticmethod
    def calc_standard_deviation_of_skewness_coefficient(data):
        return np.sqrt((6 * (data.size - 2)) / ((data.size + 1) * (data.size + 3)))
    
    @staticmethod
    def calc_standard_deviation_of_kurtosis_coefficient(data):
        return np.sqrt((24 * data.size * (data.size - 2) * (data.size - 3)) / (((data.size + 1)**2) * (data.size + 3) * (data.size + 5)))
    
    @staticmethod
    def calc_standard_deviation_of_counterkurtosis_coefficient(data, kurtosis_coefficient):
        return np.sqrt((np.abs(kurtosis_coefficient)) / (29 * data.size)) * (np.sqrt(np.abs((kurtosis_coefficient**2) - 1)**3))**1/4

    @staticmethod
    def calc_confidence_interval_of_average(average, standard_deviation_of_average, quantile_student):
        return np.array([average - quantile_student * standard_deviation_of_average,
            average + quantile_student * standard_deviation_of_average])     
                        
    @staticmethod
    def calc_confidence_interval_of_median(data, quantile_normal):
        sorted_data = np.sort(data)
        j = round(data.size / 2 - 1 - quantile_normal * (np.sqrt(data.size) / 2))
        k = round(data.size / 2 + quantile_normal * (np.sqrt(data.size) / 2))
        return np.array([sorted_data[j], sorted_data[k]])     
                                                  
    @staticmethod
    def calc_confidence_interval_of_standard_deviation(standard_deviation, standard_deviation_of_standard_deviation, quantile_student):
        return np.array([standard_deviation - quantile_student * standard_deviation_of_standard_deviation,
            standard_deviation + quantile_student * standard_deviation_of_standard_deviation])    
    
    @staticmethod
    def calc_confidence_interval_of_skewness_coefficient(skewness_coefficient, standard_deviation_of_skewness_coefficient, quantile_student):
        return np.array([skewness_coefficient - quantile_student * standard_deviation_of_skewness_coefficient,
            skewness_coefficient + quantile_student * standard_deviation_of_skewness_coefficient])    
    
    @staticmethod
    def calc_confidence_interval_of_kurtosis_coefficient(kurtosis_coefficient, standard_deviation_of_kurtosis_coefficient, quantile_student):
        return np.array([kurtosis_coefficient - quantile_student * standard_deviation_of_kurtosis_coefficient,
            kurtosis_coefficient + quantile_student * standard_deviation_of_kurtosis_coefficient])  
    
    @staticmethod
    def calc_confidence_interval_of_counterkurtosis_coefficient(counterkurtosis_coefficient, standard_deviation_of_counterkurtosis_coefficient, quantile_student):
        return np.array([counterkurtosis_coefficient - quantile_student * standard_deviation_of_counterkurtosis_coefficient,
            counterkurtosis_coefficient + quantile_student * standard_deviation_of_counterkurtosis_coefficient])  