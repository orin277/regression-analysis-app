from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableWidgetItem, QHeaderView
import sys
import main_ui
import numpy as np
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvas
from univariate_regression import UnivariateRegression
from univariate_nonlinear_regression import UnivariateNonlinearRegression
from correlation import Correlation
from selection import Selection
from identification_normal_distribution import IdentificationNormalDistribution
from multivariate_regression import MultivariateRegression
from f_test import FTest
import csv


class RegressionAnalysisApp(QtWidgets.QMainWindow, main_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(RegressionAnalysisApp, self).__init__(parent)
        self.setupUi(self)
        self.action.triggered.connect(self.load_onedimensional_data)
        self.action_3.triggered.connect(self.load_multidimensional_data)
        self.btnRestoreLinearRegression.clicked.connect(self.restore_linear_regression)
        self.btnRestoreNonlinearRegression.clicked.connect(self.restore_nonlinear_regression)
        self.rbMultivariateRegression.toggled.connect(self.hide_independent_indicator)
        self.rbUnivariateRegression.toggled.connect(self.show_independent_indicator)

        self.rbUnivariateRegression.setChecked(True)
        self.btnRestoreLinearRegression.setEnabled(False)

        self.data = np.empty((0))

    def hide_independent_indicator(self):
        self.cmbIndependentIndicator.hide()
        self.labelIndependentIndicator.hide()

    def show_independent_indicator(self):
        self.cmbIndependentIndicator.show()
        self.labelIndependentIndicator.show()

    def load_onedimensional_data(self):
        res = QFileDialog.getOpenFileName(self, "Open File", "", "Text files (*.txt)")
        if (res == ("", "")): return;

        self.data= {}
        self.data["1"] = list()
        self.data["2"] = list()

        with open(res[0]) as file:
            for row in file:
                values = row.strip().split()
                self.data["1"].append(values[0])
                self.data["2"].append(values[1])

        self.cmbDependentIndicator.clear()
        self.cmbIndependentIndicator.clear()
        self.cmbDependentIndicator_2.clear()
        self.cmbIndependentIndicator_2.clear()
        self.cmbDependentIndicator.addItems([i for i in self.data.keys()])
        self.cmbIndependentIndicator.addItems([i for i in self.data.keys()])
        self.cmbDependentIndicator_2.addItems([i for i in self.data.keys()])
        self.cmbIndependentIndicator_2.addItems([i for i in self.data.keys()])

        self.fill_in_selection_table()

        self.btnRestoreLinearRegression.setEnabled(True)

    def load_multidimensional_data(self):
        res = QFileDialog.getOpenFileName(self, "Open File", "", "Text files (*.csv)")
        if (res == ("", "")): return;

        with open(res[0]) as csvfile:
            reader = csv.reader(csvfile)

            headers = next(reader)
            self.data = {header: [] for header in headers}

            for row in reader:
                row = [float(val) if val.replace('.', '', 1).isdigit() else val for val in row]

                for i, val in enumerate(row):
                    self.data [headers[i]].append(val)

        # delete string indicators
        for key, value in list(self.data.items()):
            if all(isinstance(x, str) and not x.isdigit() for x in value):
                del self.data[key]

        self.cmbDependentIndicator.clear()
        self.cmbIndependentIndicator.clear()
        self.cmbDependentIndicator_2.clear()
        self.cmbIndependentIndicator_2.clear()
        self.cmbDependentIndicator.addItems([i for i in self.data.keys()])
        self.cmbIndependentIndicator.addItems([i for i in self.data.keys()])
        self.cmbDependentIndicator_2.addItems([i for i in self.data.keys()])
        self.cmbIndependentIndicator_2.addItems([i for i in self.data.keys()])

        self.fill_in_selection_table()

        self.btnRestoreLinearRegression.setEnabled(True)

    def restore_linear_regression(self):
        if self.rbUnivariateRegression.isChecked():
            column1 = self.cmbIndependentIndicator.currentText()
            column2 = self.cmbDependentIndicator.currentText()

            self.univariate_regression = UnivariateRegression(np.array(self.data[column1], dtype=np.float32), 
                np.array(self.data[column2], dtype=np.float32))
            self.ftest = FTest(self.univariate_regression.determination_coefficient, 
                2, len(self.data[column2]))

            self.identif_norm_for_residuals = IdentificationNormalDistribution(self.univariate_regression.residuals)

            self.correlation = Correlation(np.array(self.data[column1], dtype=np.float32), 
                np.array(self.data[column2], dtype=np.float32))

            self.update_app_for_univariate_linear_regression()
        else:
            column = self.cmbDependentIndicator.currentText()
            new_data = {key: value for key, value in self.data.items() if key != column}

            data_list = []
            for key in new_data:
                data_list.append(new_data[key])

            self.multivariate_regression = MultivariateRegression(np.array(data_list, dtype=np.float32).T, 
                np.array(self.data[column], dtype=np.float32))
            
            self.ftest = FTest(self.multivariate_regression.determination_coefficient, 
                self.multivariate_regression.params.size, len(self.data[column]))

            self.identif_norm_for_residuals = IdentificationNormalDistribution(self.multivariate_regression.residuals)

            self.update_app_for_multivariate_linear_regression()

    def restore_nonlinear_regression(self):
        column1 = self.cmbIndependentIndicator_2.currentText()
        column2 = self.cmbDependentIndicator_2.currentText()

        linear_regression = UnivariateRegression(np.array(self.data[column1], 
            dtype=np.float32), np.array(self.data[column1], dtype=np.float32) / np.array(self.data[column2], dtype=np.float32))

        self.univariate_nonlinear_regression = UnivariateNonlinearRegression(np.array(self.data[column1], 
            dtype=np.float32), np.array(self.data[column2], dtype=np.float32), linear_regression.a["value"], 
            linear_regression.b["value"], linear_regression.x / linear_regression.restored_regression)

        self.ftest = FTest(self.univariate_nonlinear_regression.determination_coefficient, 
            2, len(self.data[column2]))

        self.identif_norm_for_residuals = IdentificationNormalDistribution(self.univariate_nonlinear_regression.residuals)

        self.correlation = Correlation(np.array(self.data[column1], dtype=np.float32), 
            np.array(self.data[column2], dtype=np.float32))

        self.update_app_for_univariate_nonlinear_regression()

    def update_app_for_univariate_linear_regression(self):
        self.fill_in_univariate_linear_regression_table()
        self.update_regression_line_chart()
        self.labelResidualVariance.setText("Залишкова дисперсія: " 
            + str(round(self.univariate_regression.residuals_variance, 8)))
        self.labelDeterminationCoeff.setText("Коефіцієнт детермінації: " 
            + str(round(self.univariate_regression.determination_coefficient, 4)))
        self.fill_in_residuals_norm_table()
        self.fill_in_data_for_ftest_linear_regression()
        
        self.fill_in_table_of_static_characteristics()
        self.fill_in_table_of_static_characteristics2()
        self.fill_in_table_correlation_coefficients()

    def update_app_for_univariate_nonlinear_regression(self):
        self.fill_in_univariate_nonlinear_regression_table()
        self.update_nonlinear_regression_line_chart()
        self.labelResidualVariance_2.setText("Залишкова дисперсія: " 
            + str(round(self.univariate_nonlinear_regression.residuals_variance, 8)))
        self.labelDeterminationCoeff_2.setText("Коефіцієнт детермінації: " 
            + str(round(self.univariate_nonlinear_regression.determination_coefficient, 4)))
        self.fill_in_residuals_norm_table()
        self.fill_in_data_for_ftest_nonlinear_regression()
        
        self.fill_in_table_of_static_characteristics()
        self.fill_in_table_of_static_characteristics2()
        self.fill_in_table_correlation_coefficients()

    def update_app_for_multivariate_linear_regression(self):
        self.fill_in_multivariate_linear_regression_table()
        self.update_residuals_chart()
        self.labelResidualVariance.setText("Залишкова дисперсія: " 
            + str(round(self.multivariate_regression.residuals_variance, 8)))
        self.labelDeterminationCoeff.setText("Коефіцієнт детермінації: " 
            + str(round(self.multivariate_regression.determination_coefficient, 4)))
        self.fill_in_residuals_norm_table()
        self.fill_in_data_for_ftest_linear_regression()

    def fill_in_selection_table(self):
        self.tableSelection.setRowCount(len(next(iter(self.data.values()))))
        self.tableSelection.setColumnCount(len(self.data))

        self.tableSelection.setHorizontalHeaderLabels(list(self.data.keys()))

        for i, key in enumerate(self.data.keys()):
            for j, value in enumerate(self.data[key]):
                item = QTableWidgetItem(str(value))
                self.tableSelection.setItem(j, i, item)

    def fill_in_data_for_ftest_linear_regression(self):
        self.labelFtestStats.setText("Статистика: " + str(round(self.ftest.f, 4)))
        self.labelFtestQuantile.setText("Квантиль: " + str(round(self.ftest.quantile_fisher, 4)))
        result = "регресія незначуща" if self.ftest.check_insignificance() else "регресія значуща";
        self.labelFtestResult.setText("Висновок: " + result)

    def fill_in_data_for_ftest_nonlinear_regression(self):
        self.labelFtestStats_2.setText("Статистика: " + str(round(self.ftest.f, 4)))
        self.labelFtestQuantile_2.setText("Квантиль: " + str(round(self.ftest.quantile_fisher, 4)))
        result = "регресія незначуща" if self.ftest.check_insignificance() else "регресія значуща";
        self.labelFtestResult_2.setText("Висновок: " + result)

    def fill_in_residuals_norm_table(self):
        self.tableResidualsNorm.setColumnCount(2)
        self.tableResidualsNorm.setHorizontalHeaderLabels(["", "Результат"])
        self.tableResidualsNorm.setRowCount(4)

        header = self.tableResidualsNorm.horizontalHeader()       
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        self.tableResidualsNorm.setItem(0, 0, QTableWidgetItem("Статистика асиметрії"))
        self.tableResidualsNorm.setItem(0, 1, QTableWidgetItem(str(round(self.identif_norm_for_residuals.skewness_statistics, 4))))

        self.tableResidualsNorm.setItem(1, 0, QTableWidgetItem("Статистика ексцесу"))
        self.tableResidualsNorm.setItem(1, 1, QTableWidgetItem(str(round(self.identif_norm_for_residuals.kurtosis_statistics, 4))))

        self.tableResidualsNorm.setItem(2, 0, QTableWidgetItem("Квантиль норм. розподілу"))
        self.tableResidualsNorm.setItem(2, 1, QTableWidgetItem(str(round(self.identif_norm_for_residuals.quantile_normal, 4))))

        if (self.identif_norm_for_residuals.identify_distribution()):
            result = "Нормальний розподіл ідентифіковано"
        else:
            result = "Нормальний розподіл не ідентифіковано"

        self.tableResidualsNorm.setItem(3, 0, QTableWidgetItem("Висновок"))
        self.tableResidualsNorm.setItem(3, 1, QTableWidgetItem(result))

    def fill_in_univariate_linear_regression_table(self):
        self.tableRegression.setRowCount(2)

        self.tableRegression.setItem(0, 0, QTableWidgetItem("a1"))
        self.tableRegression.setItem(0, 1, QTableWidgetItem(str(round(self.univariate_regression.a["value"], 4))))
        self.tableRegression.setItem(0, 2, QTableWidgetItem(str(round(self.univariate_regression.a["std"], 4))))
        self.tableRegression.setItem(0, 3, QTableWidgetItem("[" + str(round(self.univariate_regression.a["confidence_interval"][0], 4)) + 
            "; " + str(round(self.univariate_regression.a["confidence_interval"][1], 4)) + "]"))
        self.tableRegression.setItem(0, 4, QTableWidgetItem(str(round(self.univariate_regression.a["stats"], 4))))
        self.tableRegression.setItem(0, 5, QTableWidgetItem(str(round(self.univariate_regression.quantile_student, 4))))
        result = "= 0" if self.univariate_regression.check_insignificance_a() else "!= 0";
        self.tableRegression.setItem(0, 6, QTableWidgetItem(result))

        self.tableRegression.setItem(1, 0, QTableWidgetItem("a2"))
        self.tableRegression.setItem(1, 1, QTableWidgetItem(str(round(self.univariate_regression.b["value"], 4))))
        self.tableRegression.setItem(1, 2, QTableWidgetItem(str(round(self.univariate_regression.b["std"], 4))))
        self.tableRegression.setItem(1, 3, QTableWidgetItem("[" + str(round(self.univariate_regression.b["confidence_interval"][0], 4)) + 
            "; " + str(round(self.univariate_regression.b["confidence_interval"][1], 4)) + "]"))
        self.tableRegression.setItem(1, 4, QTableWidgetItem(str(round(self.univariate_regression.b["stats"], 4))))
        self.tableRegression.setItem(1, 5, QTableWidgetItem(str(round(self.univariate_regression.quantile_student, 4))))
        result = "= 0" if self.univariate_regression.check_insignificance_a() else "!= 0";
        self.tableRegression.setItem(1, 6, QTableWidgetItem(result)) 

    def fill_in_univariate_nonlinear_regression_table(self):
        self.tableNonlinearRegression.setRowCount(2)

        self.tableNonlinearRegression.setItem(0, 0, QTableWidgetItem("a1"))
        self.tableNonlinearRegression.setItem(0, 1, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.a["value"], 4))))
        self.tableNonlinearRegression.setItem(0, 2, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.a["std"], 4))))
        self.tableNonlinearRegression.setItem(0, 3, QTableWidgetItem("[" + str(round(self.univariate_nonlinear_regression.a["confidence_interval"][0], 4)) + 
            "; " + str(round(self.univariate_nonlinear_regression.a["confidence_interval"][1], 4)) + "]"))
        self.tableNonlinearRegression.setItem(0, 4, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.a["stats"], 4))))
        self.tableNonlinearRegression.setItem(0, 5, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.quantile_student, 4))))
        result = "= 0" if self.univariate_nonlinear_regression.check_insignificance_a() else "!= 0";
        self.tableNonlinearRegression.setItem(0, 6, QTableWidgetItem(result))

        self.tableNonlinearRegression.setItem(1, 0, QTableWidgetItem("a2"))
        self.tableNonlinearRegression.setItem(1, 1, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.b["value"], 4))))
        self.tableNonlinearRegression.setItem(1, 2, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.b["std"], 4))))
        self.tableNonlinearRegression.setItem(1, 3, QTableWidgetItem("[" + str(round(self.univariate_nonlinear_regression.b["confidence_interval"][0], 4)) + 
            "; " + str(round(self.univariate_nonlinear_regression.b["confidence_interval"][1], 4)) + "]"))
        self.tableNonlinearRegression.setItem(1, 4, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.b["stats"], 4))))
        self.tableNonlinearRegression.setItem(1, 5, QTableWidgetItem(str(round(self.univariate_nonlinear_regression.quantile_student, 4))))
        result = "= 0" if self.univariate_nonlinear_regression.check_insignificance_a() else "!= 0";
        self.tableNonlinearRegression.setItem(1, 6, QTableWidgetItem(result)) 

    def fill_in_multivariate_linear_regression_table(self):
        self.tableRegression.setRowCount(self.multivariate_regression.params.size)

        param_significance = self.multivariate_regression.check_insignificance() 

        for i in range(self.multivariate_regression.params.size):
            self.tableRegression.setItem(i, 0, QTableWidgetItem("a" + str(i)))
            self.tableRegression.setItem(i, 1, QTableWidgetItem(str(round(self.multivariate_regression.params[i], 4))))
            self.tableRegression.setItem(i, 2, QTableWidgetItem(str(round(self.multivariate_regression.param_std[i], 4))))
            self.tableRegression.setItem(i, 3, QTableWidgetItem("[" 
                + str(round(self.multivariate_regression.param_confidence_intervals[i][0], 4)) 
                + "; " + str(round(self.multivariate_regression.param_confidence_intervals[i][1], 4)) + "]"))
            self.tableRegression.setItem(i, 4, QTableWidgetItem(str(round(self.multivariate_regression.param_stats[i], 4))))
            self.tableRegression.setItem(i, 5, QTableWidgetItem(str(round(self.multivariate_regression.quantile_student, 4))))
            result = "= 0" if param_significance[i] else "!= 0";
            self.tableRegression.setItem(i, 6, QTableWidgetItem(result))

    def update_regression_line_chart(self):
        figure = self.univariate_regression.draw_regression_line()
        figure.suptitle("Кореляційне поле")

        if (self.widgetChart_3.layout()):
            self.widgetChart_3.layout().replaceWidget(self.widgetChart_3.layout().itemAt(0).widget(), 
                FigureCanvas(figure))
        else:
            layout = QtWidgets.QVBoxLayout(self.widgetChart_3)
            layout.addWidget(FigureCanvas(figure))

    def update_nonlinear_regression_line_chart(self):
        figure = self.univariate_nonlinear_regression.draw_regression_line()
        figure.suptitle("Кореляційне поле")

        if (self.widgetChart_4.layout()):
            self.widgetChart_4.layout().replaceWidget(self.widgetChart_4.layout().itemAt(0).widget(), 
                FigureCanvas(figure))
        else:
            layout = QtWidgets.QVBoxLayout(self.widgetChart_4)
            layout.addWidget(FigureCanvas(figure))

    def update_residuals_chart(self):
        figure = self.multivariate_regression.draw_residual_diagram()
        figure.suptitle("Діаграма залишків")

        if (self.widgetChart_3.layout()):
            self.widgetChart_3.layout().replaceWidget(self.widgetChart_3.layout().itemAt(0).widget(), 
                FigureCanvas(figure))
        else:
            layout = QtWidgets.QVBoxLayout(self.widgetChart_3)
            layout.addWidget(FigureCanvas(figure))
        
    def fill_in_table_correlation_coefficients(self):
        self.tableWidget_8.setRowCount(4)
        self.tableWidget_8.setItem(0, 0, QTableWidgetItem("Пірсона"))
        self.tableWidget_8.setItem(0, 1, QTableWidgetItem(str(self.correlation.pearson_coefficient.value)))
        self.tableWidget_8.setItem(0, 2, QTableWidgetItem("[" + str(round(self.correlation.pearson_coefficient.confidence_interval[0], 4)) + "; " +
                str(round(self.correlation.pearson_coefficient.confidence_interval[1], 4)) + "]"))
        self.tableWidget_8.setItem(0, 3, QTableWidgetItem(str(self.correlation.pearson_coefficient.stats)))
        self.tableWidget_8.setItem(0, 4, QTableWidgetItem(str(self.correlation.pearson_coefficient.quantile_student)))

        if (self.correlation.pearson_coefficient.determine_presence_of_connection() == True):
            self.tableWidget_8.setItem(0, 5, QTableWidgetItem("Значущий"))
            self.tableWidget_8.setItem(0, 6, QTableWidgetItem("Є лінійний зв'зок"))
        else:
            self.tableWidget_8.setItem(0, 5, QTableWidgetItem("Незначущий"))
            self.tableWidget_8.setItem(0, 6, QTableWidgetItem("Немає лінійного зв'язку"))

        self.tableWidget_8.setItem(1, 0, QTableWidgetItem("Спірмена"))
        self.tableWidget_8.setItem(1, 1, QTableWidgetItem(str(self.correlation.spearman_coefficient.value)))
        self.tableWidget_8.setItem(1, 3, QTableWidgetItem(str(self.correlation.spearman_coefficient.stats)))
        self.tableWidget_8.setItem(1, 4, QTableWidgetItem(str(self.correlation.spearman_coefficient.quantile_student)))

        if (self.correlation.spearman_coefficient.determine_presence_of_connection() == True):
            self.tableWidget_8.setItem(1, 5, QTableWidgetItem("Значущий"))
            self.tableWidget_8.setItem(1, 6, QTableWidgetItem("Є монотонний зв'зок"))
        else:
            self.tableWidget_8.setItem(1, 5, QTableWidgetItem("Незначущий"))
            self.tableWidget_8.setItem(1, 6, QTableWidgetItem("Немає монотонного зв'язку"))

        self.tableWidget_8.setItem(2, 0, QTableWidgetItem("Кендалла"))
        self.tableWidget_8.setItem(2, 1, QTableWidgetItem(str(self.correlation.kendall_coefficient.value)))
        self.tableWidget_8.setItem(2, 3, QTableWidgetItem(str(self.correlation.kendall_coefficient.stats)))
        self.tableWidget_8.setItem(2, 4, QTableWidgetItem(str(self.correlation.kendall_coefficient.quantile_normal)))

        if (self.correlation.kendall_coefficient.determine_presence_of_connection() == True):
            self.tableWidget_8.setItem(2, 5, QTableWidgetItem("Значущий"))
            self.tableWidget_8.setItem(2, 6, QTableWidgetItem("Є монотонний зв'зок"))
        else:
            self.tableWidget_8.setItem(2, 5, QTableWidgetItem("Незначущий"))
            self.tableWidget_8.setItem(2, 6, QTableWidgetItem("Немає монотонного зв'язку"))

        self.tableWidget_8.setItem(3, 0, QTableWidgetItem("Кореляційне відношення"))
        self.tableWidget_8.setItem(3, 1, QTableWidgetItem(str(self.correlation.correlation_relation.value)))
        self.tableWidget_8.setItem(3, 3, QTableWidgetItem(str(self.correlation.correlation_relation.stats)))
        self.tableWidget_8.setItem(3, 4, QTableWidgetItem(str(self.correlation.correlation_relation.quantile_fisher)))

        self.tableWidget_9.setRowCount(0)
        if (self.correlation.correlation_relation.determine_presence_of_connection() == True):
            self.tableWidget_8.setItem(3, 5, QTableWidgetItem("Значущий"))
            self.tableWidget_8.setItem(3, 6, QTableWidgetItem("Є стохастичний зв'зок"))

            self.correlation.correlation_relation.test_for_equality_of_pearson_coefficient()

            self.tableWidget_9.setRowCount(1)
            self.tableWidget_9.setItem(0, 0, QTableWidgetItem(str(self.correlation.correlation_relation.pearson_coefficient.value)))
            self.tableWidget_9.setItem(0, 1, QTableWidgetItem(str(self.correlation.correlation_relation.value)))
            self.tableWidget_9.setItem(0, 2, QTableWidgetItem(str(self.correlation.correlation_relation.stats2)))
            self.tableWidget_9.setItem(0, 3, QTableWidgetItem(str(self.correlation.correlation_relation.quantile_fisher)))

            if (self.correlation.correlation_relation.determine_linear_relationship() == True):
                self.tableWidget_9.setItem(0, 4, QTableWidgetItem("Значущий"))
                self.tableWidget_9.setItem(0, 5, QTableWidgetItem("Є лінійний зв'зок"))
            else:
                self.tableWidget_9.setItem(0, 4, QTableWidgetItem("Незначущий"))
                self.tableWidget_9.setItem(0, 5, QTableWidgetItem("Немає лінійного зв'язку"))
        else:
            self.tableWidget_8.setItem(3, 5, QTableWidgetItem("Незначущий"))
            self.tableWidget_8.setItem(3, 6, QTableWidgetItem("Немає стохастичного зв'язку"))

    def fill_in_table_of_static_characteristics(self):
        self.tableWidget_11.setRowCount(5)
        data1_stats = Selection(self.correlation.data1)

        self.tableWidget_11.setItem(0, 0, QTableWidgetItem("Середнє арифметичне"))
        self.tableWidget_11.setItem(0, 1, QTableWidgetItem(str(round(data1_stats.average["value"], 4))))
        self.tableWidget_11.setItem(0, 2, QTableWidgetItem("[" + str(round(data1_stats.average["confidence_interval"][0], 4)) + "; " +
                str(round(data1_stats.average["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_11.setItem(1, 0, QTableWidgetItem("Медіана"))
        self.tableWidget_11.setItem(1, 1, QTableWidgetItem(str(round(data1_stats.median["value"], 4))))
        self.tableWidget_11.setItem(1, 2, QTableWidgetItem("[" + str(round(data1_stats.median["confidence_interval"][0], 4)) + "; " +
                str(round(data1_stats.median["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_11.setItem(2, 0, QTableWidgetItem("Середньоквадратичне"))
        self.tableWidget_11.setItem(2, 1, QTableWidgetItem(str(round(data1_stats.standard_deviation["value"], 4))))
        self.tableWidget_11.setItem(2, 2, QTableWidgetItem("[" + str(round(data1_stats.standard_deviation["confidence_interval"][0], 4)) + "; " +
                str(round(data1_stats.standard_deviation["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_11.setItem(3, 0, QTableWidgetItem("Коефіцієнт асиметрії"))
        self.tableWidget_11.setItem(3, 1, QTableWidgetItem(str(round(data1_stats.skewness_coefficient["value"], 4))))
        self.tableWidget_11.setItem(3, 2, QTableWidgetItem("[" + str(round(data1_stats.skewness_coefficient["confidence_interval"][0], 4)) + "; " +
                str(round(data1_stats.skewness_coefficient["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_11.setItem(4, 0, QTableWidgetItem("Коефіцієнт ексцесу"))
        self.tableWidget_11.setItem(4, 1, QTableWidgetItem(str(round(data1_stats.kurtosis_coefficient["value"], 4))))
        self.tableWidget_11.setItem(4, 2, QTableWidgetItem("[" + str(round(data1_stats.kurtosis_coefficient["confidence_interval"][0], 4)) + "; " +
                str(round(data1_stats.kurtosis_coefficient["confidence_interval"][1], 4)) + "]"))

    def fill_in_table_of_static_characteristics2(self):
        self.tableWidget_12.setRowCount(5)
        data2_stats = Selection(self.correlation.data2)

        self.tableWidget_12.setItem(0, 0, QTableWidgetItem("Середнє арифметичне"))
        self.tableWidget_12.setItem(0, 1, QTableWidgetItem(str(round(data2_stats.average["value"], 4))))
        self.tableWidget_12.setItem(0, 2, QTableWidgetItem("[" + str(round(data2_stats.average["confidence_interval"][0], 4)) + "; " +
                str(round(data2_stats.average["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_12.setItem(1, 0, QTableWidgetItem("Медіана"))
        self.tableWidget_12.setItem(1, 1, QTableWidgetItem(str(round(data2_stats.median["value"], 4))))
        self.tableWidget_12.setItem(1, 2, QTableWidgetItem("[" + str(round(data2_stats.median["confidence_interval"][0], 4)) + "; " +
                str(round(data2_stats.median["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_12.setItem(2, 0, QTableWidgetItem("Середньоквадратичне"))
        self.tableWidget_12.setItem(2, 1, QTableWidgetItem(str(round(data2_stats.standard_deviation["value"], 4))))
        self.tableWidget_12.setItem(2, 2, QTableWidgetItem("[" + str(round(data2_stats.standard_deviation["confidence_interval"][0], 4)) + "; " +
                str(round(data2_stats.standard_deviation["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_12.setItem(3, 0, QTableWidgetItem("Коефіцієнт асиметрії"))
        self.tableWidget_12.setItem(3, 1, QTableWidgetItem(str(round(data2_stats.skewness_coefficient["value"], 4))))
        self.tableWidget_12.setItem(3, 2, QTableWidgetItem("[" + str(round(data2_stats.skewness_coefficient["confidence_interval"][0], 4)) + "; " +
                str(round(data2_stats.skewness_coefficient["confidence_interval"][1], 4)) + "]"))

        self.tableWidget_12.setItem(4, 0, QTableWidgetItem("Коефіцієнт ексцесу"))
        self.tableWidget_12.setItem(4, 1, QTableWidgetItem(str(round(data2_stats.kurtosis_coefficient["value"], 4))))
        self.tableWidget_12.setItem(4, 2, QTableWidgetItem("[" + str(round(data2_stats.kurtosis_coefficient["confidence_interval"][0], 4)) + "; " +
                str(round(data2_stats.kurtosis_coefficient["confidence_interval"][1], 4)) + "]"))


    def fill_in_table_of_identification_of_normal_distribution(self):
        self.tableWidget_10.setRowCount(4)

        header = self.tableWidget_10.horizontalHeader()       
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        self.tableWidget_10.setItem(0, 0, QTableWidgetItem("Статистика асиметрії"))
        self.tableWidget_10.setItem(0, 1, QTableWidgetItem(str(round(self.identificationNormalDistribution1.skewness_statistics, 4))))
        self.tableWidget_10.setItem(0, 2, QTableWidgetItem(str(round(self.identificationNormalDistribution2.skewness_statistics, 4))))

        self.tableWidget_10.setItem(1, 0, QTableWidgetItem("Статистика ексцесу"))
        self.tableWidget_10.setItem(1, 1, QTableWidgetItem(str(round(self.identificationNormalDistribution1.kurtosis_statistics, 4))))
        self.tableWidget_10.setItem(1, 2, QTableWidgetItem(str(round(self.identificationNormalDistribution2.kurtosis_statistics, 4))))

        self.tableWidget_10.setItem(2, 0, QTableWidgetItem("Квантиль норм. розподілу"))
        self.tableWidget_10.setItem(2, 1, QTableWidgetItem(str(round(self.identificationNormalDistribution1.quantile_normal, 4))))
        self.tableWidget_10.setItem(2, 2, QTableWidgetItem(str(round(self.identificationNormalDistribution2.quantile_normal, 4))))

        result1 = ""
        if (self.identificationNormalDistribution1.identify_distribution()):
            result1 = "Нормальний розподіл ідентифіковано"
        else:
            result1 = "Нормальний розподіл не ідентифіковано"

        result2 = ""
        if (self.identificationNormalDistribution2.identify_distribution()):
            result2 = "Нормальний розподіл ідентифіковано"
        else:
            result2 = "Нормальний розподіл не ідентифіковано"

        self.tableWidget_10.setItem(3, 0, QTableWidgetItem("Висновок"))
        self.tableWidget_10.setItem(3, 1, QTableWidgetItem(result1))
        self.tableWidget_10.setItem(3, 2, QTableWidgetItem(result2))

def main():
    app = QApplication(sys.argv)
    form = RegressionAnalysisApp()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()