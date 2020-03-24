import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sb


class Regression:

    def __init__(self, data_file=None):

        self.data = None
        self.measured_values = None
        self.predictor_values = None

        self.predicted_values = None
        self.residuals = None

        self.model = None
        self.height_coef = None
        self.counts_coef = None
        self.weight_coef = None
        self.y_int = None
        self.equation = None

        self.r2 = None
        self.mean_abs_error = None
        self.mean_sq_error = None
        self.rmse = None

        self.import_data(data_file=data_file)
        self.calculate_regression()
        self.plot_regression()

    def import_data(self, data_file):

        self.data = pd.read_excel(io=data_file, sheet_name="Test",
                                  names=["Subject", "Height", "Weight", "BMI", "Counts", "Speed"])

    def calculate_regression(self):

        self.predictor_values = self.data[["Counts", "BMI"]]
        self.measured_values = self.data["Speed"].values

        regressor = LinearRegression()
        regressor.fit(self.predictor_values, self.measured_values)
        self.r2 = regressor.score(self.predictor_values, self.measured_values)

        coeff_df = pd.DataFrame(regressor.coef_, self.predictor_values.columns, columns=['Coefficient'])

        """self.height_coef = coeff_df["Coefficient"]["Height"]
        self.counts_coef = coeff_df["Coefficient"]["Counts"]
        self.weight_coef = coeff_df["Coefficient"]["Weight"]
        self.y_int = regressor.intercept_

        self.equation = str(round(self.counts_coef, 5)) + " x counts + " + \
                        str(round(self.height_coef, 5)) + " x height (m) + " + \
                        str(round(self.weight_coef, 5)) + " x weight (kg) + " + \
                        str(round(self.y_int, 5))

        print(self.equation)"""

        self.predicted_values = regressor.predict(self.predictor_values)
        self.residuals = [measured - predicted for measured, predicted in
                          zip(self.measured_values, self.predicted_values)]

        self.mean_abs_error = metrics.mean_absolute_error(y_true=self.measured_values, y_pred=self.predicted_values)
        self.mean_sq_error = metrics.mean_squared_error(y_true=self.measured_values, y_pred=self.predicted_values)
        self.rmse = np.sqrt(metrics.mean_squared_error(y_true=self.measured_values, y_pred=self.predicted_values))

    def plot_regression(self):

        sb.lmplot(x="Counts", y="Speed", data=x.data)
        plt.scatter(x.data["Counts"], x.predicted_values, color='red')
        plt.legend(labels=["Regression line", "Measured", "95%CI", "Predicted"], loc='upper left')
        plt.ylabel("Speed (m/s)")
        plt.title("MLR (counts + BMI ~ speed): r2 = {}".format(round(self.r2, 3)))

    def plot_error_measures(self):

        plt.bar(x=["Mean abs.", "Mean square", "RMSE"], height=[self.mean_abs_error, self.mean_sq_error, self.rmse],
                color='grey', edgecolor='black')
        plt.ylabel("Error (m/s)")
        plt.title("Multiple Linear Regression Error Measures")


x = Regression("/Users/kyleweber/Desktop/Data/OND07/Processed Data/TreadmillRegressionGroup.xlsx")
