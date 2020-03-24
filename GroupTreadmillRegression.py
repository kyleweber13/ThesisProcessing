import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class Regression():

    def __init__(self, data_file=None):

        self.data = None
        self.measured_values = None
        self.predictor_values = None

        self.predicted_values = None

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

        self.predictor_values = self.data[["Counts", "Height", "Weight"]]
        self.measured_values = self.data["Speed"].values

        regressor = LinearRegression()
        regressor.fit(self.predictor_values, self.measured_values)
        self.r2 = regressor.score(self.predictor_values, self.measured_values)

        coeff_df = pd.DataFrame(regressor.coef_, self.predictor_values.columns, columns=['Coefficient'])

        self.height_coef = coeff_df["Coefficient"]["Height"]
        self.counts_coef = coeff_df["Coefficient"]["Counts"]
        self.weight_coef = coeff_df["Coefficient"]["Weight"]
        self.y_int = regressor.intercept_

        self.equation = str(self.counts_coef) + " x counts + " + \
                        str(self.height_coef) + " x height (cm) + " + \
                        str(self.weight_coef) + " x weight (kg) + " + \
                        str(self.y_int)

        self.predicted_values = regressor.predict(self.predictor_values)

        self.mean_abs_error = metrics.mean_absolute_error(y_true=self.measured_values, y_pred=self.predicted_values)
        self.mean_sq_error = metrics.mean_squared_error(y_true=self.measured_values, y_pred=self.predicted_values)
        self.rmse = np.sqrt(metrics.mean_squared_error(y_true=self.measured_values, y_pred=self.predicted_values))

    def plot_regression(self):

        plt.scatter(self.data["Counts"], self.data["Speed"], color='black', s=8, marker="X", label="Measured")
        plt.scatter(self.data["Counts"], self.predicted_values, color='red', s=8, marker="o", label="Predicted")

        plt.legend(loc='upper left')
        plt.xlabel("Counts")
        plt.ylabel("Speed (m/s)")

        plt.title("MLR (counts + height + weight ~ speed): r2 = {}".format(round(self.r2, 3)))


# x = Regression("/Users/kyleweber/Desktop/Data/OND07/Processed Data/TreadmillRegressionGroup.xlsx")
