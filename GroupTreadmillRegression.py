import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sb
from scipy.stats import t
from mpl_toolkits.mplot3d import Axes3D


class Regression:

    def __init__(self, data_file=None):

        self.data = None
        self.measured_values = None
        self.predictor_values = None

        self.predicted_values = None
        self.residuals = None

        self.model = None
        self.counts_coef = 0
        self.bmi_coef = 0
        self.y_int = 0
        self.equation = None

        self.r2 = None
        self.mean_abs_error = None
        self.mean_sq_error = None
        self.rmse = None
        self.confidence_interval = 0

        self.import_data(data_file=data_file)
        self.calculate_regression()

    def import_data(self, data_file):

        self.data = pd.read_excel(io=data_file, sheet_name="Test",
                                  names=["Subject", "Age", "Height", "Weight", "BMI", "Counts", "Speed"])

    def calculate_regression(self):

        self.predictor_values = self.data[["Counts", "BMI"]]
        self.measured_values = self.data["Speed"].values

        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(self.predictor_values, self.measured_values)
        self.r2 = regressor.score(self.predictor_values, self.measured_values)

        coeff_df = pd.DataFrame(regressor.coef_, self.predictor_values.columns, columns=['Coefficient'])

        self.counts_coef = coeff_df["Coefficient"]["Counts"]
        self.bmi_coef = coeff_df["Coefficient"]["BMI"]
        self.y_int = regressor.intercept_

        self.equation = str(round(self.counts_coef, 5)) + " x counts + " + \
                        str(round(self.bmi_coef, 15)) + " x BMI (kg/m/m) + " + \
                        str(round(self.y_int, 5))

        print(self.equation)

        self.predicted_values = regressor.predict(self.predictor_values)
        self.residuals = [measured - predicted for measured, predicted in
                          zip(self.measured_values, self.predicted_values)]

        self.mean_abs_error = metrics.mean_absolute_error(y_true=self.measured_values, y_pred=self.predicted_values)
        self.mean_sq_error = metrics.mean_squared_error(y_true=self.measured_values, y_pred=self.predicted_values)
        self.rmse = np.sqrt(metrics.mean_squared_error(y_true=self.measured_values, y_pred=self.predicted_values))

        # Calculates 95% confidence interval
        df = self.data.shape[0] - 3  # df = n_datapoints - n_terms
        t_crit = t.ppf(0.95, df)  # finds critical t-value

        self.confidence_interval = self.rmse * t_crit

    def calculate_value(self, count_value, bmi):

        return round(self.counts_coef * count_value + self.bmi_coef * bmi + self.y_int, 3)

    def plot_regression(self):

        sb.lmplot(x="Counts", y="Speed", data=self.data, height=6.5, aspect=1.5)
        plt.scatter(self.data["Counts"], self.predicted_values, color='red')
        plt.legend(labels=["Regression line", "Measured", "95%CI", "Predicted"], loc='upper left')
        plt.ylabel("Speed (m/s)")
        plt.title("MLR (counts + BMI ~ speed): r2 = {}".format(round(self.r2, 3)))

    def plot_3d(self):

        fig = plt.figure(figsize=(9, 7))
        ax = Axes3D(fig)

        ax.scatter(self.data["Counts"], self.data["BMI"], self.data['Speed'], s=15, c='red', marker="o")

        ax.set_xlabel("Counts")
        ax.set_ylabel("Height (m)")
        ax.set_zlabel("Speed (m/s)")
        ax.set_title("Group Level Multiple Linear Regression")

    def plot_corr_matrix(self):

        df = pd.DataFrame(self.data, columns=["Height", "Weight", "BMI", "Counts", "Speed"])

        matrix = df.corr()

        sb.heatmap(matrix, cmap="RdYlGn", annot=True)
        plt.title("Correlation Matrix")

    def plot_error_measures(self):

        plt.bar(x=["Mean abs.", "Mean square", "RMSE"], height=[self.mean_abs_error, self.mean_sq_error, self.rmse],
                color='grey', edgecolor='black')
        plt.ylabel("Error (m/s)")
        plt.title("Multiple Linear Regression Error Measures")


x = Regression("/Users/kyleweber/Desktop/Data/OND07/Processed Data/TreadmillRegressionGroup.xlsx")
