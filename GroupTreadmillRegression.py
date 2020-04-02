import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sb
from scipy.stats import pearsonr as r
from scipy.stats import t
from mpl_toolkits.mplot3d import Axes3D


class Regression:

    def __init__(self, data_file=None):

        self.data = None
        self.pref_data = None
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

        self.ind_rmse = None
        self.ind_rmse_corr_mat = None

        self.import_data(data_file=data_file)
        self.calculate_regression()
        self.ind_rmse, self.ind_rmse_corr_mat = self.calculate_individual_rmse()

    def import_data(self, data_file):

        self.data = pd.read_excel(io=data_file, sheet_name="Test",
                                  names=["Subject", "Age", "Height", "Weight", "BMI", "Counts", "Speed"])

        self.pref_data = self.data.iloc[2::5]

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

        self.equation = str(round(self.counts_coef, 5)) + " x counts/15s + " + \
                        str(round(self.bmi_coef, 15)) + " x BMI (kg/m2) + " + \
                        str(round(self.y_int, 5))

        print("\nGait speed (m/s) = ", self.equation)

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

        # plt.subplot()
        sb.lmplot(x="Counts", y="Speed", data=self.data, height=6.5, aspect=1.5)
        plt.subplots_adjust(top=0.94)
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

    def plot_regression_corr_matrix(self):

        df = pd.DataFrame(self.data, columns=["Height", "Weight", "BMI", "Counts", "Speed"])

        matrix = df.corr()

        sb.heatmap(matrix, cmap="RdYlGn", annot=True)
        plt.title("Correlation Matrix")

    def plot_error_measures(self):

        plt.bar(x=["Mean abs.", "Mean square", "RMSE"], height=[self.mean_abs_error, self.mean_sq_error, self.rmse],
                color='grey', edgecolor='black')
        plt.ylabel("Error (m/s)")
        plt.title("Multiple Linear Regression Error Measures")

    def calculate_individual_rmse(self):

        rmse_list = []
        rmse_perc_pref = []

        for start_ind in range(self.data.shape[0])[::5]:
            counts = [i for i in self.data.iloc[start_ind:start_ind + 5]["Counts"]]

            pred_values = [self.calculate_value(count_value=count, bmi=self.data.iloc[start_ind]["BMI"])
                           for count in counts]

            true_speed = [i for i in self.data.iloc[start_ind:start_ind + 5]["Speed"]]

            rmse = np.sqrt(metrics.mean_squared_error(y_true=true_speed, y_pred=pred_values))
            rmse_list.append(float(rmse))
            rmse_perc_pref.append(float(100 * rmse / self.data.iloc[start_ind + 2]["Speed"]))

        pref_speeds = [self.pref_data.iloc[i]["Speed"] for i in range(self.pref_data.shape[0])]
        heights = [self.pref_data.iloc[i]["Height"] for i in range(self.pref_data.shape[0])]

        r_value = round(r(pref_speeds, rmse_perc_pref)[0], 3)

        rmse_df = pd.DataFrame(list(zip(self.pref_data["Subject"], self.pref_data["Age"],
                                        self.pref_data["Height"], self.pref_data["Weight"],
                                        self.pref_data["BMI"], pref_speeds, rmse_list, rmse_perc_pref)),
                               columns=["Subject", "Age", "Height", "Weight", "BMI",
                                        "Pref Speed", "RMSE", "RMSE (% Pref)"])

        corr_mat = rmse_df[["Pref Speed", "RMSE", "RMSE (% Pref)"]].corr()

        sb.heatmap(corr_mat, cmap="RdYlGn", annot=True)
        plt.title("Individual Participant Correlations")

        return rmse_df, corr_mat


# x = Regression("/Users/kyleweber/Desktop/Data/OND07/Processed Data/TreadmillRegressionGroup.xlsx")
