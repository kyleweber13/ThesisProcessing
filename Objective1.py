import LocateUsableParticipants
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import os
import researchpy as rp
import pingouin as pg
import numpy as np
import statsmodels.stats.api as sms

usable_subjs = LocateUsableParticipants.SubjectSubset(check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/"
                                                                 "OND07_ProcessingStatus.xlsx",
                                                      wrist_ankle=False, wrist_hr=False,
                                                      wrist_hracc=False, hr_hracc=False,
                                                      ankle_hr=False, ankle_hracc=False,
                                                      wrist_only=False, ankle_only=False,
                                                      hr_only=False, hracc_only=False,
                                                      require_treadmill=True, require_all=True)


class Objective1:

    def __init__(self, activity_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Activity and Kappa Data/'
                                          '3a_Activity_RepeteadOnlyActivityMinutes.xlsx',
                 intensity=None):

        os.chdir("/Users/kyleweber/Desktop/")

        self.activity_data_file = activity_data_file
        self.intensity = intensity
        self.df_percent = None
        self.descriptive_stats = None
        self.shapiro_df = None
        self.levene_df = None
        self.aov = None
        self.posthoc = None
        self.df_ci = None

        """RUNS METHODS"""
        self.load_data()
        self.calculate_cis()

    def load_data(self):

        # Activity Minutes/Percent
        df = pd.read_excel(self.activity_data_file)

        self.df_percent = df[["ID", 'Group', 'Model', 'Sedentary%', 'Light%', 'Moderate%', 'Vigorous%']]

        self.df_percent["MVPA%"] = self.df_percent["Moderate%"] + self.df_percent["Vigorous%"]
        self.df_percent["TotalActive%"] = self.df_percent["Light%"] + self.df_percent["MVPA%"]

        # means = self.df_percent.describe().iloc[1, 1:]
        # sd = self.df_percent.describe().iloc[2, 1:]
        # self.descriptive_stats = pd.DataFrame(list(zip(means.keys(), means, sd)), columns=["Intensity", "Mean", "SD"])

        means = self.df_percent.groupby("Model").mean()
        stds = self.df_percent.groupby("Model").std()
        stds.columns = ["IDSD", "SedentarySD", "LightSD", "ModerateSD", "VigorousSD", "MVPASD", "TotalActiveSD"]

        self.descriptive_stats = pd.concat([means, stds], axis=1)
        self.descriptive_stats = self.descriptive_stats.drop(columns=["ID", "IDSD"])

    def check_assumptions(self, show_plots=False):

        shapiro_lists = []
        levene_lists = []
        by_model = self.df_percent.groupby("Model")

        for model_name in ["Ankle", "Wrist", "HR", "HR-Acc"]:
            for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
                result = scipy.stats.shapiro(by_model.get_group(model_name)[intensity])
                shapiro_lists.append({"Model": model_name, "Intensity": intensity,
                                      "W": result[0], "p": result[1], "Violation": result[1] <= .05})

        self.shapiro_df = pd.DataFrame(shapiro_lists, columns=["Model", "Intensity", "W", "p", "Violation"])

        for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
            levene = scipy.stats.levene(by_model.get_group("Wrist")[intensity], by_model.get_group("Ankle")[intensity])
            levene_lists.append({"SortIV": "Wrist-Ankle", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

            levene = scipy.stats.levene(by_model.get_group("Wrist")[intensity], by_model.get_group("HR")[intensity])
            levene_lists.append({"SortIV": "Wrist-HR", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

            levene = scipy.stats.levene(by_model.get_group("Wrist")[intensity], by_model.get_group("HR-Acc")[intensity])
            levene_lists.append({"SortIV": "Wrist-HRAcc", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

            levene = scipy.stats.levene(by_model.get_group("Ankle")[intensity], by_model.get_group("HR")[intensity])
            levene_lists.append({"SortIV": "Ankle-HR", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

            levene = scipy.stats.levene(by_model.get_group("Ankle")[intensity], by_model.get_group("HR-Acc")[intensity])
            levene_lists.append({"SortIV": "Ankle-HRAcc", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

            levene = scipy.stats.levene(by_model.get_group("HR")[intensity], by_model.get_group("HR-Acc")[intensity])
            levene_lists.append({"SortIV": "HR-HRAcc", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

        self.levene_df = pd.DataFrame(levene_lists, columns=["SortIV", "Intensity", "W", "p", "Violation"])

        if show_plots:
            by_model.boxplot(column=["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"])

    def perform_anova(self, intensity):
        self.aov = pg.rm_anova(data=self.df_percent, dv=intensity, within="Model", subject="ID", correction=True,
                               detailed=True)
        print(self.aov)

        self.posthoc = pg.pairwise_ttests(dv=intensity, subject='ID', within="Model",
                                          data=self.df_percent,
                                          padjust="bonf", effsize="hedges", parametric=True)
        print(self.posthoc)

    def plot_main_effects(self, error_bars="95% Conf."):

        plt.subplots(1, 3, figsize=(10, 6))
        plt.subplots_adjust(hspace=.3, wspace=0.25)

        plt.suptitle("Activity Totals by Model (Mean ± {}) [n={}]".format(error_bars, len(set(self.df_percent["ID"]))))

        # SEDENTARY
        plt.subplot(1, 3, 1)
        model_means = self.descriptive_stats["Sedentary%"]
        model_ci = self.df_ci["Sedentary"]
        plt.bar([i for i in model_means.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_ci], capsize=8, ecolor='black',
                color=["White", "silver", "grey", "#404042"], edgecolor='black', linewidth=2)
        plt.ylabel("% of Valid Data")
        plt.title("Sedentary")
        plt.yticks(np.arange(0, 120, 20))

        # LIGHT
        plt.subplot(1, 3, 2)
        model_means = self.descriptive_stats["Light%"]
        model_ci = self.df_ci["Light"]
        plt.bar([i for i in model_means.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_ci], capsize=8, ecolor='black',
                color=["White", "silver", "grey", "#404042"], edgecolor='black', linewidth=2)
        plt.ylabel(" ")
        plt.title("Light Activity")

        # MVPA
        plt.subplot(1, 3, 3)
        model_means = self.descriptive_stats["MVPA%"]
        model_ci = self.df_ci["MVPA"]
        plt.bar([i for i in model_means.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_ci], capsize=8, ecolor='black',
                color=["White", "silver", "grey", "#404042"], edgecolor='black', linewidth=2)
        plt.ylabel(" ")
        plt.title("MVPA")

    def calculate_cis(self):

        cis = []
        for column in [3, 4, 7]:
            data = self.df_percent[["ID", "Model", self.df_percent.keys()[column]]]

            for model in ["Ankle", "HR", "HR-Acc", "Wrist"]:
                ci_range = sms.DescrStatsW(data.groupby("Model").
                                           get_group(model)[self.df_percent.keys()[column]]).tconfint_mean()
                ci_width = (ci_range[1] - ci_range[0]) / 2

                cis.append(ci_width)

        output = np.array(cis).reshape(3, 4)

        self.df_ci = pd.DataFrame(output).transpose()
        self.df_ci.columns = ["Sedentary", "Light", "MVPA"]
        self.df_ci.insert(loc=0, column="Model", value=["Ankle", "HR", "HR-Acc", "Wrist"])


x = Objective1()
