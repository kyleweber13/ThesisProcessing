import LocateUsableParticipants
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import os
import researchpy as rp
import pingouin as pg

usable_subjs = LocateUsableParticipants.SubjectSubset(check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/"
                                                                 "OND07_ProcessingStatus.xlsx",
                                                      wrist_ankle=False, wrist_hr=False,
                                                      wrist_hracc=False, hr_hracc=False,
                                                      ankle_hr=False, ankle_hracc=False,
                                                      wrist_only=False, ankle_only=False,
                                                      hr_only=False, hracc_only=False,
                                                      require_treadmill=True, require_all=True)


class Objective1:

    def __init__(self, activity_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/'
                                          'Activity Level Comparison/ActivityGroupsData_AllActivityMinutes.xlsx',
                 intensity=None):

        os.chdir("/Users/kyleweber/Desktop/")

        self.activity_data_file = activity_data_file
        self.intensity = intensity
        self.df_percent = None
        self.shapiro_df = None
        self.levene_df = None
        self.aov = None
        self.posthoc = None

        """RUNS METHODS"""
        self.load_data()
        self.perform_anova(intensity=self.intensity)

    def load_data(self):

        # Activity Minutes/Percent
        df = pd.read_excel(self.activity_data_file)

        self.df_percent = df[["ID", 'Group', 'Model', 'Sedentary%', 'Light%', 'Moderate%', 'Vigorous%']]

        self.df_percent["MVPA%"] = self.df_percent["Moderate%"] + self.df_percent["Vigorous%"]

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

    def plot_main_effects(self):

        plt.subplots(2, 2, figsize=(12, 7))
        plt.subplots_adjust(hspace=.30)

        plt.suptitle("Effect of Model on Total Activity")

        plt.subplot(2, 2, 1)
        model_means = rp.summary_cont(self.df_percent.groupby(['Model']))["Sedentary%"]["Mean"]
        model_sd = rp.summary_cont(self.df_percent.groupby(['Model']))["Sedentary%"]["SD"]
        plt.bar([i for i in model_sd.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_sd], capsize=10, ecolor='black',
                color=["Red", "Blue", "Green", "Purple"], edgecolor='black', linewidth=2)
        plt.ylabel("% of Collection")
        plt.title("Sedentary")

        plt.subplot(2, 2, 2)
        model_means = rp.summary_cont(self.df_percent.groupby(['Model']))["Light%"]["Mean"]
        model_sd = rp.summary_cont(self.df_percent.groupby(['Model']))["Light%"]["SD"]
        plt.bar([i for i in model_sd.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_sd], capsize=10, ecolor='black',
                color=["Red", "Blue", "Green", "Purple"], edgecolor='black', linewidth=2)
        plt.ylabel(" ")
        plt.title("Light")

        plt.subplot(2, 2, 3)
        model_means = rp.summary_cont(self.df_percent.groupby(['Model']))["Moderate%"]["Mean"]
        model_sd = rp.summary_cont(self.df_percent.groupby(['Model']))["Moderate%"]["SD"]
        plt.bar([i for i in model_sd.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_sd], capsize=10, ecolor='black',
                color=["Red", "Blue", "Green", "Purple"], edgecolor='black', linewidth=2)
        plt.ylabel("% of Collection")
        plt.title("Moderate")

        plt.subplot(2, 2, 4)
        model_means = rp.summary_cont(self.df_percent.groupby(['Model']))["Vigorous%"]["Mean"]
        model_sd = rp.summary_cont(self.df_percent.groupby(['Model']))["Vigorous%"]["SD"]
        plt.bar([i for i in model_sd.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_sd], capsize=10, ecolor='black',
                color=["Red", "Blue", "Green", "Purple"], edgecolor='black', linewidth=2)
        plt.ylabel(" ")
        plt.title("Vigorous")


# x = Objective1(intensity="Sedentary%")
