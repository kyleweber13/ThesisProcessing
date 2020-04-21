import LocateUsableParticipants
from Subject import Subject
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import statsmodels.stats.power as smp
from matplotlib import pyplot as plt
import numpy as np
import scipy
import os
import seaborn as sns
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


class Objective3:

    def __init__(self, activity_data_file=None, kappa_data_file=None):

        os.chdir("/Users/kyleweber/Desktop/")

        self.activity_data_file = activity_data_file
        self.kappa_data_file = kappa_data_file
        self.df_mins = None
        self.df_percent = None
        self.df_kappa = None
        self.df_kappa_long = None
        self.shapiro_df = None
        self.levene_df = None
        self.aov = None
        self.kappa_aov = None
        self.posthoc_para = None
        self.posthoc_nonpara = None
        self.kappa_posthoc = None

        """RUNS METHODS"""
        self.load_data()
        # self.check_assumptions()
        self.perform_kappa_anova()

    def load_data(self):

        # Activity Minutes/Percent
        df = pd.read_excel(self.activity_data_file)

        self.df_mins = df[["ID", 'Group', 'Model', 'Sedentary', 'Light', 'Moderate', 'Vigorous']]
        self.df_percent = df[["ID", 'Group', 'Model', 'Sedentary%', 'Light%', 'Moderate%', 'Vigorous%']]

        self.df_percent["MVPA%"] = self.df_percent["Moderate%"] + self.df_percent["Vigorous%"]

        # Cohen's Kappa data
        self.df_kappa = pd.read_excel(self.kappa_data_file)

    def check_assumptions(self, show_plots=False):
        """Runs Shapiro-Wilk and Levene's test for each group x model combination and prints results.
           Shows boxplots sorted by group and model"""

        print("\n============================== Checking ANOVA assumptions ==============================")

        # Results df
        shapiro_lists = []

        levene_lists = []

        # Data sorted by Group
        by_group = self.df_percent.groupby("Group")

        for group_name in ["HIGH", "LOW"]:
            for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
                shapiro = scipy.stats.shapiro(by_group.get_group(group_name)[intensity])
                shapiro_lists.append({"SortIV": group_name, "Intensity": intensity,
                                      "W": shapiro[0], "p": shapiro[1], "Violation": shapiro[1] <= .05})

        for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
            levene = scipy.stats.levene(by_group.get_group("HIGH")[intensity], by_group.get_group("LOW")[intensity])
            levene_lists.append({"SortIV": "Group", "Intensity": intensity,
                                 "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

        # Data sorted by Model
        by_model = self.df_percent.groupby("Model")

        for model_name in ["Wrist", "Ankle", "HR", "HR-Acc"]:
            for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
                result = scipy.stats.shapiro(by_model.get_group(model_name)[intensity])
                shapiro_lists.append({"SortIV": model_name, "Intensity": intensity,
                                      "W": result[0], "p": result[1], "Violation": result[1] <= .05})

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

        self.shapiro_df = pd.DataFrame(shapiro_lists, columns=["SortIV", "Intensity", "W", "p", "Violation"])
        self.levene_df = pd.DataFrame(levene_lists, columns=["SortIV", "Intensity", "W", "p", "Violation"])

        print("\nSHAPIRO-WILK TEST FOR NORMALITY\n")
        print(self.shapiro_df)

        print("\nLEVENE TEST FOR HOMOGENEITY OF VARIANCE\n")
        print(self.levene_df)

        if show_plots:
            by_group.boxplot(column=["Sedentary%", "Light%", "Moderate%", "Vigorous%"])
            by_model.boxplot(column=["Sedentary%", "Light%", "Moderate%", "Vigorous%"])

    def perform_activity_anova(self, activity_intensity, data_type="percent"):

        if data_type == "percent":
            df = self.df_percent
            activity_intensity = activity_intensity + "%"
        if data_type == "minutes":
            df = self.df_mins

        # PLOTTING ---------------------------------------------------------------------------------------------------
        # Creates 2x1 subplots of group means
        plt.subplots(1, 2, figsize=(12, 7))
        plt.subplots_adjust(wspace=0.20)
        plt.suptitle("Group x Model Mixed ANOVA: {} Activity".format(activity_intensity))

        # Two activity level groups: one line for each intensity
        plt.subplot(1, 2, 1)
        sns.pointplot(data=df, x="Group", y=activity_intensity, hue="Model",
                      dodge=True, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.ylabel("{}".format(data_type.capitalize()))

        # Four intensity groups: one line for each activity level group
        plt.subplot(1, 2, 2)
        sns.pointplot(data=df, x="Model", y=activity_intensity, hue="Group",
                      dodge=True, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.ylabel("")

        # STATISTICAL ANALYSIS ---------------------------------------------------------------------------------------
        print("\nPerforming Group x Model mixed ANOVA on {} activity.".format(activity_intensity))

        # Group x Intensity mixed ANOVA
        self.aov = pg.mixed_anova(dv=activity_intensity, within="Model", between="Group", subject="ID", data=df,
                                  correction=True)
        pg.print_table(self.aov)

        group_p = self.aov.loc[self.aov["Source"] == "Group"]["p-unc"]
        group_sig = group_p.values[0] <= 0.05

        model_p = self.aov.loc[self.aov["Source"] == "Model"]["p-unc"]
        model_sig = model_p.values[0] <= 0.05

        interaction_p = self.aov.loc[self.aov["Source"] == "Interaction"]["p-unc"]
        interaction_sig = interaction_p.values[0] <= 0.05

        print("ANOVA quick summary:")
        if model_sig:
            print("-Main effect of Model (p = {})".format(round(model_p.values[0], 3)))
        if not model_sig:
            print("-No main effect of Model")
        if group_sig:
            print("-Main effect of Group (p = {})".format(round(group_p.values[0], 3)))
        if not group_sig:
            print("-No main effect of Group")
        if interaction_sig:
            print("-Signficiant Group x Model interaction (p = {})".format(round(interaction_p.values[0], 3)))
        if not interaction_sig:
            print("-No Group x Model interaction")

        posthoc_para = pg.pairwise_ttests(dv=activity_intensity, subject='ID',
                                          within="Model", between='Group',
                                          data=df,
                                          padjust="bonf", effsize="cohen", parametric=True)
        posthoc_nonpara = pg.pairwise_ttests(dv=activity_intensity, subject='ID',
                                             within="Model", between='Group',
                                             data=df,
                                             padjust="bonf", effsize="cohen", parametric=False)

        self.posthoc_para = posthoc_para
        self.posthoc_nonpara = posthoc_nonpara
        pg.print_table(posthoc_para)
        pg.print_table(posthoc_nonpara)

    def plot_main_effects(self, intensity):

        if intensity[-1] != "%":
            intensity += "%"

        plt.subplots(3, 1, figsize=(12, 7))
        plt.suptitle("{} Activity".format(intensity.capitalize()))

        plt.subplot(1, 3, 1)
        model_means = rp.summary_cont(self.df_percent.groupby(['Model']))[intensity.capitalize()]["Mean"]
        model_sd = rp.summary_cont(self.df_percent.groupby(['Model']))[intensity.capitalize()]["SD"]
        plt.bar([i for i in model_sd.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_sd], capsize=10, ecolor='black',
                color=["Red", "Blue", "Green", "Purple"], edgecolor='black', linewidth=2)
        plt.ylabel("% of Collection")
        plt.title("Model Means")

        plt.subplot(1, 3, 2)
        group_means = rp.summary_cont(self.df_percent.groupby(['Group']))[intensity.capitalize()]["Mean"]
        group_sd = rp.summary_cont(self.df_percent.groupby(['Group']))[intensity.capitalize()]["SD"]
        plt.bar([i for i in group_means.index], [100 * i for i in group_means.values],
                yerr=[i * 100 for i in group_sd], capsize=10, ecolor='black',
                color=["Grey", "White"], edgecolor='black', linewidth=2)
        plt.title("Group Means")

        plt.subplot(1, 3, 3)
        sns.pointplot(data=x.df_percent, x="Model", y=intensity.capitalize(), hue="Group",
                      dodge=True, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.title("All Combination Means")
        plt.ylabel(" ")

    def perform_kappa_anova(self):

        # MIXED ANOVA  ------------------------------------------------------------------------------------------------
        print("\nPerforming Group x Comparison mixed ANOVA on Cohen's Kappa values.")

        # Group x Intensity mixed ANOVA
        self.df_kappa_long = self.df_kappa.melt(id_vars=('ID', "Group"), var_name="Comparison", value_name="Kappa")

        self.kappa_aov = pg.mixed_anova(dv="Kappa", within="Comparison", between="Group", subject="ID",
                                        data=self.df_kappa_long, correction=True)
        pg.print_table(self.kappa_aov)

        # POST HOC ----------------------------------------------------------------------------------------------------
        self.kappa_posthoc = pg.pairwise_ttests(dv="Kappa", subject='ID', within="Comparison", between='Group',
                                                data=self.df_kappa_long,
                                                padjust="bonf", effsize="cohen", parametric=True)

    def calculate_generalized_eta_squared_kappa(self):
        pass

    def plot_mains_effects_kappa(self):

        plt.subplots(3, 1, figsize=(12, 7))
        plt.suptitle("Cohen's Kappas (mean ± SD; n=10)")
        plt.subplots_adjust(wspace=0.25)

        plt.subplot(1, 3, 1)
        comp_means = rp.summary_cont(self.df_kappa_long.groupby(['Comparison']))["Kappa"]["Mean"]
        comp_sd = rp.summary_cont(self.df_kappa_long.groupby(['Comparison']))["Kappa"]["SD"]

        plt.bar([i for i in comp_means.index], [i for i in comp_means.values],
                yerr=[i for i in comp_sd], capsize=8, ecolor='black',
                color=['purple', 'orange', 'red', 'yellow', 'blue', 'green'], edgecolor='black', linewidth=2)
        plt.ylabel("Kappa")
        plt.title("Model Comparison")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=10)

        plt.subplot(1, 3, 2)
        group_means = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["Mean"]
        group_sd = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["SD"]

        plt.bar([i for i in group_sd.index], [i for i in group_means.values],
                yerr=[i for i in group_sd], capsize=10, ecolor='black',
                color=["lightgrey", "dimgrey"], alpha=0.5, edgecolor='black', linewidth=2)
        plt.title("Activity Group")
        plt.xlabel("Activity Level")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=10)

        plt.subplot(1, 3, 3)
        sns.pointplot(data=self.df_kappa_long, x="Group", y="Kappa", hue="Comparison",
                      dodge=False, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.title("Interaction")
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.yticks(fontsize=10)
        plt.ylabel(" ")
        plt.xlabel("Activity Level")


x = Objective3(activity_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Activity Level Comparison/'
                                  'ActivityGroupsData_AllActivityMinutes.xlsx',
               kappa_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Kappas_RepeatedOnly.xlsx')
# x.perform_activity_anova("Sedentary")
