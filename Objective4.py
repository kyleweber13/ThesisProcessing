import LocateUsableParticipants
import pandas as pd
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
        self.kappa_shapiro_df = None
        self.kappa_levene_df = None
        self.descriptive_stats_activity = None
        self.descriptive_stats_kappa = None
        self.aov = None
        self.kappa_ttest = None
        self.kappa_wilcoxon = None
        self.posthoc_para = None
        self.posthoc_nonpara = None

        """RUNS METHODS"""
        self.load_data()
        self.check_kappa_assumptions()
        self.perform_kappa_ttest()

    def load_data(self):

        # Activity Minutes/Percent ------------------------------------------------------------------------------------
        df = pd.read_excel(self.activity_data_file)

        self.df_mins = df[["ID", 'Group', 'Model', 'Sedentary', 'Light', 'Moderate', 'Vigorous']]
        self.df_percent = df[["ID", 'Group', 'Model', 'Sedentary%', 'Light%', 'Moderate%', 'Vigorous%']]

        self.df_percent["MVPA%"] = self.df_percent["Moderate%"] + self.df_percent["Vigorous%"]

        model_d = self.df_percent.groupby("Model").describe()
        group_d = self.df_percent.groupby("Group").describe()

        group_d = pd.concat([group_d, model_d])

        self.descriptive_stats_activity = pd.DataFrame(list(zip(["HIGH", "LOW", "Ankle", "Wrist"],
                                                                group_d["Sedentary%"]["mean"],
                                                                group_d["Sedentary%"]["std"],
                                                                group_d["Light%"]["mean"],
                                                                group_d["Light%"]["std"],
                                                                group_d["Moderate%"]["mean"],
                                                                group_d["Moderate%"]["std"],
                                                                group_d["Vigorous%"]["mean"],
                                                                group_d["Vigorous%"]["std"],
                                                                group_d["MVPA%"]["mean"],
                                                                group_d["MVPA%"]["std"])),
                                                       columns=["IV", "Sedentary_mean", "Sedentary_sd",
                                                                "Light_mean", "Light_sd",
                                                                "Moderate_mean", "Moderate_sd",
                                                                "Vigorous_mean", "Vigorous_sd",
                                                                "MVPA_mean", "MVPA_sd"])

        # Cohen's Kappa data -----------------------------------------------------------------------------------------
        self.df_kappa = pd.read_excel(self.kappa_data_file, sheet_name="Data")
        self.df_kappa_long = self.df_kappa.melt(id_vars=('ID', "Group"), var_name="Comparison", value_name="Kappa")

        comp_d = self.df_kappa_long.groupby("Comparison").describe()
        group_d = self.df_kappa_long.groupby("Group").describe()

        group_d = pd.concat([group_d, comp_d])

        self.descriptive_stats_kappa = pd.DataFrame(list(zip(["HIGH", "LOW", "Ankle-HR", "Ankle-HRAcc", "Ankle-Wrist",
                                                              "HR-HRAcc", "Wrist-HR", "Wrist-HRAcc"],
                                                             group_d["Kappa"]["mean"], group_d["Kappa"]["std"])),
                                                    columns=["IV", "Kappa_mean", "Kappa_sd"])

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

        for model_name in ["Wrist", "Ankle"]:
            for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
                result = scipy.stats.shapiro(by_model.get_group(model_name)[intensity])
                shapiro_lists.append({"SortIV": model_name, "Intensity": intensity,
                                      "W": result[0], "p": result[1], "Violation": result[1] <= .05})

        for intensity in ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]:
            levene = scipy.stats.levene(by_model.get_group("Wrist")[intensity], by_model.get_group("Ankle")[intensity])
            levene_lists.append({"SortIV": "Wrist-Ankle", "Intensity": intensity,
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
        plt.title("Group x Model Mixed ANOVA: {} Activity".format(activity_intensity))

        # Two activity level groups: one line for each intensity
        sns.pointplot(data=df, x="Group", y=activity_intensity, hue="Model",
                      dodge=True, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.ylabel("{}".format(data_type.capitalize()))

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
                                          padjust="bonf", effsize="hedges", parametric=True)
        posthoc_nonpara = pg.pairwise_ttests(dv=activity_intensity, subject='ID',
                                             within="Model", between='Group',
                                             data=df,
                                             padjust="bonf", effsize="hedges", parametric=False)

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

    def check_kappa_assumptions(self, show_plots=False):
        """Runs Shapiro-Wilk and Levene's test for each group x model combination and prints results.
           Shows boxplots sorted by group and model"""

        print("\n============================== Checking normality for Kappa groups... ==============================")

        high = self.df_kappa_long.groupby("Group").get_group("HIGH")["Kappa"]
        low = self.df_kappa_long.groupby("Group").get_group("LOW")["Kappa"]

        high_shapiro = scipy.stats.shapiro(high)
        low_shapiro = scipy.stats.shapiro(low)

        print("High activity group: W = {}, p = {}".format(round(high_shapiro[0], 3), round(high_shapiro[1], 3)))
        print("Low activity group: W = {}, p = {}".format(round(low_shapiro[0], 3), round(low_shapiro[1], 3)))

    def perform_kappa_ttest(self):

        # MIXED ANOVA  ------------------------------------------------------------------------------------------------
        print("\nPerforming unpaired T-test between activity groups on Cohen's Kappa values.")

        high = self.df_kappa_long.groupby("Group").get_group("HIGH")["Kappa"]
        low = self.df_kappa_long.groupby("Group").get_group("LOW")["Kappa"]

        self.kappa_ttest = pg.ttest(high, low, paired=False, correction="auto")

        # Approximates hedges g using d x (1 - (3 / (4*(n1 + n2) - 9))
        self.kappa_ttest["hedges-g"] = self.kappa_ttest["cohen-d"] * (1 - (3 / (4 * 2*high.shape[0] - 9)))
        print(self.kappa_ttest)

        self.kappa_wilcoxon = pg.wilcoxon(high, low)
        print(self.kappa_wilcoxon)

    def plot_mains_effects_kappa(self):

        group_means = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["Mean"]
        group_sd = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["SD"]

        plt.bar([i for i in group_sd.index], [i for i in group_means.values],
                yerr=[i for i in group_sd], capsize=10, ecolor='black',
                color=["lightgrey", "dimgrey"], alpha=0.5, edgecolor='black', linewidth=2)
        plt.title("Activity Group Comparison: t = {}, p = {}".format(self.kappa_ttest["T"][0],
                                                                     round(self.kappa_ttest["p-val"][0], 3)))

        plt.ylabel("Kappa")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=10)


x = Objective3(activity_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/'
                                  'Activity and Kappa Data/ActivityGroupsData_AnkleWristActivityMinutes.xlsx',
               kappa_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/'
                               'Activity and Kappa Data/Kappa - AnkleWrist.xlsx')
# x.check_kappa_assumptions(False)
# x.perform_kappa_ttest()
x.perform_activity_anova("Moderate")
