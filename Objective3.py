import LocateUsableParticipants
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy
import os
import seaborn as sns
import researchpy as rp
import pingouin as pg
import statsmodels.stats.api as sms

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
        self.kappa_aov = None
        self.posthoc_para = None
        self.posthoc_nonpara = None
        self.kappa_posthoc = None
        self.df_kappa_ci = None
        self.df_ci = None

        """RUNS METHODS"""
        self.load_data()
        # self.check_assumptions()
        self.calculate_cis()
        self.calculate_kappa_cis()

    def load_data(self):

        # Activity Minutes/Percent ------------------------------------------------------------------------------------
        df = pd.read_excel(self.activity_data_file)

        self.df_mins = df[["ID", 'Group', 'Model', 'Sedentary', 'Light', 'Moderate', 'Vigorous']]
        self.df_percent = df[["ID", 'Group', 'Model', 'Sedentary%', 'Light%', 'Moderate%', 'Vigorous%']]

        self.df_percent["MVPA%"] = self.df_percent["Moderate%"] + self.df_percent["Vigorous%"]

        model_d = self.df_percent.groupby("Model").describe()
        group_d = self.df_percent.groupby("Group").describe()

        group_d = pd.concat([group_d, model_d])

        self.descriptive_stats_activity = pd.DataFrame(list(zip(["HIGH", "LOW", "Ankle", "HR", "HR-Acc", "Wrist"],
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

    def calculate_cis(self):

        cis = []
        for column in [3, 4, 7]:
            data = self.df_percent[["ID", "Group", "Model", self.df_percent.keys()[column]]]

            for model in ["Ankle", "HR", "HR-Acc", "Wrist"]:
                ci_range = sms.DescrStatsW(data.groupby(["Group", "Model"]).
                                           get_group(("HIGH", model))[self.df_percent.keys()[column]]).tconfint_mean()
                ci_width_h = (ci_range[1] - ci_range[0]) / 2

                ci_range = sms.DescrStatsW(data.groupby(["Group", "Model"]).
                                           get_group(("LOW", model))[self.df_percent.keys()[column]]).tconfint_mean()
                ci_width_l = (ci_range[1] - ci_range[0]) / 2

                print("{} - {}: HIGH = {}; LOW = {}".format(model, self.df_percent.keys()[column],
                                                            round(ci_width_h, 5), round(ci_width_l, 5)))

                cis.append(ci_width_h)
                cis.append(ci_width_l)
        output = np.array(cis).reshape(3, 8)

        self.df_ci = pd.DataFrame(output).transpose()
        self.df_ci.columns = ["Sedentary", "Light", "MVPA"]
        self.df_ci.insert(loc=0, column="Group", value=["HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW"])
        self.df_ci.insert(loc=0, column="Model", value=["Ankle", "Ankle", "HR", "HR",
                                                        "HR-Acc", "HR-Acc", "Wrist", "Wrist"])

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
                                  correction='auto')
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
        model_means = rp.summary_cont(self.df_percent.groupby(['Model']))[intensity]["Mean"]
        model_sd = rp.summary_cont(self.df_percent.groupby(['Model']))[intensity]["SD"]
        plt.bar([i for i in model_sd.index], [100 * i for i in model_means.values],
                yerr=[i * 100 for i in model_sd], capsize=10, ecolor='black',
                color=["Red", "Blue", "Green", "Purple"], edgecolor='black', linewidth=2)
        plt.ylabel("% of Collection")
        plt.title("Model Means")

        plt.subplot(1, 3, 2)
        group_means = rp.summary_cont(self.df_percent.groupby(['Group']))[intensity]["Mean"]
        group_sd = rp.summary_cont(self.df_percent.groupby(['Group']))[intensity]["SD"]
        plt.bar([i for i in group_means.index], [100 * i for i in group_means.values],
                yerr=[i * 100 for i in group_sd], capsize=10, ecolor='black',
                color=["Grey", "White"], edgecolor='black', linewidth=2)
        plt.title("Group Means")

        plt.subplot(1, 3, 3)
        sns.pointplot(data=x.df_percent, x="Model", y=intensity, hue="Group",
                      dodge=True, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.title("All Combination Means")
        plt.ylabel(" ")

    def plot_interaction(self):

        # Creates DF with values as % of 100 instead of decimal
        temp_df = self.df_percent.copy()
        temp_df = temp_df.iloc[:, 3:]*100
        temp_df["ID"] = self.df_percent["ID"]
        temp_df["Group"] = self.df_percent["Group"]
        temp_df["Model"] = self.df_percent["Model"]

        plt.subplots(3, 1)
        plt.suptitle("Total Activity by Activity Group (n={}/group)".format(int(self.df_percent.shape[0]/8)))

        plt.subplot(1, 3, 1)
        sns.pointplot(data=temp_df, x="Group", y="Sedentary%", hue="Model", order=["LOW", "HIGH"], ci=95,
                      dodge=True, capsize=.1, errwidth=1, palette='Set1', scale=.8, markers=".")
        plt.title("Sedentary")
        plt.ylabel("% of Valid Data")
        plt.ylim(0, 100)
        plt.xlabel(" ")

        plt.subplot(1, 3, 2)
        ax = sns.pointplot(data=temp_df, x="Group", y="Light%", hue="Model", order=["LOW", "HIGH"], ci=95,
                           dodge=True, capsize=.1, errwidth=1, palette='Set1', scale=.8, markers=".")
        plt.title("Light Activity")
        plt.ylabel(" ")
        plt.xlabel(" ")
        plt.ylim(0, 24)
        ax.legend_.remove()

        plt.subplot(1, 3, 3)
        ax = sns.pointplot(data=temp_df, x="Group", y="MVPA%", hue="Model", order=["LOW", "HIGH"], ci=95,
                           dodge=True, capsize=.1, errwidth=1, palette="Set1", scale=.8, markers=".")
        plt.title("MVPA")
        plt.ylabel(" ")
        plt.xlabel(" ")
        plt.ylim(0, 15)
        ax.legend_.remove()

    def check_kappa_assumptions(self, show_plots=False):
        """Runs Shapiro-Wilk and Levene's test for each group x model combination and prints results.
           Shows boxplots sorted by group and model"""

        print("\n============================== Checking ANOVA assumptions ==============================")

        # Results df
        shapiro_lists = []

        levene_lists = []

        # Data sorted by Group
        by_group = self.df_kappa_long.groupby("Group")

        for group_name in ["HIGH", "LOW"]:
            shapiro = scipy.stats.shapiro(by_group.get_group(group_name)["Kappa"])
            shapiro_lists.append({"SortIV": group_name, "W": shapiro[0],
                                  "p": shapiro[1], "Violation": shapiro[1] <= .05})

        # Data sorted by comparison
        by_comparison = self.df_kappa_long.groupby("Comparison")

        for comp_name in ["Ankle-Wrist", "Wrist-HR", "Wrist-HRAcc", "Ankle-HR", "Ankle-HRAcc", "HR-HRAcc"]:
            shapiro = scipy.stats.shapiro(by_comparison.get_group(comp_name)["Kappa"])
            shapiro_lists.append({"SortIV": comp_name, "W": shapiro[0],
                                  "p": shapiro[1], "Violation": shapiro[1] <= .05})

        # Levene's test
        levene = scipy.stats.levene(by_group.get_group("HIGH")["Kappa"], by_group.get_group("LOW")["Kappa"])
        levene_lists.append({"SortIV": "Group", "W": levene[0], "p": levene[1], "Violation": levene[1] <= .05})

        self.kappa_shapiro_df = pd.DataFrame(shapiro_lists, columns=["SortIV", "W", "p", "Violation"])
        self.kappa_levene_df = pd.DataFrame(levene_lists, columns=["SortIV", "W", "p", "Violation"])

        print("\nSHAPIRO-WILK TEST FOR NORMALITY\n")
        print(self.kappa_shapiro_df)

        print("\nLEVENE TEST FOR HOMOGENEITY OF VARIANCE\n")
        # print(self.kappa_levene_df)

        if show_plots:
            by_group.boxplot(column=["Sedentary%", "Light%", "Moderate%", "Vigorous%"])
            by_model.boxplot(column=["Sedentary%", "Light%", "Moderate%", "Vigorous%"])

    def perform_kappa_anova(self):

        # MIXED ANOVA  ------------------------------------------------------------------------------------------------
        print("\nPerforming Group x Comparison mixed ANOVA on Cohen's Kappa values.")

        # Group x Intensity mixed ANOVA
        self.kappa_aov = pg.mixed_anova(dv="Kappa", within="Comparison", between="Group", subject="ID",
                                        data=self.df_kappa_long, correction=True)
        pg.print_table(self.kappa_aov)

        # POST HOC ----------------------------------------------------------------------------------------------------
        self.kappa_posthoc = pg.pairwise_ttests(dv="Kappa", subject='ID', within="Comparison", between='Group',
                                                data=self.df_kappa_long,
                                                padjust="bonf", effsize="cohen", parametric=True)

    def calculate_kappa_cis(self):

        high_cis = []
        low_cis = []
        for column in range(2, 8):
            data = self.df_kappa[["ID", "Group", self.df_kappa.keys()[column]]]

            ci_range = sms.DescrStatsW(data.groupby("Group").
                                       get_group("HIGH")[self.df_kappa.keys()[column]]).tconfint_mean()
            ci_width_h = (ci_range[1] - ci_range[0]) / 2

            ci_range = sms.DescrStatsW(data.groupby("Group").
                                       get_group("LOW")[self.df_kappa.keys()[column]]).tconfint_mean()
            ci_width_l = (ci_range[1] - ci_range[0]) / 2

            print("{}: HIGH = {}; LOW = {}".format(data.keys()[2], round(ci_width_h, 5), round(ci_width_l, 5)))

            high_cis.append(ci_width_h)
            low_cis.append(ci_width_l)

        self.df_kappa_ci = pd.DataFrame(list(zip(high_cis, low_cis))).transpose()
        self.df_kappa_ci.columns = self.df_kappa.keys()[2:]
        self.df_kappa_ci.insert(loc=0, column="Group", value=["HIGH", "LOW"])

    def plot_main_effects_kappa(self):

        n_per_group = int(len(set(self.df_kappa_long["ID"])) / len(set(self.df_kappa_long["Group"])))

        plt.subplots(3, 1, figsize=(12, 7))
        plt.suptitle("Cohen's Kappas (mean ± SD; n/group={})".format(n_per_group))
        plt.subplots_adjust(wspace=0.25)

        # COMPARISON MEANS --------------------------------------------------------------------------------------------
        plt.subplot(1, 3, 1)
        comp_means = rp.summary_cont(self.df_kappa_long.groupby(['Comparison']))["Kappa"]["Mean"]

        # DF = n - 1
        t_crit = scipy.stats.t.ppf(.95, self.df_kappa.shape[0] - 1)

        comp_ci = rp.summary_cont(self.df_kappa_long.groupby(['Comparison']))["Kappa"]["SD"] / \
                  (self.df_kappa.shape[0] ** .5) * t_crit

        plt.bar([i for i in comp_means.index], [i for i in comp_means.values],
                yerr=[i for i in comp_ci], capsize=8, ecolor='black',
                color=['purple', 'orange', 'red', 'yellow', 'blue', 'green'], edgecolor='black', linewidth=2)

        plt.ylabel("Cohen's Kappa")
        plt.title("Model Comparison")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=10)

        # GROUP MEANS -------------------------------------------------------------------------------------------------
        plt.subplot(1, 3, 2)
        group_means = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["Mean"]

        # DF = n_per_group * n_comparisons - 1
        t_crit = scipy.stats.t.ppf(.95, n_per_group * (self.df_kappa.shape[1] - 2) - 1)

        # SEM calculated as SD / root(n_per_group * n_comparisons)
        group_ci = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["SD"] / \
                   ((n_per_group * (self.df_kappa.shape[1] - 2)) ** .5) * t_crit

        # Adds group CIs to df_kappa_ci
        self.df_kappa_ci["ALL"] = list(group_ci)

        plt.bar([i for i in group_means.index], [i for i in group_means.values],
                yerr=[i for i in group_ci], capsize=10, ecolor='black',
                color=["lightgrey", "dimgrey"], alpha=0.5, edgecolor='black', linewidth=2)

        plt.title("Activity Group")
        plt.xlabel("Activity Level")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=10)

        plt.subplot(1, 3, 3)
        sns.pointplot(data=self.df_kappa_long, x="Group", y="Kappa", hue="Comparison", ci=95,
                      dodge=False, markers='o', capsize=.1, errwidth=1, palette='Set1')
        plt.title("Interaction")
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.yticks(fontsize=10)
        plt.ylabel(" ")
        plt.xlabel("Activity Level")

    def plot_kappa_interaction(self):

        sns.pointplot(data=self.df_kappa_long, x="Group", y="Kappa", hue="Comparison", ci=95,
                      dodge=False, markers='o', capsize=.1, errwidth=1, palette='Set1')

        plt.title("Interaction")
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.yticks(fontsize=10)
        plt.ylabel("Cohen's Kappa")
        plt.xlabel("Activity Level")
        plt.legend(loc='lower left', fontsize=10)

    def plot_activity_group_means(self, error_bars="95%CI"):

        if error_bars == "95%CI":
            ci_range = sms.DescrStatsW(self.df_kappa_long.groupby("Group").get_group("HIGH")["Kappa"]).tconfint_mean()
            ci_width_h = (ci_range[1] - ci_range[0]) / 2

            ci_range = sms.DescrStatsW(self.df_kappa_long.groupby("Group").get_group("LOW")["Kappa"]).tconfint_mean()
            ci_width_l = (ci_range[1] - ci_range[0]) / 2

            e_bars = [ci_width_h, ci_width_l]

        if error_bars == "SD":
            e_bars = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["SD"]

        group_means = rp.summary_cont(self.df_kappa_long.groupby(['Group']))["Kappa"]["Mean"]

        plt.bar([i for i in group_means.index], [i for i in group_means.values],
                yerr=[i for i in e_bars], capsize=8, ecolor='black',
                color=["white", "dimgrey"], edgecolor='black', alpha=0.5, linewidth=2)
        plt.title("Cohen's Kappa by Activity Group")

        plt.ylabel("Cohen's Kappa")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.yticks(fontsize=10)


x = Objective3(activity_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Activity and Kappa Data/'
                                  '3a_Activity_RepeteadOnlyActivityMinutes.xlsx',
               kappa_data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Activity and Kappa Data/'
                               '3b_Kappa_RepeatedParticipantsOnly.xlsx')


class KappaMethod:

    def __init__(self):
        """Statistical analysis to determine if kappa values for the Ankle-Wrist comparison using ValidAll vs.
           AccelOnly data are different.

            Runs two-way mixed ANOVA (Group x DataMethod)
        """

        self.data = None
        self.data_long = None
        self.aov = None

        """RUNS METHOD"""
        self.organize_data()
        self.run_stats()

    def organize_data(self):

        id = [3024, 3026, 3029, 3030, 3031, 3032, 3034, 3037, 3039, 3043]
        groups = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
        ao = [.501, .4236, .5418, .6405, .4772, .4913, .3483, .3001, .3516, .6436]
        va = [.5466, .4728, .5615, .571, .559, .5449, .3237, .4116, .3456, .7124]

        self.data = pd.DataFrame(list(zip(id, groups, ao, va)), columns=["ID", "Group", "AccelOnly", "ValidAll"])

        self.data_long = self.data.melt(id_vars=('ID', "Group"), var_name="Method", value_name="Kappa")

    def run_stats(self):

        self.aov = pg.mixed_anova(dv="Kappa", within="Method", between="Group", subject="ID",
                                  data=self.data_long, correction=True)

        print(self.aov)
