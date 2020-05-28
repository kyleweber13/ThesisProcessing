import LocateUsableParticipants
import pandas as pd
import scipy
import os
import pingouin as pg
import seaborn as sns
import numpy as np
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

usable_subjs = LocateUsableParticipants.SubjectSubset(check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/"
                                                                 "OND07_ProcessingStatus.xlsx",
                                                      wrist_ankle=False, wrist_hr=False,
                                                      wrist_hracc=False, hr_hracc=False,
                                                      ankle_hr=False, ankle_hracc=False,
                                                      wrist_only=False, ankle_only=False,
                                                      hr_only=False, hracc_only=False,
                                                      require_treadmill=True, require_all=True)


class Objective2:

    def __init__(self, data_file):

        # os.chdir("/Users/kyleweber/Desktop/")

        self.data_file = data_file
        self.df = None
        self.oneway_rm_aov = None
        self.ttests_unpaired = None
        self.ttests_paired = None
        self.corr_mat = None
        self.descriptive_stats_kappa = None
        self.kappa_cis = None

        """RUNS METHODS"""
        self.load_data()
        self.calculate_cis()
        self.pairwise_ttests_paired()

    def load_data(self):

        self.df = pd.read_excel(self.data_file, usecols=["ID", "Ankle-Wrist", "Wrist-HR", "Wrist-HRAcc",
                                                         "Ankle-HR", "Ankle-HRAcc", "HR-HRAcc"],
                                sheet_name="Data_WalkRun")

        self.corr_mat = self.df[["Ankle-Wrist", "Wrist-HR", "Wrist-HRAcc",
                                 "Ankle-HR", "Ankle-HRAcc", "HR-HRAcc"]].corr()

        self.descriptive_stats_kappa = self.df.describe().loc[["mean", "std"]].iloc[:, 1:].transpose()
        self.descriptive_stats_kappa.columns = ["Kappa_mean", "Kappa_sd"]

        self.descriptive_stats_kappa["Kappa_sem"] = self.descriptive_stats_kappa["Kappa_sd"] / self.df.shape[0] ** .5

    def check_normality(self):
        for col_name in self.df.keys():
            result = scipy.stats.shapiro(self.df[col_name].dropna())
            print(col_name, ":", "W =", round(result[0], 3), "p =", round(result[1], 3))

    def pairwise_ttests_paired(self):

        df = self.df.melt(id_vars="ID")

        self.oneway_rm_aov = pg.rm_anova(data=df, dv="value", within="variable", subject='ID')

        self.ttests_paired = pg.pairwise_ttests(dv="value", subject='ID',
                                                within='variable', data=df,
                                                padjust="holm", effsize="hedges", parametric=True)

    def calculate_cis(self):

        cis = []
        model_list = ["Wrist-HR", "Ankle-HR", "Wrist-HRAcc", "Ankle-Wrist", "Ankle-HRAcc", "HR-HRAcc"]
        for model in model_list:
            ci_range = sms.DescrStatsW(self.df[model]).tconfint_mean()
            ci_width = (ci_range[1] - ci_range[0]) / 2
            cis.append(ci_width)

        self.kappa_cis = pd.DataFrame(list(zip(model_list, cis)))
        self.kappa_cis.columns = ["Comparison", "95%CI Width"]

    def plot_boxplot(self):

        df = self.df.copy()
        df["ID_num"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        long = df.iloc[:, 1:].melt(id_vars=["ID_num"], value_name="Kappa")
        long.columns = ["ID", "Comparison", "Kappa"]

        sns.set(style="ticks")

        # Boxplot sorted by group means
        ax = sns.boxplot(data=df.iloc[:, 1:-1], order=[i for i in df.iloc[1:-1].describe().
                         iloc[1, ].sort_values().index][:-2], color='white', fliersize=0)

        # Adds individual datapoints
        ax = sns.stripplot(data=long, x="Comparison", y="Kappa", hue="ID", palette="bright", jitter=True,
                           edgecolor='black', order=[i for i in df.iloc[1:-1].describe().
                           iloc[1, ].sort_values().index][:-2])

        ax.legend_.remove()
        ax.set_xlabel("")
        ax.set_ylabel("Cohen's Kappa")
        ax.set_yticks(np.arange(0, 1.1, .1))
        ax.set_title("Objective #2: Kappa Boxplot by Model")
        ax.set_ylim(0, 1)

    def kappa_scatterplot(self):

        marker_type = ["o", "^", "s", "o", "^", "s"]
        colours = ["white", "white", "white", "black", "black", "black"]

        for col in range(1, 7):
            plt.scatter(np.arange(1, self.df.shape[0]+1), self.df.iloc[:, col], label=self.df.keys()[col],
                        marker=marker_type[col-1], color=colours[col-1], edgecolors="black")

        plt.title("Objective #2: Cohen's Kappa by Model and Participant")
        plt.xticks(np.arange(1, self.df.shape[0]+4))
        plt.ylabel("Cohen's Kappa")
        plt.ylim(0, 1.1)
        plt.yticks(np.arange(0, 1.1, .1))
        plt.legend(loc='upper right')

    def plot_means(self):

        plt.title("Objective #2: Cohen's Kappa by Model Comparison (Mean ± 95%CI) [n={}]".format(self.df.shape[0]))
        plt.bar([i for i in self.descriptive_stats_kappa.sort_values("Kappa_mean").index],
                self.descriptive_stats_kappa.sort_values("Kappa_mean")["Kappa_mean"],
                yerr=self.kappa_cis["95%CI Width"], capsize=8, ecolor='black',
                color=["purple", "blue", "green", "red", "darkorange", 'gold'],
                edgecolor='black', linewidth=2, alpha=.65)
        plt.ylim(0, 1.1, .1)
        plt.ylabel("Cohen's Kappa")
        plt.yticks(np.arange(0, 1.1, .1))


x = Objective2(data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Activity and Kappa Data/'
                         '3b_Kappa_RepeatedParticipantsOnly.xlsx')
