import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

tm_file = "/Users/kyleweber/Desktop/Data/OND07/Processed Data/Treadmill/TreadmillRegression_Individual.xlsx"
ecg_file = "/Users/kyleweber/Desktop/Data/OND07/Processed Data/Tabular Data/ECG_QualityControl_Testing_KW.xlsx"


class Treadmill:

    def __init__(self, file):

        self.df = pd.read_excel(file, usecols=["ID", "Counts", "TrueSpeed", "PredSpeed", "Difference"])

        self.df_stats = pd.read_excel(file, sheet_name="Stats", usecols=["SEE", "r2"])

        self.df_eq = pd.read_excel(file, sheet_name="Equations")

    def plot_all_data(self):

        # Graph marker things
        icons = []
        for c in ["red", 'dodgerblue', 'black', 'white']:
            for s in ['o', '^', 'P', 's', 'D', "*"]:
                icons.append(c + " " + s)

        ids = [i for i in set(self.df["ID"])]
        data = self.df.groupby("ID")

        for id, loop_num in zip(ids, np.arange(0, int(self.df.shape[0] / self.df.shape[1]), 1)):

            df = data.get_group(id)

            # Graphing
            plt.scatter(df["Counts"], df["TrueSpeed"], s=25, label="",
                        edgecolors='black', marker=icons[loop_num].split(" ")[-1], c=icons[loop_num].split(" ")[0])

        plt.ylabel("Speed (m/s)")
        plt.xlabel("Counts (15-s epochs)")
        plt.title("All Treadmill Walk Data by Participant")

    def plot_scatter(self):

        print("\nCONFIDENCE INTERVAL CALCULATION IS WRONG")

        t_crit = scipy.stats.t.ppf(.95, len(set(self.df["ID"])) - 1)

        ci_width = t_crit * self.df_stats["SEE"].describe().iloc[1]

        true_diff_r = scipy.stats.pearsonr(self.df["TrueSpeed"], self.df["Difference"])

        colours = ["black", 'dimgray', 'lightgray', 'white']
        markers = ['o', '^', 'P', 's', 'D', "*"]

        plt.subplots(1, 2, figsize=(10, 6))
        plt.subplots_adjust(wspace=.26)
        plt.suptitle("True vs. Predicted Speeds (Individual SEE 95%CI)")

        for loop_num, subj in zip(np.arange(0, len(set(self.df["ID"]))), set(self.df["ID"])):
            data = self.df.loc[self.df['ID'] == subj]

            plt.subplot(1, 2, 1)
            plt.scatter(data["TrueSpeed"], data["PredSpeed"], s=30, edgecolors='black', c='white')

            plt.subplot(1, 2, 2)
            plt.scatter(data["TrueSpeed"], data["TrueSpeed"] - data["PredSpeed"], s=30,
                        edgecolors='black', c=colours[loop_num % 4], marker=markers[loop_num % 5])

        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0.25, 2.5, .1), np.arange(0.25, 2.5, .1),
                 linestyle='dashed', color='black')
        plt.fill_between(x=np.arange(.25, 2.5, .1),
                         y1=[i for i in np.arange(.25, 2.5, .1) - ci_width],
                         y2=[i for i in np.arange(.25, 2.5, .1) + ci_width],
                         color='red', alpha=.35)
        plt.xticks(np.arange(.25, 2.5, .25))
        plt.yticks(np.arange(.25, 2.5, .25))
        plt.title("All Participants (n={})".format(len(set(self.df["ID"]))))
        plt.xlabel("True Speed (m/s)")
        plt.ylabel("Predicted Speed (m/s)")

        plt.subplot(1, 2, 2)

        plt.plot(np.arange(.25, 2.5, .25), [ci_width for i in range(len(np.arange(.25, 2.5, .25)))],
                 linestyle='dashed', color='black')
        plt.plot(np.arange(.25, 2.5, .25), [-ci_width for i in range(len(np.arange(.25, 2.5, .25)))],
                 linestyle='dashed', color='black')

        plt.xticks(np.arange(.25, 2.5, .25))

        plt.title("Difference Score (r = {}, p = {})".format(round(true_diff_r[0], 3), round(true_diff_r[1], 3)))
        plt.xlabel("True Speed (m/s)")
        plt.ylabel("Difference (True - Predicted) (m/s)")

    def plot_scatter_v2(self):

        print("\nCONFIDENCE INTERVAL CALCULATION IS WRONG")

        # ADDITIONAL INFORMATION -------------------------------------------------------------------------------------
        t_crit = scipy.stats.t.ppf(.95, len(set(self.df["ID"])) - 1)

        ci_width = t_crit * self.df_stats["SEE"].describe().iloc[1]

        true_diff_r = scipy.stats.pearsonr(self.df["TrueSpeed"], self.df["Difference"])
        counts_speed_r = scipy.stats.pearsonr(self.df["Counts"], self.df["TrueSpeed"])

        colours = ["black", 'dimgray', 'lightgray', 'white']
        markers = ['o', '^', 'P', 's', 'D', "*"]

        # SETS UP PLOT
        plt.subplots(2, 2, figsize=(10, 6))
        plt.subplots_adjust(wspace=.26, hspace=.4)

        # GENERATING MAIN PLOTS IN LOOP -------------------------------------------------------------------------------
        for loop_num, subj in zip(np.arange(0, len(set(self.df["ID"]))), set(self.df["ID"])):
            data = self.df.loc[self.df['ID'] == subj]

            # Counts vs. Speed
            plt.subplot(2, 2, 1)
            plt.scatter(data["Counts"], data["TrueSpeed"], s=25, edgecolors='black', c='white')

            # Bland-Altman-esque plot
            plt.subplot(2, 2, 2)
            plt.scatter(data["TrueSpeed"], data["Difference"], s=25,
                        edgecolors='black', c=colours[loop_num % 4], marker=markers[loop_num % 5])

        # Makes plot 1 pretty
        plt.subplot(2, 2, 1)
        plt.title("All Data (n={}, r2 = {})".format(len(set(self.df["ID"])), round(counts_speed_r[0]**2, 3)))
        plt.ylabel("Speed (m/s)")
        plt.xlabel("Counts (15-s epoch)")

        # Makes plot 2 pretty
        plt.subplot(2, 2, 2)

        plt.plot(np.arange(.25, 2.5, .25), [ci_width for i in range(len(np.arange(.25, 2.5, .25)))],
                 linestyle='dashed', color='black')
        plt.plot(np.arange(.25, 2.5, .25), [-ci_width for i in range(len(np.arange(.25, 2.5, .25)))],
                 linestyle='dashed', color='black')

        plt.xticks(np.arange(.25, 2.5, .25))

        plt.title("Difference Score (r = {}, p = {})".format(round(true_diff_r[0], 3), round(true_diff_r[1], 3)))
        plt.xlabel("True Speed (m/s)")
        plt.ylabel("True - Predicted Speed (m/s)")

        # PLOT 3: STANDARD ERRORS -------------------------------------------------------------------------------------
        plt.subplot(2, 2, 3)
        plt.title("Individual SEE")
        plt.plot(np.arange(1, 22), self.df_stats.sort_values("SEE")["SEE"], marker="o", color='black',
                 markeredgecolor='black', linestyle="", markersize=4)
        plt.xticks(np.arange(1, 22, 1), fontsize=8)
        plt.ylabel("SEE (m/s)")
        plt.xlabel("Participants")

        # PLOT 4: r2 VALUES -------------------------------------------------------------------------------------------
        plt.subplot(2, 2, 4)
        plt.title("Individual r2")
        plt.plot(np.arange(1, 22), self.df_stats.sort_values("SEE")["r2"], marker="o", color='black', markersize=4,
                 markeredgecolor='black', linestyle="")
        plt.xticks(np.arange(1, 22, 1), fontsize=8)
        plt.ylim(0, 1.05)
        plt.ylabel("r2")
        plt.xlabel("Participants")

    def bland_altman(self, error_bars="SD"):

        # Grpah icons
        icons = []
        for c in ["red", 'dodgerblue', 'black', 'white']:
            for s in ['o', '^', 'P', 's', 'D', "*"]:
                icons.append(c + " " + s)

        true_diff_r = scipy.stats.pearsonr(self.df["TrueSpeed"], self.df["Difference"])

        print("True speed vs. difference score: r = {}, p = {}".format(round(true_diff_r[0], 3),
                                                                       round(true_diff_r[1], 3)))

        if error_bars == "SEE":
            # Calculates 95%CI of individual SEEs as t_crit * SEM
            t_crit = scipy.stats.t.ppf(.95, len(set(self.df["ID"])) - 1)

            see_sem = self.df_stats["SEE"].describe().iloc[2] / (self.df_stats.shape[0] ** .5)

            ci_width = t_crit * see_sem

            bar_type = "95% CI of SEE values"

            print("95%CI width = {} m/s".format(round(ci_width, 5)))

        if error_bars == "SD":
            # Calculates limits of agreement as ±1.96 SD of Differences

            ci_width = 1.96 * self.df["Difference"].describe()[2]

            bar_type = "±1.96 SD of Difference Scores"

        # colours = ["black", 'dimgray', 'lightgray', 'white']
        colours = ["red", 'dodgerblue', 'black', 'grey']
        markers = ['o', '^', 'P', 's', 'D', "*"]

        plt.subplots(1, 1, figsize=(10, 6))

        for loop_num, subj in zip(np.arange(0, len(set(self.df["ID"]))), set(self.df["ID"])):
            data = self.df.loc[self.df['ID'] == subj]

            # Bland-Altman-esque plot
            plt.scatter(data["TrueSpeed"], data["Difference"], s=25, label="",
                        edgecolors='black', c=icons[loop_num].split(" ")[0], marker=icons[loop_num].split(" ")[-1])

        plt.plot(np.arange(0, 2.5, .25), [ci_width for i in range(len(np.arange(0, 2.5, .25)))],
                 linestyle='dashed', color='black', label="Upper Limit")

        plt.axhline(y=self.df["Difference"].describe().iloc[1], linestyle='dotted', color='black', label="Bias")

        plt.plot(np.arange(0, 2.5, .25), [-ci_width for i in range(len(np.arange(0, 2.5, .25)))],
                 linestyle='dashed', color='black', label="Lower Limit")

        plt.xticks(np.arange(.25, 2.5, .25))
        plt.xlim(.25, 2.25)

        plt.title("Treadmill Speed vs. Difference Score ({})".format(bar_type))
        plt.xlabel("Treadmill Speed (m/s)")
        plt.ylabel("Treadmill - Predicted Speed (m/s)")
        plt.legend()

    def plot_equations(self):

        for row in range(0, self.df_eq.shape[0]):
            data = self.df_eq.iloc[row, :]

            # speeds = [data["Intercept"] + data["Slope"] * i for i in
            #           np.arange(data["Walk1Counts"], data["Walk5Counts"], 20)]

            speeds = [data["Intercept"] + data["Slope"] * i for i in
                      np.arange(min(self.df["Counts"]), max(self.df["Counts"]), 20)]

            # plt.plot(np.arange(data["Walk1Counts"], data["Walk5Counts"], 20), speeds, color='black')
            plt.plot(np.arange(min(self.df["Counts"]), max(self.df["Counts"]), 20), speeds, color='black')

        plt.xlabel("Counts")
        plt.ylabel("Predicted Speed (m/s)")
        plt.title("Individual Regression Equations")


x = Treadmill(tm_file)


def plot_gait_stuff():

    w_speeds = np.arange(50, 100, 1)
    r_speeds = np.arange(100, 250, 1)

    w_vo2 = [(0.1 * s + 3.5) / 3.5 for s in w_speeds]
    r_vo2 = [(0.2 * s + 3.5) / 3.5 for s in r_speeds]

    plt.axhline(y=0, color='black')
    plt.plot([i / 60 for i in w_speeds], w_vo2, color='blue')
    plt.plot([i / 60 for i in r_speeds], r_vo2, color='red')
    plt.axvline(x=100 / 60, linestyle='dashed', color='red', label="Walk-Run Equation Use")
    plt.fill_betweenx(x1=50 / 60, x2=100 / 60, color='blue', y=(1, max(w_vo2)), alpha=.6,
                      label="Walk Equation Accurate (ACSM)")
    plt.fill_betweenx(x1=135 / 60, x2=4, color='red', y=(max(w_vo2), max(r_vo2)), alpha=.6,
                      label="Run Equation Accurate (ACSM)")

    plt.fill_betweenx(x1=2.16, x2=2.24, color='green', y=(-5, max(r_vo2)), alpha=.4,
                      label="Walk-Run Transition (Hansen 2017)")

    plt.fill_between(x=np.arange(1.666, 2.25, .01), y1=max(w_vo2), y2=max(r_vo2), color='grey', alpha=.6,
                     label="Questionable Zone")

    plt.fill_betweenx(x1=1, x2=1.67, color='orange', y=(-2.5, 0), alpha=.4, label="Normal Walk Speed (Waters 1999)")
    plt.fill_betweenx(x1=0.5, x2=2.1, color='purple', y=(-5, -2.5), alpha=.4, label="Cadence is linear (Latt 2008)")

    plt.legend(loc='lower right')
    plt.xlabel("Speed (m/s)")
    plt.ylabel("METs")
    plt.title("Discussion Things for Thesis")


plot_gait_stuff()
