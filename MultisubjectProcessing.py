import LocateUsableParticipants
from Subject import Subject
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from matplotlib import pyplot as plt
import numpy as np
import scipy

usable_subjs = LocateUsableParticipants.find_usable(check_file="/Users/kyleweber/Desktop/Data/OND07/"
                                                               "Tabular Data/OND07_ProcessingStatus.xlsx",
                                                    require_ecg=False, require_wrist=False, require_ankle=True,
                                                    require_all=False, require_ecg_and_one_accel=False,
                                                    require_ecg_and_ankle=False)

usable_subjs.remove("3035")
usable_subjs.remove("3025")


def loop_subjects_standalone(subj_list):
    diff_list = []
    mean_abs_diff_list = []

    for subj in subj_list:
        try:
            x = Subject(
                # What data to load in
                subjectID=int(subj),
                load_ecg=True, load_ankle=True, load_wrist=False,
                load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                from_processed=True,

                # Model parameters
                rest_hr_window=30,
                n_epochs_rest_hr=30,
                hracc_threshold=25,
                filter_ecg=True,
                epoch_len=15,

                # Data files
                raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
                crop_index_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/CropIndexes_All.csv",
                treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",
                demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",
                sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",
                output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
                processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",
                write_results=False)

        except:
            usable_subjs.remove(str(subj))
            pass

        ind_data, group_data, difference_list, rms_diff, mean_abs_diff = x.valid_all.calculate_regression_diff()

        for d in difference_list:
            diff_list.append(d)
        mean_abs_diff_list.append(mean_abs_diff)

    return diff_list, mean_abs_diff_list


def histogram_ind_vs_group_speed_differences():
    """Generates a histogram that shows the difference in predicted gait speed between individual and group regression
       equations for all specified participants with 95%CI shaded.
       Also plots histogram of each participant's average absolute difference.
    """

    def loop_subjects(subj_list):
        diff_list = []
        mean_abs_diff_list = []

        for subj in subj_list:
            try:
                x = Subject(
                    # What data to load in
                    subjectID=int(subj),
                    load_ecg=True, load_ankle=True, load_wrist=False,
                    load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                    from_processed=True,

                    # Model parameters
                    rest_hr_window=30,
                    n_epochs_rest_hr=30,
                    hracc_threshold=30,
                    filter_ecg=True,
                    epoch_len=15,

                    # Data files
                    raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
                    crop_index_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/CropIndexes_All.csv",
                    treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",
                    demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",
                    sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",
                    output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
                    processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",
                    write_results=False)

            except:
                usable_subjs.remove(str(subj))
                pass

            ind_data, group_data, difference_list, rms_diff, mean_abs_diff = x.valid_all.calculate_regression_diff()

            for d in difference_list:
                diff_list.append(d)
            mean_abs_diff_list.append(mean_abs_diff)

        return diff_list, mean_abs_diff_list

    diff, mean_diff = loop_subjects(usable_subjs)

    # Calculates difference's 95%CI
    diff_sd = np.std(diff)
    t_crit = scipy.stats.t.ppf(0.95, len(diff)-1)
    ci_width = diff_sd * t_crit

    plt.subplots(1, 2, figsize=(10, 7))
    plt.suptitle("Predicted Gait Speed Comparison: Waking Hours Above Meaningful Threshold")

    # Histogram: epoch-by-epoch difference during waking/active hours for all participants
    plt.subplot(1, 2, 1)
    plt.fill_between(x=[np.mean(diff) - ci_width, np.mean(diff) + ci_width], y1=0, y2=100,
                     color='#1576DC', alpha=0.35,
                     label="95% CI ({} to {})".format(round(np.mean(diff) - ci_width, 3),
                                                      round(np.mean(diff) + ci_width, 3)))
    plt.ylim(0, max(plt.hist(diff, bins=np.arange(min(diff), max(diff), 0.05),
                             density=True, color='#1576DC', edgecolor='black', cumulative=False)[0])*1.1)
    plt.xlabel("Difference (m/s)")
    plt.ylabel("Percent of All Data")
    plt.legend()
    plt.title("Predicted Gait Speed (Individual - Group)")

    plt.subplot(1, 2, 2)
    plt.hist(mean_diff, bins=np.arange(0, 1, 0.025), color="#B81313", edgecolor='black')
    plt.xlim(0, max(mean_diff)*1.1)
    plt.ylabel("Number of participants")
    plt.xlabel("Difference (m/s)")
    plt.title("Mean Absolute Difference by Participant")


def cohenskappa_ind_vs_group():
    """Function that performs epoch-by-epoch intensity classification analysis on waking/active periods for all
       designated participants. Plots each participant's Cohen's Kappa value."""

    def loop_subjects(subj_list):
        kappa_list = []

        for subj in subj_list:
            try:
                x = Subject(
                    # What data to load in
                    subjectID=int(subj),
                    load_ecg=True, load_ankle=True, load_wrist=False,
                    load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                    from_processed=True,

                    # Model parameters
                    rest_hr_window=30,
                    n_epochs_rest_hr=30,
                    hracc_threshold=30,
                    filter_ecg=True,
                    epoch_len=15,

                    # Data files
                    raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
                    crop_index_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/CropIndexes_All.csv",
                    treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",
                    demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",
                    sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",
                    output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
                    processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",
                    write_results=False)

                kappa_list.append(x.stats.regression_kappa_all)

            except:
                usable_subjs.remove(str(subj))
                pass

        return kappa_list

    # Runs loop function
    kappas = loop_subjects(subj_list=usable_subjs)

    df = pd.DataFrame(zip(usable_subjs, kappas), columns=["Subj", "Kappa"])

    plt.scatter(df.sort_values(by="Kappa")["Subj"], df.sort_values(by="Kappa")["Kappa"], c='red')
    plt.axhline(y=df.describe()["Kappa"]["mean"],
                linestyle='dashed', color='black', label="Mean = {}".format(round(df.describe()["Kappa"]["mean"], 3)))
    plt.legend()
    plt.ylabel("Cohen's Kappa")
    plt.xlabel("Subject ID")
    plt.xticks(rotation=45)
    plt.title("Waking, Active Intensity Agreement")


class AnovaComparisonRegressionActivityMinutes:
    """Class that analyzes differences in total activity minutes calculated from individual and group regression
       equations. Uses values from all specified subjects. Performs one-way repeated measures ANOVA followed by
       Tukey post-hoc tests."""

    def __init__(self, subj_list=None):

        self.subj_list = subj_list
        self.ind_minutes = []
        self.group_minutes = []
        self.df = None
        self.df_long = None
        self.aov = None
        self.aov_results = None
        self.tukey = None

        """RUNS METHODS"""
        self.import_data()
        self.shape_data()
        self.run_anova()
        self.plot_means()

    def import_data(self):

        self.ind_minutes, self.group_minutes = self.loop_subjects()

    def loop_subjects(self):

        ind_minutes = []
        group_minutes = []

        for subj in self.subj_list:
            try:
                x = Subject(
                    # What data to load in
                    subjectID=int(subj),
                    load_ecg=True, load_ankle=True, load_wrist=False,
                    load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                    from_processed=True,

                    # Model parameters
                    rest_hr_window=30,
                    n_epochs_rest_hr=30,
                    hracc_threshold=25,
                    filter_ecg=True,
                    epoch_len=15,

                    # Data files
                    raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
                    crop_index_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/CropIndexes_All.csv",
                    treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",
                    demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",
                    sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",
                    output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
                    processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",
                    write_results=False)

                ind_minutes.append([value for value in x.valid_all.ankle_totals.values()][3::2])
                group_minutes.append([value for value in x.valid_all.ankle_totals_group.values()][3::2])

            except:
                usable_subjs.remove(str(subj))
                pass

        return ind_minutes, group_minutes

    def shape_data(self):

        # Shaping dataframes
        ind = np.array(self.ind_minutes)
        group = np.array(self.group_minutes)
        combined = np.concatenate((ind, group), axis=1)

        self.df = pd.DataFrame(combined, columns=["IndLight", "IndModerate", "IndVigorous",
                                                  "GroupLight", "GroupModerate", "GroupVigorous"])
        self.df.insert(loc=0, column="ID", value=usable_subjs[:self.df.shape[0]])

        self.df_long = pd.melt(frame=self.df, id_vars="ID", var_name="Group", value_name="Minutes")

    def run_anova(self):

        self.aov = AnovaRM(self.df_long, depvar="Minutes", subject="ID", within=["Group"])
        self.aov_results = self.aov.fit()

        print("\n" + "======================================== MAIN EFFECTS ========================================")
        print("\n", self.aov_results.anova_table)

        self.tukey = "n.s."

        if self.aov_results.anova_table["Pr > F"][0] <= 0.05:
            print("")
            tukey_data = MultiComparison(self.df_long["Minutes"], self.df_long["Group"])
            self.tukey = tukey_data.tukeyhsd(alpha=0.05)
            print("============================================ POST HOC ===========================================")
            print(self.tukey.summary())

    def plot_means(self):

        means = [self.df["IndLight"].describe()['mean'], self.df["GroupLight"].describe()['mean'],
                 self.df["IndModerate"].describe()['mean'], self.df["GroupModerate"].describe()['mean'],
                 self.df["IndVigorous"].describe()['mean'], self.df["GroupVigorous"].describe()['mean']]

        sd = [self.df["IndLight"].describe()['std'], self.df["GroupLight"].describe()['std'],
              self.df["IndModerate"].describe()['std'], self.df["GroupModerate"].describe()['std'],
              self.df["IndVigorous"].describe()['std'], self.df["GroupVigorous"].describe()['std']]

        plt.bar(["IndLight", "GroupLight", "IndMod", "GroupMod", "IndVig", "GroupVig"], means,
                color=('green', 'green', 'orange', 'orange', 'red', 'red'), edgecolor='black')

        # Standard error of the means error bars
        plt.errorbar(["IndLight", "GroupLight", "IndMod", "GroupMod", "IndVig", "GroupVig"], means,
                     [i/(len(self.df)**(1/2)) for i in sd],
                     linestyle="", color='black', capsize=4, capthick=1, barsabove=True)

        plt.ylabel("Minutes")
        plt.title("Means ± SEM")


data = AnovaComparisonRegressionActivityMinutes(subj_list=usable_subjs)
data.plot_means()