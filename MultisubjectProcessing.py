import LocateUsableParticipants
from Subject import Subject
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from matplotlib import pyplot as plt
import numpy as np
import scipy
import os

usable_subjs = LocateUsableParticipants.SubjectSubset(check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/"
                                                                 "OND07_ProcessingStatus.xlsx",
                                                      wrist_ankle=False, wrist_hr=False,
                                                      wrist_hracc=False, hr_hracc=False,
                                                      ankle_hr=False, ankle_hracc=False,
                                                      wrist_only=False, ankle_only=False,
                                                      hr_only=False, hracc_only=False,
                                                      require_treadmill=True, require_all=True)


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

                if x.demographics["Height"] < 125 or x.demographics["Weight"] < 40:
                    usable_subjs.remove(str(subj))
                    break

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
            print("\n", self.tukey.summary())

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


class RelativeActivityEffect:

    def __init(self):
        """Statistical analysis between relatively active and inactive groups for between-model differences in
           activity minutes.
        """

        pass


class AverageAnkleCountsStratify:

    def __init__(self, subj_list=None, check_file=None, n_groups=4):
        """Calculates average ankle counts during waking/worn hours and during waking/worn/valid ECG hours.
           Purpose: to determine if these two ways of calculating average counts are the same. Affects how
           participants are stratified into relative activity level groups.
        """

        self.check_file = check_file
        self.avg_counts_accels = []
        self.avg_counts_validecg = []
        self.ids = []
        self.subj_list = subj_list
        self.n_groups = n_groups
        self.n_per_group = 1

        self.df = None
        self.q1_subjs = None
        self.q4_subjs = None

        self.r = None
        self.ttest_counts = None
        self.ttest_age = None
        self.ttest_height = None
        self.ttest_weight = None
        self.wilcoxon_counts = None
        self.wilcoxon_age = None
        self.wilcoxon_weight = None
        self.wilcoxon_height = None
        self.wilcoxon_sex = None

        if os.path.exists(self.check_file):
            self.import_data()

        if not os.path.exists(self.check_file) or self.check_file is None:
            self.loop_participants(subj_list=subj_list)

        self.create_groups(n_groups=self.n_groups)
        self.calculate_stats()

    def import_data(self):
        """Imports data from Excel sheet."""

        self.df = pd.read_excel(io=self.check_file,
                                columns=["Ankle Valid Counts", "Wrist Valid Counts", "Age", "Sex", "Weight", "Height"])

        self.df.dropna(inplace=True)

        self.df = self.df.loc[self.df["ID"].isin(self.subj_list)]

    def loop_participants(self, subj_list):
        """Loops through all participants and calculates average ankle accelerometer counts from
           waking hours and waking hours with valid ECG.
        """

        for subj in subj_list:

            try:

                x = Subject(
                        # What data to load in
                        subjectID=subj,
                        load_ecg=False, load_ankle=True, load_wrist=True,
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

                z = Subject(
                        # What data to load in
                        subjectID=subj,
                        load_ecg=True, load_ankle=True, load_wrist=True,
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

                self.avg_counts_accels.append(x.valid_accelonly.avg_ankle_counts)
                self.avg_counts_validecg.append(z.valid_all.avg_ankle_counts)

                self.ids.append(x.subjectID)

            except:
                pass

            self.df = pd.DataFrame(list(zip(self.ids, self.avg_counts_accels, self.avg_counts_validecg)),
                                   columns=["ID", "All", "Valid ECG"])

    def create_groups(self, n_groups):

        self.n_per_group = int((self.df.shape[0] - self.df.shape[0] % n_groups) / n_groups)

        sorted_by_anklecounts = self.df.sort_values(["Ankle Valid Counts"])

        self.q1_subjs = sorted_by_anklecounts.iloc[0:self.n_per_group]
        self.q4_subjs = sorted_by_anklecounts.iloc[-self.n_per_group:]

        self.df = self.df.sort_values(["Ankle Valid Counts"])
        self.df["Group"] = ["Low" for i in range(self.n_per_group)] + ["High" for i in range(self.n_per_group)]

    def calculate_stats(self):
        """Pearson correlation between ankle and wrist counts, independent samples T-test, Wilcoxon signed rank test."""

        # COUNTS COMPARISON ------------------------------------------------------------------------------------------
        self.r = scipy.stats.pearsonr(self.df["Ankle Valid Counts"], self.df["Wrist Valid Counts"])
        print("Correlation (ankle ~ wrist counts): r = {}, p = {}".format(round(self.r[0], 3), round(self.r[1], 3)))

        # BETWEEN-GROUP COMPARISONS ----------------------------------------------------------------------------------
        print("\n------------------------ Comparison between {} groups created using ankle counts"
              "------------------------ ".format(self.n_groups))

        # Ankle counts
        self.ttest_counts = scipy.stats.ttest_ind(self.q1_subjs["Ankle Valid Counts"],
                                                  self.q4_subjs["Ankle Valid Counts"])
        print("Independent T-tests:")
        print("-Counts: t = {}, p = {}".format(round(self.ttest_counts[0], 3), round(self.ttest_counts[1], 3)))

        # Consider one-sided
        self.wilcoxon_counts = scipy.stats.wilcoxon(self.q1_subjs["Ankle Valid Counts"],
                                                    self.q4_subjs["Ankle Valid Counts"])

        # Age
        self.ttest_age = scipy.stats.ttest_ind(self.q1_subjs["Age"], self.q4_subjs["Age"])
        print("-Age:    t = {}, p = {}".format(round(self.ttest_age[0], 3), round(self.ttest_age[1], 3)))

        # Weight
        self.ttest_weight = scipy.stats.ttest_ind(self.q1_subjs["Weight"], self.q4_subjs["Weight"])
        print("-Weight: t = {}, p = {}".format(round(self.ttest_weight[0], 3), round(self.ttest_weight[1], 3)))

        # Height
        self.ttest_height = scipy.stats.ttest_ind(self.q1_subjs["Height"], self.q4_subjs["Height"])
        print("-Height: t = {}, p = {}".format(round(self.ttest_height[0], 3), round(self.ttest_weight[1], 3)))

        # Sex
        group1_n_females = [i for i in self.q1_subjs["Sex"].values].count(1)
        group2_n_females = [i for i in self.q4_subjs["Sex"].values].count(1)

        print("\n-Females per group:")
        print("     -Low activity: {}".format(group1_n_females))
        print("     -High activity: {}".format(group2_n_females))

    def show_boxplots(self):

        plt.subplots(2, 2, figsize=(10, 7))

        plt.subplot(2, 2, 1)
        plt.boxplot(x=[self.q1_subjs["Ankle Valid Counts"], self.q4_subjs["Ankle Valid Counts"]],
                    labels=["Low activity", "High activity"])
        plt.ylabel("Average Ankle Counts")
        plt.title("Ankle Counts")

        plt.subplot(2, 2, 2)
        plt.boxplot(x=[self.q1_subjs["Age"], self.q4_subjs["Age"]], labels=["Low activity", "High activity"])
        plt.ylabel("Age (years)")
        plt.title("Age")

        plt.subplot(2, 2, 3)
        plt.boxplot(x=[self.q1_subjs["Height"], self.q4_subjs["Height"]], labels=["Low activity", "High activity"])
        plt.ylabel("Height (cm)")
        plt.title("Height")

        plt.subplot(2, 2, 4)
        plt.boxplot(x=[self.q1_subjs["Weight"], self.q4_subjs["Weight"]], labels=["Low activity", "High activity"])
        plt.ylabel("Weight (kg)")
        plt.title("Weight")


z = AverageAnkleCountsStratify(subj_list=usable_subjs.participant_list,
                               check_file="/Users/kyleweber/Desktop/Data/OND07/Processed Data/"
                                          "ECGValidity_AccelCounts_All.xlsx",
                               n_groups=2)


class AverageAnkleCountsValidInvalid:

    def __init__(self, subj_list=None):
        """Calculates average counts during wear periods for both when ECG was valid and invalid.
           Purpose: to determine if movement influences the validity of ECG data.
        """

        self.valid_counts = []
        self.invalid_counts = []
        self.ids = []
        self.passed_ids = []
        self.subj_list = subj_list

        self.df = None
        self.ttest = None
        self.differences = None
        self.valid_higher = 0
        self.invalid_higher = 0
        self.no_diff = 0

        self.loop_participants(subj_list=subj_list)
        self.calculate_stats()
        self.plot_results()

    def loop_participants(self, subj_list):

        for subj in subj_list:

            try:

                z = Subject(
                        # What data to load in
                        subjectID=subj,
                        load_ecg=True, load_ankle=False, load_wrist=True,
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

                index_len = min([len(z.wrist.epoch.svm), len(z.ecg.epoch_validity), len(z.nonwear.status)])

                self.invalid_counts.append(np.asarray([z.wrist.epoch.svm[i] for i in range(index_len) if
                                                       z.ecg.epoch_validity[i] == 1 and
                                                       z.nonwear.status[i] == 0]).mean())
                self.valid_counts.append(np.asarray([z.wrist.epoch.svm[i] for i in range(index_len) if
                                                     z.ecg.epoch_validity[i] == 0 and
                                                     z.nonwear.status[i] == 0]).mean())

                self.ids.append(subj)

            except:
                self.passed_ids.append(subj)
                pass

        self.df = pd.DataFrame(list(zip(self.ids, self.invalid_counts, self.valid_counts)),
                               columns=["ID", "InvalidECG", "ValidECG"])

    def calculate_stats(self):

        self.ttest = scipy.stats.ttest_rel(self.df["InvalidECG"], self.df["ValidECG"])
        print("Paired T-Test: t = {}, p = {}".format(round(self.ttest[0], 3), round(self.ttest[1], 3)))

        self.differences = [invalid - valid for invalid, valid in zip(self.invalid_counts, self.valid_counts)]

        for diff in self.differences:
            if diff > 0:
                self.invalid_higher += 1
            if diff < 0:
                self.valid_higher += 1
            if diff == 0:
                self.no_diff += 1

    def plot_results(self):

        plt.title("Average Counts During Valid/Invalid ECG (p = {})".format(round(self.ttest[1], 3)))

        plt.scatter(self.df["ValidECG"], self.df["InvalidECG"], c='black', label="Data")
        plt.xlabel("Valid ECG")
        plt.ylabel("Invalid ECG")
        plt.plot(np.arange(0, 140), np.arange(0, 140), color='black', linestyle='dashed', label="y=x")
        plt.xlim(0, 140)
        plt.ylim(0, 140)

        plt.fill_between(x=[i for i in range(0, 140)],
                         y1=[i for i in range(0, 140)], y2=140,
                         color='red', alpha=0.25,
                         label="Higher during invalid ECG (n={})".format(self.invalid_higher))

        plt.fill_between(x=[i for i in range(0, 140)],
                         y1=0, y2=[i for i in range(0, 140)],
                         color='green', alpha=0.25,
                         label="Higher during valid ECG (n={})".format(self.valid_higher))
        plt.legend()


# y = AverageAnkleCountsValidInvalid(subj_list=np.arange(3002, 3045))
