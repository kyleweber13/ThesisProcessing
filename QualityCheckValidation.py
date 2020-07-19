import ImportEDF
from random import randint
import random
import matplotlib.pyplot as plt
import ECG
import csv
import pandas as pd
import numpy as np
import sklearn.metrics


def qc_check(raw_edf_folder='/Users/kyleweber/Desktop/Data/OND07/EDF/',
             # output_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/QualityControl_Output.csv"
             output_file="/Users/kyleweber/Desktop/ECGNonwear.csv",
             subject_num=None, start=None,
             epoch_len=15, sample_rate=250, write_results=True, show_fft=False):
    """Imports a random segment of a random ECG file, runs the quality check algorithm on it, plots the raw and
       filtered data, and appends the quality check results to a .csv file.
    """

    plt.close()

    print("\n" + "Plotting random section of data from random participant...")

    sub_list = np.arange(3002, 3045)

    if subject_num is None and start is None:
        subjectID = random.choice(np.delete(sub_list, np.argwhere(sub_list == 3038)))
    if subject_num is not None:
        subjectID = subject_num

    ecg_filepath = raw_edf_folder + "OND07_WTL_{}_01_BF.EDF".format(subjectID)

    print("Using file {}".format(ecg_filepath))

    # if subject_num is None and start is None:
    if start is None:
        file_start, file_end, fs = ImportEDF.check_file(ecg_filepath, print_summary=False)
        file_duration = ((file_end - file_start).days * 86400 + (file_end - file_start).seconds) * fs

        start_index = randint(0, file_duration - epoch_len * sample_rate)
        start_index -= start_index % (sample_rate * epoch_len)

    if start is not None:
        start_index = start

    print("Testing index {}-{} ({}-second window).".format(start_index, start_index + epoch_len * sample_rate,
                                                           epoch_len))

    ecg_object = ECG.ECG(filepath=ecg_filepath, age=0, start_offset=start_index, end_offset=epoch_len*sample_rate,
                         epoch_len=15, load_raw=True, load_accel=True, from_processed=False, write_results=False)

    ecg_object.subjectID = subjectID

    plt.ion()
    validity_data = ECG.ECG.plot_random_qc(self=ecg_object, input_index=0).rule_check_dict
    plt.show(block=True)
    plt.ioff()
    plt.close()

    user_entry = input()

    if user_entry == "1":
        user_entry = "Non-wear"
    else:
        user_entry = "Wear"

    output_data = [subjectID, start_index,
                   validity_data["Valid Period"],
                   validity_data["HR Valid"], validity_data["HR"],
                   validity_data["Max RR Interval Valid"], validity_data["Max RR Interval"],
                   validity_data["RR Ratio Valid"], validity_data["RR Ratio"],
                   validity_data["Voltage Range Valid"], validity_data["Voltage Range"],
                   validity_data["Correlation Valid"], validity_data["Correlation"], validity_data["Accel Counts"],
                   validity_data["Accel Flatline"], validity_data["Accel SD"],
                   user_entry]

    if write_results:
        with open(output_file, "a") as outfile:
            writer = csv.writer(outfile, lineterminator="\n", delimiter=",")
            writer.writerow(output_data)
        print("Result saved.")
    if not write_results:
        print("Result not saved.")

    df_fft = None
    if show_fft:
        y = np.fft.fft(ecg_object.raw[:ecg_object.sample_rate * epoch_len])
        f = np.fft.fftfreq(len(y), 1 / ecg_object.sample_rate)

        df_fft = pd.DataFrame(list(zip(f, np.abs(y))), columns=["Freq", "Power"])

        df_data = pd.DataFrame(list(zip(np.arange(0, epoch_len, 1 / ecg_object.sample_rate),
                                        ecg_object.raw[:ecg_object.sample_rate * epoch_len])),
                               columns=["Time (s)", "Voltage"])

        fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 6))
        fig.subplots_adjust(hspace=.25)
        plt.suptitle("ECG FFT ({}-second window)".format(epoch_len))
        ax1.plot(df_data["Time (s)"], df_data["Voltage"], color='red')
        ax2.plot(df_fft["Freq"].loc[df_fft["Freq"] >= 0], df_fft["Power"].loc[df_fft["Freq"] >= 0], color='black')

        ax2.axvline(60, linestyle='dashed', color='orange', label="60Hz", linewidth=1)

        if ecg_object.epoch_validity[0] == 0:
            ax2.axvline(ecg_object.epoch_hr[0] / 60, color='red', linewidth=1,
                        label="HR ({}bpm = {}Hz)".format(ecg_object.epoch_hr[0], round(ecg_object.epoch_hr[0]/60, 1)))

            for i in range(2, 6):
                ax2.axvline(ecg_object.epoch_hr[0] / 60 * i, color='red',
                            linewidth=1, linestyle='dashed', label='HR Harmonic')

        ax2.set_xlim(-1, 65)
        ax2.set_xticks(np.arange(0, 65, 5))
        ax1.set_ylabel("Voltage")
        ax1.set_xlabel("Time (s)")
        ax2.set_ylabel("Power")
        ax2.set_xlabel("Hz")
        ax2.legend()

    return ecg_object, output_data, user_entry, df_fft


def qc_check_repeat(results_file, repeats_file, raw_edf_folder, assessor_initials=None):

    # INPUT DATA - ALREADY PROCESSED ----------------------------------------------------------------------------------
    input_data = pd.read_excel(io=results_file, header=0, index_col=None, sheet_name="QualityControl_Testing",
                                   usecols=(1, 2))

    # Creates list of format id_index for each row
    input_sections = [str(id) + "_" + str(index) for id, index in zip(input_data["ID"], input_data["Index"])]

    # INPUT DATA FROM SECOND TESTER -----------------------------------------------------------------------------------
    repeated_data = pd.read_csv(filepath_or_buffer=repeats_file, delimiter=",", usecols=[0, 1, 2, 3])
    repeat_sections = [str(id) + "_" + str(index) for id, index in zip(repeated_data["ID"], repeated_data["Index"])]

    # Removes repeated values from input_sections
    for sect in repeat_sections:
        if sect in input_sections:
            input_sections.remove(sect)

    loop_tally = 0
    # Runs quality check for non-repeated sections
    for sect in input_sections:
        loop_tally += 1

        if loop_tally > 100:
            print("\n" + "Reached 100 sections. SWITCH!")
            break

        ecg, data, result, fft = qc_check(raw_edf_folder=raw_edf_folder, output_file=None,
                                          subject_num=int(sect[0:4]), start=int(sect.split("_")[1]),
                                          write_results=False)

        # Appends results to output_file
        with open(repeats_file, "a") as outfile:
            writer = csv.writer(outfile, lineterminator="\n", delimiter=",")
            writer.writerow([assessor_initials, sect[0:4], sect.split("_")[1], result])
        print("Result saved.")


"""qc_check_repeat(results_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/QualityControl_Testing_KW.xlsx",
                repeats_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/QualityControl_Testing_Repeats.csv",
                raw_edf_folder='/Users/kyleweber/Desktop/Data/OND07/EDF/',
                assessor_initials="KK")"""


class AnalyzeResults:

    def __init__(self, results_file):

        self.results_file = results_file
        self.data = None

        self.valid_periods = None
        self.invalid_periods = None
        self.rrrcausedfail = None

        self.rrr_dict = None

        self.analyze_results()

    def analyze_results(self):

        # Reads in tracking sheet
        self.data = pd.read_excel(io=self.results_file, header=0, index_col=None,
                                  usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                           12, 13, 14, 15, 17, 19, 20, 21, 22))

        self.valid_periods = self.data.loc[self.data["Valid Period"]]
        self.invalid_periods = self.data.loc[self.data["Valid Period"] == False]

        self.rrrcausedfail = self.data.loc[self.data["RRRCausedFail"]]

        self.rrr_dict = {"Valid Mean": round(self.valid_periods.describe()["RR Ratio"]["mean"], 3),
                         "Valid SD": round(self.valid_periods.describe()["RR Ratio"]["std"], 3),
                         "Invalid Mean": round(self.invalid_periods.describe()["RR Ratio"]["mean"], 3),
                         "Invalid SD":  round(self.invalid_periods.describe()["RR Ratio"]["std"], 3),
                         "RRCF Mean": round(self.rrrcausedfail.describe()["RR Ratio"]["mean"], 3),
                         "RRCF SD": round(self.rrrcausedfail.describe()["RR Ratio"]["std"], 3)}

    def rr_histogram(self):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))

        ax1.set_title("RR Ratio Histogram by Data Validity")

        ax1.hist(x=self.valid_periods["RR Ratio"], bins=np.arange(0, 15, 0.5),
                 weights=np.ones(len(self.valid_periods["RR Ratio"])) * 100 / len(self.valid_periods["RR Ratio"]),
                 color='#32CD32', edgecolor='black', label="Valid (n={})".format(len(self.valid_periods)))
        ax1.axvline(x=2.5, color='red', linestyle='dashed', label="Threshold (2.5)")
        ax1.set_ylabel("Percent")
        ax1.set_xlim(0, 15)
        ax1.legend()

        ax2.hist(x=self.invalid_periods["RR Ratio"], bins=np.arange(0, 15, 0.5),
                 weights=np.ones(len(self.invalid_periods["RR Ratio"])) * 100 / len(self.invalid_periods["RR Ratio"]),
                 color='#FF0000', edgecolor='black', label="Invalid (n={})".format(len(self.invalid_periods)))
        ax2.axvline(x=2.5, color='red', linestyle='dashed', label="Threshold (2.5)")
        ax2.set_ylabel("Percent")
        ax2.set_xlim(0, 15)
        ax2.legend()

        ax3.hist(x=self.rrrcausedfail["RR Ratio"], bins=np.arange(0, 15, 0.5),
                 weights=np.ones(len(self.rrrcausedfail["RR Ratio"])) * 100 / len(self.rrrcausedfail["RR Ratio"]),
                 color='#FF8C00', edgecolor='black', label="RRCF (n={})".format(len(self.rrrcausedfail)))
        ax3.axvline(x=2.5, color='red', linestyle='dashed', label="Threshold (2.5)")
        ax3.set_ylabel("Percent")
        ax3.legend()
        ax3.set_xlim(0, 15)
        ax3.set_xticks(np.arange(0, 15, 1))
        ax3.set_xlabel("Ratio (longest:shortest)")



"""
x = AnalyzeResults("/Users/kyleweber/Desktop/Data/OND07/Tabular Data/QualityControl_Testing.xlsx")

ecg, data, result = qc_check(write_results=True)
"""


def plot_results(cumulative=False):
    import pandas as pd
    df = pd.read_csv("/Users/kyleweber/Desktop/ECGNonwearAlgorithm.csv")

    plt.subplots(2, 2)
    plt.suptitle("n={} (Non-wear={})".format(df.shape[0], df.loc[df["Wear/Nonwear"] == 'Non-wear'].shape[0]))

    plt.subplot(2, 2, 1)
    plt.title("Accel Counts")

    plt.hist(df.loc[(df["Wear/Nonwear"] == "Wear") & (df["Accel Counts"] < 100)]["Accel Counts"],
             bins=np.arange(0, 50, 1), cumulative=cumulative,
             color='dodgerblue', edgecolor='black', label="Wear")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(df.loc[(df["Wear/Nonwear"] == "Non-wear") & (df["Accel Counts"] < 100)]["Accel Counts"],
             bins=np.arange(0, 50, 1), cumulative=cumulative,
             color='red', edgecolor='black', label="Non-wear")
    plt.xlabel("Counts")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Voltage Range")

    plt.hist(df.loc[(df["Wear/Nonwear"] == "Wear") & (df["Accel Counts"] < 100)]["Voltage Range"],
             bins=np.arange(0, 20000, 250), cumulative=cumulative,
             color='dodgerblue', edgecolor='black', label='Wear')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(df.loc[(df["Wear/Nonwear"] == "Non-wear") & (df["Accel Counts"] < 100)]["Voltage Range"],
             bins=np.arange(0, 20000, 250), cumulative=cumulative,
             color='red', edgecolor='black', label='Non-wear')
    plt.xlabel("Voltage")
    plt.legend()


# plot_results(cumulative=False)


class ROCAnalysis:

    def __init__(self, data_file="/Users/kyleweber/Desktop/ECGNonwearAlgorithm.csv", show_plot=False,
                 threshold_method="AUC"):

        self.data_file = data_file
        self.threshold_method = threshold_method

        self.df = self.import_data()

        self.df_roc = self.loop_thresholds(0, 1500, 10)
        self.print_results()

        if show_plot:
            self.plot_roc()

    def import_data(self):
        df = pd.read_csv(self.data_file, usecols=["Voltage Range", "Accel Counts", "Wear/Nonwear"])
        df.columns = ["Volt", "Counts", "GS"]

        return df

    def algorithm(self, volt_thresh, count_thresh):

        print("Testing voltage threshold = {} and count threshold = {}...".format(volt_thresh, count_thresh))
        output = []

        for i in range(self.df.shape[0]):
            data = self.df.iloc[i]

            verdict = "Non-wear" if data["Volt"] <= volt_thresh and data["Counts"] <= count_thresh else "Wear"
            output.append(verdict)

        agreement = [a == gs for a, gs in zip(output, self.df["GS"])].count(True) / len(output) * 100
        agreement = round(agreement, 2)

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for alg, gs in zip(output, self.df["GS"]):
            if alg == "Non-wear" and gs == "Non-wear":
                tp += 1
            if alg == "Non-wear" and gs == "Wear":
                fp += 1
            if alg == "Wear" and gs == "Non-wear":
                fn += 1
            if alg == "Wear" and gs == "Wear":
                tn += 1

        sens = round(tp / (tp + fn), 3)
        spec = round(tn / (tn + fp), 3)

        return agreement, sens, spec, output

    def loop_thresholds(self, min_val=0, max_val=1500, step=10):

        v_list = []
        c_list = []
        agree_list = []
        sens_list = []
        spec_list = []
        auc_list = []

        for v in np.arange(min_val, max_val, step):
            agree, sens, spec, data = self.algorithm(volt_thresh=v, count_thresh=1)

            v_list.append(v)
            c_list.append(1)
            agree_list.append(agree)
            sens_list.append(sens)
            spec_list.append(spec)

            auc = sklearn.metrics.roc_auc_score(y_true=[1 if i == "Wear" else 0 for i in self.df["GS"]],
                                                y_score=[1 if i == "Wear" else 0 for i in data])
            auc_list.append(round(auc, 3))

        output = pd.DataFrame(list(zip(v_list, c_list, agree_list, sens_list, spec_list, auc_list)),
                              columns=["Volt_Thresh", "Count_Thresh", "Accuracy", "Sens", "Spec", "AUC"])
        output["Distance"] = [((1-sens)**2 + (1-spec)**2)**(1/2) for spec, sens in zip(output["Spec"], output["Sens"])]
        output["Sum"] = output["Sens"] + output["Spec"]

        return output

    def print_results(self):

        accuracy_thresh = self.df_roc.loc[self.df_roc["Accuracy"] ==
                                          max(self.df_roc["Accuracy"])].iloc[0].loc["Volt_Thresh"]
        auc_thresh = self.df_roc.loc[self.df_roc["AUC"] == max(self.df_roc["AUC"])].iloc[0].loc["Volt_Thresh"]
        distance_thresh = self.df_roc.loc[self.df_roc["Distance"] ==
                                          min(self.df_roc["Distance"])].iloc[0].loc["Volt_Thresh"]
        sum_thresh = self.df_roc.loc[self.df_roc["Sum"] ==
                                     max(self.df_roc["Sum"])].iloc[0].loc["Volt_Thresh"]

        print("\n--------------------------------- ROC RESULTS ---------------------------------")
        print("-Maximum accuracy: {} μV".format(accuracy_thresh))
        print("-Maximum AUC: {} μV".format(auc_thresh))
        print("-Minimum distance to (0, 1): {} μV".format(distance_thresh))
        print("-Highest Sens + Spec sum: {} μV".format(sum_thresh))

    def plot_roc(self, threshold_method="AUC"):

        plt.plot([1-spec for spec in self.df_roc["Spec"]], self.df_roc["Sens"], color='black', label="ROC line")
        plt.plot(np.arange(0, 1, .05), np.arange(0, 1, .05), linestyle='dashed', color='red', label="Chance")
        plt.xlabel("1 - specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC Analysis: ECG Voltage Threshold for Non-Wear "
                  "Detection (Determinant = {})".format(threshold_method))
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        if self.threshold_method in ["AUC", "Sum", "Accuracy"]:
            threshold = self.df_roc.loc[self.df_roc[threshold_method] ==
                                        max(self.df_roc[threshold_method])].iloc[0]
        if self.threshold_method == "Distance":
            threshold = self.df_roc.loc[self.df_roc[threshold_method] ==
                                        min(self.df_roc[threshold_method])].iloc[0]

        plt.plot(1-threshold["Spec"], threshold["Sens"], color="green", marker="o",
                 label="Sens = {}, Spec = {}, Accuracy = {}%, AUC = {}, Volt = {}".format(threshold["Sens"],
                                                                                          threshold["Spec"],
                                                                                          threshold["Accuracy"],
                                                                                          threshold["AUC"],
                                                                                          threshold["Volt_Thresh"]))
        plt.legend()


roc = ROCAnalysis(show_plot=True, threshold_method="AUC")
#ecg, output, user, fft = qc_check(write_results=True, show_fft=True, subject_num=None, start=None)