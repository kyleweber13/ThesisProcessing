import ImportEDF
from random import randint
import random
import matplotlib.pyplot as plt
import ECG
import csv
import pandas
import numpy as np


def qc_check(raw_edf_folder='/Users/kyleweber/Desktop/Data/OND07/EDF/',
             output_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/QualityControl_Output.csv",
             subject_num=None, start=None,
             epoch_len=15, sample_rate=250, write_results=True):
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

    if subject_num is None and start is None:
        file_start, file_end = ImportEDF.check_file(ecg_filepath, print_summary=False)
        file_duration = ((file_end - file_start).days * 86400 + (file_end - file_start).seconds) * 250

        start_index = randint(0, file_duration - epoch_len * sample_rate)
        start_index -= start_index % (sample_rate * epoch_len)

    if start is not None:
        start_index = start

    print("Testing index {}-{} ({}-second window).".format(start_index, start_index + epoch_len * sample_rate,
                                                           epoch_len))

    ecg_object = ECG.ECG(filepath=ecg_filepath, age=0, start_offset=start_index, end_offset=epoch_len*sample_rate,
                         epoch_len=15, load_raw=True, from_processed=False, write_results=False)

    ecg_object.subjectID = subjectID

    plt.ion()
    validity_data = ECG.ECG.plot_random_qc(self=ecg_object, input_index=0).rule_check_dict
    plt.show(block=True)
    plt.ioff()
    plt.close()

    user_entry = input()

    if user_entry == "1":
        user_entry = 1
    else:
        user_entry = 0

    output_data = [subjectID, start_index,
                   validity_data["Valid Period"],
                   validity_data["HR Valid"], validity_data["HR"],
                   validity_data["Max RR Interval Valid"], validity_data["Max RR Interval"],
                   validity_data["RR Ratio Valid"], validity_data["RR Ratio"],
                   validity_data["Voltage Range Valid"], validity_data["Voltage Range"],
                   validity_data["Correlation Valid"], validity_data["Correlation"], user_entry]

    if write_results:
        with open(output_file, "a") as outfile:
            writer = csv.writer(outfile, lineterminator="\n", delimiter=",")
            writer.writerow(output_data)
        print("Result saved.")
    if not write_results:
        print("Result not saved.")

    return ecg_object, output_data, user_entry


def qc_check_repeat(results_file, repeats_file, raw_edf_folder, assessor_initials=None):

    # INPUT DATA - ALREADY PROCESSED ----------------------------------------------------------------------------------
    input_data = pandas.read_excel(io=results_file, header=0, index_col=None, sheet_name="QualityControl_Testing",
                                   usecols=(1, 2))

    # Creates list of format id_index for each row
    input_sections = [str(id) + "_" + str(index) for id, index in zip(input_data["ID"], input_data["Index"])]

    # INPUT DATA FROM SECOND TESTER -----------------------------------------------------------------------------------
    repeated_data = pandas.read_csv(filepath_or_buffer=repeats_file, delimiter=",", usecols=[0, 1, 2, 3])
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

        ecg, data, result = qc_check(raw_edf_folder=raw_edf_folder, output_file=None,
                                     subject_num=int(sect[0:4]), start=int(sect.split("_")[1]), write_results=False)

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
        self.data = pandas.read_excel(io=self.results_file, header=0, index_col=None,
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

        ax3.hist(x=x.rrrcausedfail["RR Ratio"], bins=np.arange(0, 15, 0.5),
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