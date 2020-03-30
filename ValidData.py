from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import csv
import statistics as stats
import scipy.stats
from datetime import timedelta


class AllDevices:

    def __init__(self, subject_object=None, write_results=False):
        """Generates a class instance that creates and stores data where all devices/models generated valid data.
           Removes periods of invalid ECG data, sleep, and non-wear (NOT CURRENTLY IMPLEMENTED)"""

        print()
        print("====================================== REMOVING INVALID DATA ========================================")
        print("\n" + "Using ECG and sleep data to find valid epochs...")

        self.subject_object = subject_object
        self.write_results = write_results

        # Data sets ---------------------------------------------------------------------------------------------------
        self.data_len = 1
        self.epoch_timestamps = None

        # Data from other objects; invalid data not removed
        self.ankle_intensity = None
        self.ankle_intensity_group = None
        self.wrist_intensity = None
        self.hr_intensity = None
        self.hracc_intensity = None

        # Data used to determine what epochs are valid
        self.hr_validity = None
        self.sleep_validity = None

        # Data that only contains valid epochs
        self.ankle = None
        self.ankle_group = None
        self.wrist = None
        self.hr = None
        self.hr_acc = None

        # Dictionaries for activity totals using valid data
        self.ankle_totals = None
        self.ankle_totals_group = None
        self.wrist_totals = None
        self.hr_totals = None
        self.hracc_totals = None
        self.ankle_hracc_comparison = None
        self.hr_hracc_comparison = None

        self.percent_valid = None
        self.hours_valid = None
        self.final_epoch_validity = None
        self.validity_dict = None

        self.hr_validity_counts = None

        # =============================================== RUNS METHODS ================================================

        # Organizing data depending on what data are available
        self.organize_data()

        # Data used to determine which epochs are valid ---------------------------------------------------------------
        # Removal based on HR data
        if self.hr_validity is not None:
            self.remove_invalid_hr()

        # Removal based on sleep data
        if self.subject_object.sleeplog_file is not None:
            self.remove_invalid_sleep()

        self.recalculate_activity_totals()

        self.generate_validity_report()

        self.check_ecgvalidity_activitylevel()

        self.calculate_ankle_hracc_diff()
        self.calculate_hr_hracc_diff()

        if self.write_results:
            self.write_activity_totals()
            self.write_validity_report()
            self.write_valid_epochs()

    def organize_data(self):

        # Uses timestamps from an available device (should all be equivalent)
        data_len = []
        try:
            data_len.append(len(self.subject_object.ankle.epoch.svm))
        except AttributeError:
            data_len.append(None)
        try:
            data_len.append(len(self.subject_object.wrist.epoch.svm))
        except AttributeError:
            data_len.append(None)
        try:
            data_len.append(len(self.subject_object.ecg.epoch_timestamps))
        except AttributeError:
            data_len.append(None)

        # Sets shortest data set length to data_len
        self.data_len = min([i for i in data_len if i is not None])

        try:
            self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps
        except AttributeError:
            try:
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except AttributeError:
                self.epoch_timestamps = self.subject_object.ecg.epoch_timestamps

        # Ankle intensity data
        try:
            if self.subject_object.ankle_filepath is not None:
                self.ankle_intensity = self.subject_object.ankle.model.epoch_intensity if \
                    self.subject_object.ankle.model.epoch_intensity is not None else None

                self.ankle_intensity_group = self.subject_object.ankle.model.epoch_intensity_group if \
                    self.subject_object.ankle.model.epoch_intensity_group is not None else None

        # Error handling if ankle model not run (no treadmill protocol?)
        except AttributeError:
            self.ankle_intensity = None
            self.ankle_intensity_group = None

        # Wrist intensity data
        if self.subject_object.wrist_filepath is not None:
            self.wrist_intensity = self.subject_object.wrist.model.epoch_intensity if \
                self.subject_object.wrist.model.epoch_intensity is not None else None

        # HR intensity data
        if self.subject_object.ecg_filepath is not None:
            self.hr_intensity = self.subject_object.ecg.epoch_intensity if \
                self.subject_object.ecg.epoch_intensity is not None else None

            # HR validity status
            self.hr_validity = self.subject_object.ecg.epoch_validity if \
                self.subject_object.ecg.epoch_validity is not None else None

        # HR-Acc intensity data
        if self.subject_object.ecg_filepath is not None and self.subject_object.ankle_filepath is not None:
            self.hracc_intensity = self.subject_object.hr_acc.model.epoch_intensity if \
                self.subject_object.hr_acc is not None else None

        # Sleep validity data
        if self.subject_object.sleeplog_file is not None:
            self.sleep_validity = self.subject_object.sleep.status if \
                self.subject_object.sleep.status is not None else None

    def remove_invalid_hr(self):
        """Removes invalid epochs from Ankle and Wrist data based on HR validity."""

        print("\n" + "Removing invalid HR epochs...")

        self.hr = self.hr_intensity

        if self.ankle_intensity is not None:
            self.ankle = [self.ankle_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]
            self.ankle_group = [self.ankle_intensity_group[i] if self.hr_validity[i] == 0
                                else None for i in range(self.data_len)]

        if self.wrist_intensity is not None:
            self.wrist = [self.wrist_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]

        if self.hracc_intensity is not None:
            self.hr_acc = [self.hracc_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]

        print("Complete.")

    def remove_invalid_sleep(self):
        """Removes invalid epochs from Ankle, Wrist, and HR data based on sleep validity.
           If invalid data has been removed using HR data, this method further removes invalid periods due to sleep.
           If invalid data has not been removed using HR data, this method removes invalid periods due to sleep from
           the "raw" data.
        """

        print("\n" + "Removing epochs during sleep...")

        # Ankle -------------------------------------------------------------------------------------------------------
        # If ankle data available and invalid data was not removed using HR
        if self.ankle_intensity is not None and self.ankle is None:
            self.ankle = [self.ankle_intensity[i] if self.sleep_validity[i] == 0 else None
                          for i in range(self.data_len)]
            self.ankle_group = [self.ankle_intensity_group[i] if self.sleep_validity[i] == 0 else None
                                for i in range(self.data_len)]

        # If ankle data available and invalid data was removed
        if self.ankle_intensity is not None and self.ankle is not None:
            self.ankle = [self.ankle[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]
            self.ankle_group = [self.ankle_group[i] if self.sleep_validity[i] == 0
                                else None for i in range(self.data_len)]

        # Wrist ------------------------------------------------------------------------------------------------------
        # If wrist data available and invalid data was not removed using HR
        if self.wrist_intensity is not None and self.wrist is None:
            self.wrist = [self.wrist_intensity[i] if self.sleep_validity[i] == 0 else None
                          for i in range(self.data_len)]

        # If wrist data available and invalid data was removed using HR
        if self.wrist_intensity is not None and self.wrist is not None:
            self.wrist = [self.wrist[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        # Heart Rate -------------------------------------------------------------------------------------------------
        if self.hr_intensity is not None and self.hr is None:
            self.hr = [self.hr_intensity[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        if self.hr_intensity is not None and self.hr is not None:
            self.hr = [self.hr[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        # HR-ACC -----------------------------------------------------------------------------------------------------
        if self.hracc_intensity is not None:
            self.hr_acc = [self.hracc_intensity[i] if self.sleep_validity[i] == 0 else None
                           for i in range(self.data_len)]

        print("Complete.")

    def generate_validity_report(self):

        try:
            self.percent_valid = round(100 * (self.data_len - self.ankle.count(None)) / self.data_len, 1)

            self.hours_valid = round((len(self.ankle) * self.subject_object.epoch_len / 3600) *
                                     (self.percent_valid / 100), 2)

            self.final_epoch_validity = ["Invalid" if i is None else "Valid" for i in self.ankle]

        except (TypeError, AttributeError):

            try:
                self.percent_valid = round(100 * (self.data_len - self.wrist.count(None)) / self.data_len, 1)

                self.hours_valid = round((len(self.wrist) * self.subject_object.epoch_len / 3600) *
                                         (self.percent_valid / 100), 2)

                self.final_epoch_validity = ["Invalid" if i is None else "Valid" for i in self.wrist]

            except (TypeError, AttributeError):

                self.percent_valid = round(100 * (self.data_len - self.hr.count(None)) / self.data_len, 1)

                self.hours_valid = round((len(self.hr) * self.subject_object.epoch_len / 3600) *
                                         (self.percent_valid / 100), 2)

                self.final_epoch_validity = ["Invalid" if i is None else "Valid" for i in self.hr]

            self.hours_valid = round(self.final_epoch_validity.count("Valid") * self.subject_object.epoch_len / 3600, 2)

            print("\n" + "Validity check complete. {}% of the original "
                         "data is valid ({} hours).".format(self.percent_valid, self.hours_valid))

        if self.subject_object.sleeplog_file is None:
            self.validity_dict = {"Valid ECG %": 100 - self.subject_object.ecg.quality_report["Percent invalid"],
                                  "ECG Hours Lost": self.subject_object.ecg.quality_report["Hours lost"],
                                  "Sleep %": 0,
                                  "Sleep Hours Lost": 0,
                                  "Total Valid %": self.percent_valid,
                                  "Total Hours Valid": self.hours_valid}

        if self.subject_object.sleeplog_file is not None:
            self.validity_dict = {"Valid ECG %": 100 - self.subject_object.ecg.quality_report["Percent invalid"],
                                  "ECG Hours Lost": self.subject_object.ecg.quality_report["Hours lost"],
                                  "Sleep %": self.subject_object.sleep.report["Sleep%"],
                                  "Sleep Hours Lost": round(self.subject_object.sleep.report["SleepDuration"]
                                                            / 60, 2),
                                  "Total Valid %": self.percent_valid,
                                  "Total Hours Valid": self.hours_valid}

    def check_ecgvalidity_activitylevel(self):

        print("\n" + "Checking accelerometer data to determine if invalid "
                     "ECG periods were during more/less movement...")

        self.hr_validity_counts = {"Wrist valid": None, "Wrist invalid": None,
                                   "Ankle valid": None, "Ankle invalid": None}

        wrist_ttest_t = None
        wrist_ttest_p = None
        ankle_ttest_t = None
        ankle_ttest_p = None

        if self.subject_object.wrist_filepath is not None:

            wrist_valid_data = [self.subject_object.wrist.epoch.svm[i]
                                for i in range(0, self.data_len)
                                if self.hr_validity[i] == 0]

            wrist_invalid_data = [self.subject_object.wrist.epoch.svm[i]
                                  for i in range(0, self.data_len)
                                  if self.hr_validity[i] == 1]

            valid_mean = stats.mean(wrist_valid_data)
            valid_sem = stats.stdev(wrist_valid_data) / (len(wrist_valid_data) ** (1/2))
            invalid_mean = stats.mean(wrist_invalid_data)
            invalid_sem = stats.stdev(wrist_invalid_data) / (len(wrist_invalid_data) ** (1/2))

            self.hr_validity_counts["Wrist valid"] = round(stats.mean(wrist_valid_data), 1)
            self.hr_validity_counts["Wrist valid SEM"] = valid_sem
            self.hr_validity_counts["Wrist invalid"] = round(stats.mean(wrist_invalid_data), 1)
            self.hr_validity_counts["Wrist invalid SEM"] = invalid_sem

            ttest_result = scipy.stats.ttest_ind(wrist_valid_data, wrist_invalid_data)

            wrist_ttest_t = round(ttest_result[0], 2)
            wrist_ttest_p = round(ttest_result[1], 5)

            if wrist_ttest_p < .05:
                print("-Wrist activity may have had a statistically significant effect on ECG validity:")
            if wrist_ttest_p >= .05:
                print("-Wrist activity does not appear to have had a statistically significant effect on ECG validity:")

            print("    - Valid counts = {}; invalid counts = {} "
                  "(t = {}, p ~ {})".format(self.hr_validity_counts["Wrist valid"],
                                            self.hr_validity_counts["Wrist invalid"], wrist_ttest_t, wrist_ttest_p))

        if self.subject_object.ankle_filepath is not None:
            ankle_valid_data = [self.subject_object.ankle.epoch.svm[i]
                                for i in range(0, self.data_len)
                                if self.hr_validity[i] == 0]

            ankle_invalid_data = [self.subject_object.ankle.epoch.svm[i]
                                  for i in range(0, self.data_len)
                                  if self.hr_validity[i] == 1]

            valid_mean = stats.mean(ankle_valid_data)
            valid_sem = stats.stdev(ankle_valid_data) / (len(ankle_valid_data) ** (1/2))
            invalid_mean = stats.mean(ankle_valid_data)
            invalid_sem = stats.stdev(ankle_invalid_data) / (len(ankle_invalid_data) ** (1/2))

            self.hr_validity_counts["Ankle valid"] = round(stats.mean(ankle_valid_data), 1)
            self.hr_validity_counts["Ankle valid SEM"] = valid_sem
            self.hr_validity_counts["Ankle invalid"] = round(stats.mean(ankle_invalid_data), 1)
            self.hr_validity_counts["Ankle invalid SEM"] = invalid_sem

            ttest_result = scipy.stats.ttest_ind(ankle_valid_data, ankle_invalid_data)

            ankle_ttest_t = round(ttest_result[0], 2)
            ankle_ttest_p = round(ttest_result[1], 5)

            if ankle_ttest_p < .05:
                print("-Ankle activity may have had a statistically significant effect on ECG validity:")
            if ankle_ttest_p >= .05:
                print("-Ankle activity does not appear to have had a statistically significant effect on ECG validity:")

            print("    - Valid counts = {}; invalid counts = {} "
                  "(t = {}, p ~ {})".format(self.hr_validity_counts["Ankle valid"],
                                            self.hr_validity_counts["Ankle invalid"], ankle_ttest_t, ankle_ttest_p))

        self.validity_dict["Ankle Valid Counts"] = self.hr_validity_counts["Ankle valid"]
        self.validity_dict["Ankle Invalid Counts"] = self.hr_validity_counts["Ankle invalid"]
        self.validity_dict["Ankle Counts (t)"] = ankle_ttest_t
        self.validity_dict["Ankle Counts (p)"] = ankle_ttest_p
        self.validity_dict["Wrist Valid Counts"] = self.hr_validity_counts["Wrist valid"]
        self.validity_dict["Wrist Invalid Counts"] = self.hr_validity_counts["Wrist invalid"]
        self.validity_dict["Wrist Counts (t)"] = wrist_ttest_t
        self.validity_dict["Wrist Counts (p)"] = wrist_ttest_p

    def plot_validity_comparison(self):

        plt.bar(x=["Valid Wrist", "Invalid Wrist", "Valid Ankle", "Invalid Ankle"],
                height=[self.hr_validity_counts["Wrist valid"], self.hr_validity_counts["Wrist invalid"],
                        self.hr_validity_counts["Ankle valid"], self.hr_validity_counts["Ankle invalid"]],
                color=['blue', 'blue', 'red', 'red'], edgecolor='black', alpha=0.75,
                yerr=[self.hr_validity_counts["Wrist valid SEM"], self.hr_validity_counts["Wrist invalid SEM"],
                      self.hr_validity_counts["Ankle valid SEM"], self.hr_validity_counts["Ankle invalid SEM"]],
                capsize=3)
        plt.ylabel("Counts")
        plt.title("Participant {}: Accelerometer counts based on ECG validity "
                  "(mean ± SEM)".format(self.subject_object.subjectID))

    def plot_validity_data(self):
        """Generates 4 subplots for each activity model with invalid data removed."""

        # x-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 7))

        if self.subject_object.sleeplog_file is None:
            ax1.set_title("Participant {}: Valid Data ({}% valid) (no sleep data)".format(self.subject_object.subjectID,
                                                                                          self.percent_valid))

        # Fills in region where participant was asleep
        if self.subject_object.sleeplog_file is not None:
            ax1.set_title("Participant {}: Valid Data ({}% valid) (green = sleep)".format(self.subject_object.subjectID,
                                                                                          self.percent_valid))
            for day1, day2 in zip(self.subject_object.sleep.data[:], self.subject_object.sleep.data[1:]):
                try:
                    # Overnight sleep
                    ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax3.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax4.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)

                    # Daytime naps
                    ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax3.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax4.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)

                except (AttributeError, TypeError):
                    pass

        if self.ankle_intensity is not None:
            ax1.plot(self.epoch_timestamps[:self.data_len], self.ankle[:self.data_len], color='#606060', label='Ankle')
            ax1.set_ylim(-0.1, 3)
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Intensity Cat.")

        if self.wrist_intensity is not None:
            ax2.plot(self.epoch_timestamps[:self.data_len], self.wrist[:self.data_len], color='#606060', label='Wrist')
            ax2.set_ylim(-0.1, 3)
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Intensity Cat.")

        if self.hr_intensity is not None:
            ax3.plot(self.epoch_timestamps[:self.data_len], self.hr[:self.data_len], color='red', label='HR')
            ax3.set_ylim(-0.1, 3)
            ax3.legend(loc='upper left')
            ax3.set_ylabel("Intensity Cat.")

        if self.hracc_intensity is not None:
            ax4.plot(self.epoch_timestamps[:self.data_len], self.hr_acc[:self.data_len], color='black', label='HR-Acc')
            ax4.set_ylim(-0.1, 3)
            ax4.legend(loc='upper left')
            ax4.set_ylabel("Intensity Cat.")

        ax4.xaxis.set_major_formatter(xfmt)
        ax4.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

        plt.show()

    def recalculate_activity_totals(self):

        # ANKLE -------------------------------------------------------------------------------------------------------
        if self.ankle is not None:
            epoch_to_minutes = 60 / self.subject_object.ankle.epoch_len

            n_valid_epochs = len(self.ankle) - self.ankle.count(None)

            self.ankle_totals = {"Model": "Ankle",
                                 "Sedentary": self.ankle.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.ankle.count(0) / n_valid_epochs, 3),
                                 "Light": (self.ankle.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.ankle.count(1) / n_valid_epochs, 3),
                                 "Moderate": self.ankle.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.ankle.count(2) / n_valid_epochs, 3),
                                 "Vigorous": self.ankle.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.ankle.count(3) / n_valid_epochs, 3)}

            self.ankle_totals_group = {"Model": "AnkleGroup",
                                       "Sedentary": self.ankle_group.count(0) / epoch_to_minutes,
                                       "Sedentary%": round(self.ankle_group.count(0) / n_valid_epochs, 3),
                                       "Light": (self.ankle_group.count(1)) / epoch_to_minutes,
                                       "Light%": round(self.ankle_group.count(1) / n_valid_epochs, 3),
                                       "Moderate": self.ankle_group.count(2) / epoch_to_minutes,
                                       "Moderate%": round(self.ankle_group.count(2) / n_valid_epochs, 3),
                                       "Vigorous": self.ankle_group.count(3) / epoch_to_minutes,
                                       "Vigorous%": round(self.ankle_group.count(3) / n_valid_epochs, 3)}

        if self.ankle is None:
            self.ankle_totals = {"Model": "Ankle",
                                 "Sedentary": 0, "Sedentary%": 0,
                                 "Light": 0, "Light%": 0,
                                 "Moderate": 0, "Moderate%": 0,
                                 "Vigorous": 0, "Vigorous%": 0}

            self.ankle_totals_group = {"Model": "Ankle",
                                       "Sedentary": 0, "Sedentary%": 0,
                                       "Light": 0, "Light%": 0,
                                       "Moderate": 0, "Moderate%": 0,
                                       "Vigorous": 0, "Vigorous%": 0}

        # WRIST -------------------------------------------------------------------------------------------------------
        if self.wrist is not None:
            epoch_to_minutes = 60 / self.subject_object.wrist.epoch_len

            n_valid_epochs = len(self.wrist) - self.wrist.count(None)

            self.wrist_totals = {"Model": "Wrist",
                                 "Sedentary": self.wrist.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.wrist.count(0) / n_valid_epochs, 3),
                                 "Light": (self.wrist.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.wrist.count(1) / n_valid_epochs, 3),
                                 "Moderate": self.wrist.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.wrist.count(2) / n_valid_epochs, 3),
                                 "Vigorous": self.wrist.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.wrist.count(3) / n_valid_epochs, 3)}

            if self.wrist is None:
                self.wrist_totals = {"Model": "Wrist",
                                     "Sedentary": 0, "Sedentary%": 0,
                                     "Light": 0, "Light%": 0,
                                     "Moderate": 0, "Moderate%": 0,
                                     "Vigorous": 0, "Vigorous%": 0}

        # HEART RATE --------------------------------------------------------------------------------------------------
        if self.hr is not None:
            epoch_to_minutes = 60 / self.subject_object.ecg.epoch_len

            n_valid_epochs = len(self.hr) - self.hr.count(None)

            self.hr_totals = {"Model": "HR",
                              "Sedentary": self.hr.count(0) / epoch_to_minutes,
                              "Sedentary%": round(self.hr.count(0) / n_valid_epochs, 3),
                              "Light": (self.hr.count(1)) / epoch_to_minutes,
                              "Light%": round(self.hr.count(1) / n_valid_epochs, 3),
                              "Moderate": self.hr.count(2) / epoch_to_minutes,
                              "Moderate%": round(self.hr.count(2) / n_valid_epochs, 3),
                              "Vigorous": self.hr.count(3) / epoch_to_minutes,
                              "Vigorous%": round(self.hr.count(3) / n_valid_epochs, 3)}

        if self.hr is None:
            self.hr_totals = {"Model": "HR",
                              "Sedentary": 0, "Sedentary%": 0,
                              "Light": 0, "Light%": 0,
                              "Moderate": 0, "Moderate%": 0,
                              "Vigorous": 0, "Vigorous%": 0}

        # HR-ACC ------------------------------------------------------------------------------------------------------
        if self.hr_acc is not None:
            epoch_to_minutes = 60 / self.subject_object.hr_acc.epoch_len

            n_valid_epochs = len(self.hr_acc) - self.hr_acc.count(None)

            self.hracc_totals = {"Model": "HR-Acc",
                                 "Sedentary": self.hr_acc.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.hr_acc.count(0) / n_valid_epochs, 3),
                                 "Light": (self.hr_acc.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.hr_acc.count(1) / n_valid_epochs, 3),
                                 "Moderate": self.hr_acc.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.hr_acc.count(2) / n_valid_epochs, 3),
                                 "Vigorous": self.hr_acc.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.hr_acc.count(3) / n_valid_epochs, 3)}

    def write_activity_totals(self):

        with open("{}Model Output/OND07_WTL_{}_01_Valid_Activity_Totals.csv".format(self.subject_object.output_dir,
                                                                                    self.subject_object.subjectID),
                  'w', newline='') as outfile:

            fieldnames = ['Model', 'Sedentary', 'Sedentary%', 'Light', 'Light%',
                          'Moderate', 'Moderate%', 'Vigorous', 'Vigorous%']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            if self.subject_object.ankle is not None:
                writer.writerow(self.ankle_totals)

            if self.subject_object.wrist is not None:
                writer.writerow(self.wrist_totals)

            if self.subject_object.ecg is not None:
                writer.writerow(self.hr_totals)

            if self.subject_object.ankle is not None and self.subject_object.ecg is not None:
                writer.writerow(self.hracc_totals)

            print()
            print("Saved activity profiles from valid data to file "
                  "{}Model Output/OND07_WTL_{}_01_Valid_Activity_Totals.csv".format(self.subject_object.output_dir,
                                                                                    self.subject_object.subjectID))

    def write_valid_epochs(self):

        with open("{}OND07_WTL_{}_01_Valid_EpochIntensityData.csv".format(self.subject_object.output_dir,
                                                                          self.subject_object.subjectID),
                  'w', newline='') as outfile:

            writer = csv.writer(outfile, delimiter=",", lineterminator="\n")

            writer.writerow(["Timestamp", "Validity", "Wrist", "Ankle", "HR"])

            # Prevents TypeError during writing to .csv if object is None
            if self.wrist_intensity is None:
                wrist_intensity = [None for i in range(len(self.epoch_timestamps))]
            if self.wrist_intensity is not None:
                wrist_intensity = self.wrist

            if self.ankle_intensity is None:
                ankle_intensity = [None for i in range(len(self.epoch_timestamps))]
            if self.ankle_intensity is not None:
                ankle_intensity = self.ankle

            if self.hr_intensity is None:
                hr_intensity = [None for i in range(len(self.epoch_timestamps))]
            if self.hr_intensity is not None:
                hr_intensity = self.hr

            writer.writerows(zip(self.epoch_timestamps, self.final_epoch_validity,
                                 wrist_intensity, ankle_intensity, hr_intensity))

        print("\n" + "Saved epoch-by-epoch intensity data to file "
                     "{}Model Output/OND07_WTL_{}_01_Valid_EpochIntensityData.csv"
              .format(self.subject_object.output_dir, self.subject_object.subjectID))

    def calculate_hr_hracc_diff(self):
        """Calculates difference between HR-Acc and ankle models for each intensity. Positive value indicates
           HR-Acc model measured more activity. Returns dictionary."""

        if self.hr_acc is None:
            return None

        self.hr_hracc_comparison = {"Sedentary": self.hracc_totals["Sedentary"] - self.hr_totals["Sedentary"],
                                    "Sedentary%": round(self.hracc_totals["Sedentary%"] -
                                                        self.hr_totals["Sedentary%"], 5),

                                    "Light": self.hracc_totals["Light"] - self.hr_totals["Light"],
                                    "Light%": round(self.hracc_totals["Light%"] - self.hr_totals["Light%"], 5),

                                    "Moderate": self.hracc_totals["Moderate"] - self.hr_totals["Moderate"],
                                    "Moderate%": round(self.hracc_totals["Moderate%"] - self.hr_totals["Moderate%"], 5),

                                    "Vigorous": self.hracc_totals["Vigorous"] - self.hr_totals["Vigorous"],
                                    "Vigorous%": round(self.hracc_totals["Vigorous%"] - self.hr_totals["Vigorous%"], 5),

                                    "MVPA": self.hracc_totals["Moderate"] + self.hracc_totals["Vigorous"] -
                                            (self.hr_totals["Moderate"] + self.hr_totals["Vigorous"]),
                                    "MVPA%": round((self.hracc_totals["Moderate%"] + self.hracc_totals["Vigorous%"]) -
                                                   (self.hr_totals["Moderate%"] + self.hr_totals["Vigorous%"]), 5)}

    def calculate_ankle_hracc_diff(self):
        """Calculates difference between HR-Acc and HR models for each intensity. Positive value indicates
           HR-Acc model measured more activity. Returns dictionary."""

        if self.hr_acc is None:
            return None

        self.ankle_hracc_comparison = {"Sedentary": self.hracc_totals["Sedentary"] - self.ankle_totals["Sedentary"],
                                       "Sedentary%": round(self.hracc_totals["Sedentary%"] -
                                                           self.ankle_totals["Sedentary%"], 5),

                                       "Light": self.hracc_totals["Light"] - self.ankle_totals["Light"],
                                       "Light%": round(self.hracc_totals["Light%"] - self.ankle_totals["Light%"], 5),

                                       "Moderate": self.hracc_totals["Moderate"] - self.ankle_totals["Moderate"],
                                       "Moderate%": round(self.hracc_totals["Moderate%"] -
                                                          self.ankle_totals["Moderate%"], 5),

                                       "Vigorous": self.hracc_totals["Vigorous"] - self.ankle_totals["Vigorous"],
                                       "Vigorous%": round(self.hracc_totals["Vigorous%"] -
                                                          self.ankle_totals["Vigorous%"], 5),

                                       "MVPA": (self.hracc_totals["Moderate"] + self.hracc_totals["Vigorous"]) -
                                               (self.ankle_totals["Moderate"] + self.ankle_totals["Vigorous"]),
                                       "MVPA%": round((self.hracc_totals["Moderate%"] + self.hracc_totals["Vigorous%"])
                                                      - (self.ankle_totals["Moderate%"] +
                                                         self.ankle_totals["Vigorous%"]), 5)}

    def plot_hracc_comparisons(self):

        def autolabel(rects, valid_time):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()

                plt.annotate('{} mins'.format(round(height / 100 * 60 * valid_time, 1)),
                             xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                             xytext=(0, 0),
                             textcoords="offset points",
                             ha='center', va='center')

        sedentary_minutes = [self.ankle_hracc_comparison["Sedentary"], self.hr_hracc_comparison["Sedentary"]]
        light_minutes = [self.ankle_hracc_comparison["Light"], self.hr_hracc_comparison["Light"]]
        moderate_minutes = [self.ankle_hracc_comparison["Moderate"], self.hr_hracc_comparison["Moderate"]]
        vigorous_minutes = [self.ankle_hracc_comparison["Vigorous"], self.hr_hracc_comparison["Vigorous"]]

        plt.subplots(2, 2, figsize=(10, 7))
        plt.suptitle("Participant {}: Model Comparisons "
                     "(negative value means HR-Acc measured less time)".format(self.subject_object.subjectID))

        # Sedentary activity
        sed_perc = [i / 60 / self.validity_dict["Total Hours Valid"] * 100 for i in sedentary_minutes]

        plt.subplot(2, 2, 1)
        plt.title("Sedentary")
        sed_plot = plt.bar(["HRAcc - Ankle", "HRAcc - HR"], sed_perc, color='grey', edgecolor='black')
        autolabel(sed_plot, self.validity_dict["Total Hours Valid"])
        plt.ylabel("Δ% of valid time")
        plt.axhline(y=0, color='black', linewidth=1)

        # Light activity
        light_perc = [i / 60 / self.validity_dict["Total Hours Valid"] * 100 for i in light_minutes]
        plt.subplot(2, 2, 2)
        plt.title("Light Activity")
        light_plot = plt.bar(["HRAcc - Ankle", "HRAcc - HR"], light_perc, color='green', edgecolor='black')
        autolabel(light_plot, self.validity_dict["Total Hours Valid"])
        plt.ylabel("Δ% of valid time")
        plt.axhline(y=0, color='black', linewidth=1)

        # Moderate activity
        mod_perc = [i / 60 / self.validity_dict["Total Hours Valid"] * 100 for i in moderate_minutes]
        plt.subplot(2, 2, 3)
        plt.title("Moderate Activity")
        mod_plot = plt.bar(["HRAcc - Ankle", "HRAcc - HR"], mod_perc, color='#EA5B19', edgecolor='black')
        autolabel(mod_plot, self.validity_dict["Total Hours Valid"])
        plt.ylabel("Δ% of valid time")
        plt.axhline(y=0, color='black', linewidth=1)

        # Vigorous activity
        vig_perc = [i / 60 / self.validity_dict["Total Hours Valid"] * 100 for i in vigorous_minutes]
        plt.subplot(2, 2, 4)
        plt.title("Vigorous Activity")
        vig_plot = plt.bar(["HRAcc - Ankle", "HRAcc - HR"], vig_perc, color='red', edgecolor='black')
        autolabel(vig_plot, self.validity_dict["Total Hours Valid"])
        plt.ylabel("Δ% of valid time")
        plt.axhline(y=0, color='black', linewidth=1)

    def write_validity_report(self):

        with open(self.subject_object.output_dir + str(self.subject_object.subjectID) +
                  "_ValidityData.csv", "w") as outfile:
            fieldnames = ['Valid ECG %', 'ECG Hours Lost',
                          'Ankle Valid Counts', 'Ankle Invalid Counts',
                          'Ankle Counts (t)', 'Ankle Counts (p)',
                          'Wrist Valid Counts', 'Wrist Invalid Counts',
                          'Wrist Counts (t)', 'Wrist Counts (p)',
                          'Sleep %', 'Sleep Hours Lost',
                          'Total Valid %', "Total Hours Valid"]

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(self.validity_dict)

        print("\n" + "Saved validity summary data to file {}".format(self.subject_object.output_dir) +
              str(self.subject_object.subjectID) + "_ValidityData.csv")

    def calculate_regression_diff(self):
        """Calculates comparative measures between individual and group regression equation data.
           Values are calculated on epochs above sedentary threshold and while participant was awake.
        """

        ind_data = [self.subject_object.ankle.model.linear_speed[i]
                    for i in range(len(self.subject_object.ankle.model.linear_speed))
                    if self.subject_object.sleep.status[i] < 1
                    and (self.subject_object.ankle.model.linear_speed[i] > 0
                    or self.subject_object.ankle.model.linear_speed_group[i] > 0)]

        group_data = [self.subject_object.ankle.model.linear_speed_group[i]
                      for i in range(len(self.subject_object.ankle.model.linear_speed_group))
                      if self.subject_object.sleep.status[i] < 1
                      and (self.subject_object.ankle.model.linear_speed[i] > 0
                           or self.subject_object.ankle.model.linear_speed_group[i] > 0)]

        difference_list = [ind - group for ind, group in zip(ind_data, group_data)]
        mean_value = [(ind + group)/2 for ind, group in zip(ind_data, group_data)]

        rms_diff = np.mean([diff ** 2 for diff in difference_list])**(1/2)
        mean_abs_diff = np.mean([abs(diff) for diff in difference_list])

        return ind_data, group_data, difference_list, rms_diff, mean_abs_diff

    def plot_regression_comparison(self):

        # x-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))

        fig.suptitle("Individual vs. Group Regression Comparison")

        ax1.plot(self.subject_object.ankle.epoch.timestamps, self.ankle, color='black', label="Individual")
        ax1.plot(self.subject_object.ankle.epoch.timestamps, self.ankle_group, color='red', label="Group")
        ax1.set_xlim(self.subject_object.ankle.epoch.timestamps[0] + timedelta(seconds=3600),
                     self.subject_object.ankle.epoch.timestamps[-1] + timedelta(seconds=3600))
        ax1.set_ylabel("Intensity Category")

        ax2.plot(self.subject_object.ankle.epoch.timestamps,
                 [self.subject_object.ankle.model.linear_speed[i] if self.subject_object.sleep.status[i] == 0
                  else None for i in range(len(self.subject_object.ankle.model.linear_speed))],
                 color='black', label="Individual")

        ax2.plot(self.subject_object.ankle.epoch.timestamps,
                 [self.subject_object.ankle.model.linear_speed_group[i] if self.subject_object.sleep.status[i] == 0
                  else None for i in range(len(self.subject_object.ankle.model.linear_speed_group))],
                 color='red', label='Group')

        ax2.set_xlim(self.subject_object.ankle.epoch.timestamps[0] + timedelta(seconds=3600),
                     self.subject_object.ankle.epoch.timestamps[-1] + timedelta(seconds=3600))
        ax2.set_ylabel("Predicted Speed (m/s)")

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

        if self.subject_object.sleeplog_file is not None:

            for day1, day2 in zip(self.subject_object.sleep.data[:], self.subject_object.sleep.data[1:]):
                try:
                    # Overnight sleep
                    ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)

                    # Daytime naps
                    ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)

                except (AttributeError, TypeError):
                    pass

    def plot_regression_totals_comparison(self):

        sedentary_minutes = [self.ankle_totals["Sedentary"], self.ankle_totals_group["Sedentary"]]

        light_minutes = [self.ankle_totals["Light"], self.ankle_totals_group["Light"]]

        moderate_minutes = [self.ankle_totals["Moderate"], self.ankle_totals_group["Moderate"]]

        vigorous_minutes = [self.ankle_totals["Vigorous"], self.ankle_totals_group["Vigorous"]]

        plt.subplots(2, 2, figsize=(10, 7))
        plt.suptitle("Participant {}: Valid Only Data".format(self.subject_object.subjectID))

        # Sedentary activity
        plt.subplot(2, 2, 1)
        plt.title("Sedentary")
        plt.bar(["Individual", "Group"], sedentary_minutes, color='grey', edgecolor='black')
        plt.ylim(0, max(sedentary_minutes) * 1.2)
        plt.ylabel("Minutes")

        # Light activity
        plt.subplot(2, 2, 2)
        plt.title("Light Activity")
        plt.bar(["Individual", "Group"], light_minutes, color='green', edgecolor='black')
        plt.ylim(0, max(light_minutes) * 1.2)

        # Moderate activity
        plt.subplot(2, 2, 3)
        plt.title("Moderate Activity")
        plt.bar(["Individual", "Group"], moderate_minutes, color='#EA5B19', edgecolor='black')
        plt.ylim(0, max(moderate_minutes) * 1.2)
        plt.ylabel("Minutes")

        # Vigorous activity
        plt.subplot(2, 2, 4)
        plt.title("Vigorous Activity")
        plt.bar(["Individual", "Group"], vigorous_minutes, color='red', edgecolor='black')
        plt.ylim(0, max(vigorous_minutes) * 1.2)


class AccelOnly:

    def __init__(self, subject_object=None, write_results=True):
        """Generates a class instance that creates and stores data where all devices/models generated valid data.
           Removes periods of invalid ECG data, sleep, and non-wear (NOT CURRENTLY IMPLEMENTED)"""

        print()
        print("====================================== REMOVING INVALID DATA ========================================")
        print("\n" + "Using only sleep to find valid epochs in accelerometer data...")

        self.subject_object = subject_object
        self.write_results = write_results

        # Data sets ---------------------------------------------------------------------------------------------------
        self.data_len = 1
        self.epoch_timestamps = None

        # Data from other objects; invalid data not removed
        self.ankle_intensity = None
        self.ankle_intensity_group = None
        self.wrist_intensity = None

        # Data used to determine what epochs are valid
        self.sleep_validity = None

        # Data that only contains valid epochs
        self.ankle = None
        self.ankle_group = None
        self.wrist = None

        # Dictionaries for activity totals using valid data
        self.ankle_totals = None
        self.ankle_totals_group = None
        self.wrist_totals = None
        self.hr_totals = None
        self.hracc_totals = None

        self.percent_valid = None
        self.hours_valid = None
        self.final_epoch_validity = None
        self.validity_dict = None

        # =============================================== RUNS METHODS ================================================

        # Organizing data depending on what data are available
        self.organize_data()

        # Data used to determine which epochs are valid ---------------------------------------------------------------
        # Removal based on sleep data
        self.remove_invalid_sleep()

        self.recalculate_activity_totals()

        self.generate_validity_report()

        if self.write_results:
            self.write_activity_totals()
            self.write_validity_report()
            self.write_valid_epochs()

    def organize_data(self):

        # Uses timestamps from an available device (should all be equivalent)
        data_len = []
        try:
            data_len.append(len(self.subject_object.ankle.epoch.svm))
        except AttributeError:
            data_len.append(None)
        try:
            data_len.append(len(self.subject_object.wrist.epoch.svm))
        except AttributeError:
            data_len.append(None)

        # Sets shortest data set length to data_len
        self.data_len = min([i for i in data_len if i is not None])

        try:
            self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps
        except AttributeError:
            self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps

        # Ankle intensity data
        try:
            if self.subject_object.ankle_filepath is not None:
                self.ankle_intensity = self.subject_object.ankle.model.epoch_intensity if \
                    self.subject_object.ankle.model.epoch_intensity is not None else None

                self.ankle_intensity_group = self.subject_object.ankle.model.epoch_intensity_group if \
                    self.subject_object.ankle.model.epoch_intensity_group is not None else None

        # Error handling if ankle model not run (no treadmill protocol?)
        except AttributeError:
            self.ankle_intensity = None
            self.ankle_intensity_group = None

        # Wrist intensity data
        if self.subject_object.wrist_filepath is not None:
            self.wrist_intensity = self.subject_object.wrist.model.epoch_intensity if \
                self.subject_object.wrist.model.epoch_intensity is not None else None

        # Sleep validity data
        if self.subject_object.sleeplog_file is not None:
            self.sleep_validity = self.subject_object.sleep.status if \
                self.subject_object.sleep.status is not None else None

    def remove_invalid_sleep(self):
        """Removes invalid epochs from Ankle, Wrist, and HR data based on sleep validity.
           If invalid data has been removed using HR data, this method further removes invalid periods due to sleep.
           If invalid data has not been removed using HR data, this method removes invalid periods due to sleep from
           the "raw" data.
        """

        if self.subject_object.sleeplog_file is not None:
            print("\n" + "Removing epochs during sleep...")

            # Ankle --------------------------------------------------------------------------------------------------
            if self.ankle_intensity is not None and self.ankle is None:
                self.ankle = [self.ankle_intensity[i] if self.sleep_validity[i] == 0 else None
                              for i in range(self.data_len)]
                self.ankle_group = [self.ankle_intensity_group[i] if self.sleep_validity[i] == 0 else None
                                    for i in range(self.data_len)]

            # Wrist --------------------------------------------------------------------------------------------------
            if self.wrist_intensity is not None and self.wrist is None:
                self.wrist = [self.wrist_intensity[i] if self.sleep_validity[i] == 0 else None
                              for i in range(self.data_len)]

            print("Complete.")

        if self.subject_object.sleeplog_file is None:
            print("No sleep data. Cannot remove sleep periods.")

            self.ankle = self.ankle_intensity
            self.ankle_group = self.ankle_intensity_group
            self.wrist = self.wrist_intensity

    def generate_validity_report(self):

        try:
            self.percent_valid = round(100 * (self.data_len - self.ankle.count(None)) / self.data_len, 1)
            self.hours_valid = round((len(self.ankle) * self.subject_object.epoch_len / 3600) *
                                     (self.percent_valid / 100), 2)
            self.final_epoch_validity = ["Invalid" if i is None else "Valid" for i in self.ankle]
        except (TypeError, AttributeError):
            self.percent_valid = round(100 * (self.data_len - self.wrist.count(None)) / self.data_len, 1)
            self.hours_valid = round((len(self.wrist) * self.subject_object.epoch_len / 3600) *
                                     (self.percent_valid / 100), 2)

        self.final_epoch_validity = ["Invalid" if i is None else "Valid" for i in self.wrist]

        self.hours_valid = round(self.final_epoch_validity.count("Valid") * self.subject_object.epoch_len / 3600, 2)

        print("\n" + "Validity check complete. {}% of the original "
                     "data is valid ({} hours).".format(self.percent_valid, self.hours_valid))

        self.validity_dict = {"Valid ECG %": None,
                              "ECG Hours Lost": None,
                              "Sleep %": self.subject_object.sleep.report["Sleep%"],
                              "Sleep Hours Lost": round(self.subject_object.sleep.report["SleepDuration"] / 60,
                                                        2),
                              "Total Valid %": self.percent_valid,
                              "Total Hours Valid": self.hours_valid}

    def plot_validity_data(self):
        """Generates 4 subplots for each activity model with invalid data removed."""

        # x-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))

        if self.subject_object.sleeplog_file is None:
            ax1.set_title("Participant {}: Accel-Only Valid Data ({}% valid) "
                          "(no sleep data)".format(self.subject_object.subjectID, self.percent_valid))

        # Fills in region where participant was asleep
        if self.subject_object.sleeplog_file is not None:
            ax1.set_title("Participant {}: Accel-Only Valid Data ({}% valid) "
                          "(green = sleep)".format(self.subject_object.subjectID, self.percent_valid))

            for day1, day2 in zip(self.subject_object.sleep.data[:], self.subject_object.sleep.data[1:]):
                try:
                    # Overnight sleep
                    ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)

                    # Daytime naps
                    ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)

                except (AttributeError, TypeError):
                    pass

        if self.ankle_intensity is not None:
            ax1.plot(self.epoch_timestamps[:self.data_len], self.ankle[:self.data_len], color='#606060', label='Ankle')
            ax1.set_ylim(-0.1, 3)
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Intensity Cat.")

        if self.wrist_intensity is not None:
            ax2.plot(self.epoch_timestamps[:self.data_len], self.wrist[:self.data_len], color='#606060', label='Wrist')
            ax2.set_ylim(-0.1, 3)
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Intensity Cat.")

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

        plt.show()

    def recalculate_activity_totals(self):

        self.hr_totals = {"Model": "HR",
                          "Sedentary": 0, "Sedentary%": 0,
                          "Light": 0, "Light%": 0,
                          "Moderate": 0, "Moderate%": 0,
                          "Vigorous": 0, "Vigorous%": 0}

        # ANKLE -------------------------------------------------------------------------------------------------------
        if self.ankle is not None:
            epoch_to_minutes = 60 / self.subject_object.ankle.epoch_len

            n_valid_epochs = len(self.ankle) - self.ankle.count(None)

            self.ankle_totals = {"Model": "Ankle",
                                 "Sedentary": self.ankle.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.ankle.count(0) / n_valid_epochs, 3),
                                 "Light": (self.ankle.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.ankle.count(1) / n_valid_epochs, 3),
                                 "Moderate": self.ankle.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.ankle.count(2) / n_valid_epochs, 3),
                                 "Vigorous": self.ankle.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.ankle.count(3) / n_valid_epochs, 3)}

            # Group regression ---------------------------------------------------------------------------------------
            epoch_to_minutes = 60 / self.subject_object.ankle.epoch_len

            n_valid_epochs = len(self.ankle_group) - self.ankle_group.count(None)

            self.ankle_totals_group = {"Model": "Ankle",
                                       "Sedentary": self.ankle_group.count(0) / epoch_to_minutes,
                                       "Sedentary%": round(self.ankle_group.count(0) / n_valid_epochs, 3),
                                       "Light": (self.ankle_group.count(1)) / epoch_to_minutes,
                                       "Light%": round(self.ankle_group.count(1) / n_valid_epochs, 3),
                                       "Moderate": self.ankle_group.count(2) / epoch_to_minutes,
                                       "Moderate%": round(self.ankle_group.count(2) / n_valid_epochs, 3),
                                       "Vigorous": self.ankle_group.count(3) / epoch_to_minutes,
                                       "Vigorous%": round(self.ankle_group.count(3) / n_valid_epochs, 3)}

        # WRIST -------------------------------------------------------------------------------------------------------
        if self.wrist is not None:
            epoch_to_minutes = 60 / self.subject_object.wrist.epoch_len

            n_valid_epochs = len(self.wrist) - self.wrist.count(None)

            self.wrist_totals = {"Model": "Wrist",
                                 "Sedentary": self.wrist.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.wrist.count(0) / n_valid_epochs, 3),
                                 "Light": (self.wrist.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.wrist.count(1) / n_valid_epochs, 3),
                                 "Moderate": self.wrist.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.wrist.count(2) / n_valid_epochs, 3),
                                 "Vigorous": self.wrist.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.wrist.count(3) / n_valid_epochs, 3)}

    def write_activity_totals(self):

        with open("{}OND07_WTL_{}_01_Valid_Activity_Totals_AccelOnly.csv".format(self.subject_object.output_dir,
                                                                                 str(self.subject_object.subjectID)),
                  'w', newline='') as outfile:

            fieldnames = ['Model', 'Sedentary', 'Sedentary%', 'Light', 'Light%',
                          'Moderate', 'Moderate%', 'Vigorous', 'Vigorous%']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            if self.subject_object.ankle is not None:
                writer.writerow(self.ankle_totals)
                writer.writerow(self.ankle_totals_group)
            if self.subject_object.wrist is not None:
                writer.writerow(self.wrist_totals)

            print()
            print("Saved activity profiles from valid data to file "
                  "{}OND07_WTL_{}_01_Valid_Activity_Totals_AccelOnly.csv".format(self.subject_object.output_dir,
                                                                                 str(self.subject_object.subjectID)))

    def write_valid_epochs(self):

        with open("{}OND07_WTL_{}_01_Valid_EpochIntensityData_AccelOnly.csv".format(self.subject_object.output_dir,
                                                                                    str(self.subject_object.subjectID)),
                  'w', newline='') as outfile:

            writer = csv.writer(outfile, delimiter=",", lineterminator="\n")

            writer.writerow(["Timestamp", "Validity", "Wrist", "Ankle", "HR"])

            # Prevents TypeError during writing to .csv if object is None
            if self.wrist_intensity is None:
                wrist_intensity = [None for i in range(len(self.epoch_timestamps))]
            if self.wrist_intensity is not None:
                wrist_intensity = self.wrist

            if self.ankle_intensity is None:
                ankle_intensity = [None for i in range(len(self.epoch_timestamps))]
            if self.ankle_intensity is not None:
                ankle_intensity = self.ankle

            writer.writerows(zip(self.epoch_timestamps, self.final_epoch_validity,
                                 wrist_intensity, ankle_intensity))

        print("\n" + "Saved epoch-by-epoch intensity data to file "
                     "{}OND07_WTL_{}_01_Valid_"
                     "EpochIntensityData_AccelOnly.csv".format(self.subject_object.output_dir,
                                                               str(self.subject_object.subjectID)))

    def write_validity_report(self):

        with open(self.subject_object.output_dir + str(self.subject_object.subjectID) +
                  "_ValidityData_AccelOnly.csv", "w") as outfile:
            fieldnames = ['Valid ECG %', 'ECG Hours Lost',
                          'Sleep %', 'Sleep Hours Lost',
                          'Total Valid %', "Total Hours Valid"]

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(self.validity_dict)

        print("\n" + "Saved validity summary data to file {}".format(self.subject_object.output_dir) +
              str(self.subject_object.subjectID) + "_ValidityData_AccelOnly.csv")

    def plot_regression_comparison(self):

        # x-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))

        fig.suptitle("Individual vs. Group Regression Comparison")

        ax1.plot(self.subject_object.ankle.epoch.timestamps, self.ankle, color='black', label="Individual")
        ax1.plot(self.subject_object.ankle.epoch.timestamps, self.ankle_group, color='red', label="Group")
        ax1.set_xlim(self.subject_object.ankle.epoch.timestamps[0]+timedelta(seconds=3600),
                     self.subject_object.ankle.epoch.timestamps[-1]+timedelta(seconds=3600))
        ax1.set_ylabel("Intensity Category")

        ax2.plot(self.subject_object.ankle.epoch.timestamps,
                 [self.subject_object.ankle.model.linear_speed[i] if self.subject_object.sleep.status[i] == 0
                  else None for i in range(len(self.subject_object.ankle.model.linear_speed))],
                 color='black', label="Individual")

        ax2.plot(self.subject_object.ankle.epoch.timestamps,
                 [self.subject_object.ankle.model.linear_speed_group[i] if self.subject_object.sleep.status[i] == 0
                  else None for i in range(len(self.subject_object.ankle.model.linear_speed_group))],
                 color='red', label='Group')

        ax2.set_xlim(self.subject_object.ankle.epoch.timestamps[0]+timedelta(seconds=3600),
                     self.subject_object.ankle.epoch.timestamps[-1]+timedelta(seconds=3600))
        ax2.set_ylabel("Predicted Speed (m/s)")

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

        if self.subject_object.sleeplog_file is not None:

            for day1, day2 in zip(self.subject_object.sleep.data[:], self.subject_object.sleep.data[1:]):
                try:
                    # Overnight sleep
                    ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='green', alpha=0.35)

                    # Daytime naps
                    ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)
                    ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='green', alpha=0.35)

                except (AttributeError, TypeError):
                    pass

    def plot_regression_totals_comparison(self):

        sedentary_minutes = [self.ankle_totals["Sedentary"], self.ankle_totals_group["Sedentary"]]

        light_minutes = [self.ankle_totals["Light"], self.ankle_totals_group["Light"]]

        moderate_minutes = [self.ankle_totals["Moderate"], self.ankle_totals_group["Moderate"]]

        vigorous_minutes = [self.ankle_totals["Vigorous"], self.ankle_totals_group["Vigorous"]]

        plt.subplots(2, 2, figsize=(10, 7))
        plt.suptitle("Participant {}: Valid Only Data".format(self.subject_object.subjectID))

        # Sedentary activity
        plt.subplot(2, 2, 1)
        plt.title("Sedentary")
        plt.bar(["Individual", "Group"], sedentary_minutes, color='grey', edgecolor='black')
        plt.ylim(0, max(sedentary_minutes) * 1.2)
        plt.ylabel("Minutes")

        # Light activity
        plt.subplot(2, 2, 2)
        plt.title("Light Activity")
        plt.bar(["Individual", "Group"], light_minutes, color='green', edgecolor='black')
        plt.ylim(0, max(light_minutes) * 1.2)

        # Moderate activity
        plt.subplot(2, 2, 3)
        plt.title("Moderate Activity")
        plt.bar(["Individual", "Group"], moderate_minutes, color='#EA5B19', edgecolor='black')
        plt.ylim(0, max(moderate_minutes) * 1.2)
        plt.ylabel("Minutes")

        # Vigorous activity
        plt.subplot(2, 2, 4)
        plt.title("Vigorous Activity")
        plt.bar(["Individual", "Group"], vigorous_minutes, color='red', edgecolor='black')
        plt.ylim(0, max(vigorous_minutes) * 1.2)

    def plot_accel_ecg_quality(self):
        """Plots raw ankle and wrist accelerometer data, raw ECG data, and ECG validity status."""

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 7))

        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 8, 16], interval=1)

        ax1.set_title("Participant {}: Movement's effect on ECG validity".format(self.subjectID))

        if self.wrist_filepath is not None and self.load_raw_wrist:
            ax1.plot(self.wrist.raw.timestamps[::3], self.wrist.raw.x[::3], color='black',
                     label='Wrist ({}Hz)'.format(int(self.wrist.raw.sample_rate) / 3))
            ax1.set_ylabel("G's")
            ax1.legend(loc='upper left')
            ax1.set_ylim(-8, 8)

        if self.ankle_filepath is not None and self.load_raw_ankle:
            ax2.plot(self.ankle.raw.timestamps[::3], self.ankle.raw.x[::3], color='black',
                     label='Ankle ({}Hz'.format(int(self.ankle.raw.sample_rate) / 3))
            ax2.set_ylabel("G's")
            ax2.legend(loc='upper left')
            ax2.set_ylim(-8, 8)

        if self.ecg_filepath is not None and self.load_raw_ecg:
            ax3.plot(self.ecg.timestamps[::5], self.ecg.filtered[::5], color='red',
                     label='ECG ({}Hz, filtered)'.format(int(self.ecg.sample_rate) / 5))
            ax3.set_ylabel("Voltage")
            ax3.legend(loc='upper left')

        if self.ecg_filepath is not None and self.load_raw_wrist and self.ecg.epoch_validity is not None:
            ax4.plot(self.ecg.epoch_timestamps, self.ecg.epoch_validity, color='black', label="ECG Validity")
            ax4.fill_between(x=self.ecg.epoch_timestamps, y1=0, y2=self.ecg.epoch_validity, color='grey')
            ax4.set_ylabel("1 = invalid")
            ax4.legend(loc='upper left')

        ax4.xaxis.set_major_formatter(xfmt)
        ax4.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

    def plot_treadmill_protocol(self):

        if not self.load_ankle:
            print("No ankle data available.")

        if self.load_ankle:

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col')

            start_index = int(self.ankle.treadmill.walk_indexes[0] - 5 * (60 / self.epoch_len))
            if start_index < 0:
                start_index = 0

            end_index = int(self.ankle.treadmill.walk_indexes[-1] + 5 * (60 / self.epoch_len))

            plt.suptitle("Participant {}: HR and Accel Data during Treadmill Protocol".format(self.subjectID))

            if self.wrist_filepath is not None:
                ax1.plot(self.wrist.epoch.timestamps[start_index:end_index],
                         self.wrist.epoch.svm[start_index:end_index], label="Wrist", color='black')
                ax1.set_ylabel("Counts")
                ax1.legend(loc='upper left')

            ax2.plot(self.ankle.epoch.timestamps[start_index:end_index],
                     self.ankle.epoch.svm[start_index:end_index], label="Ankle", color='black')
            ax2.set_ylabel("Counts")
            ax2.legend(loc='upper left')

            if self.load_ecg:
                ax3.plot(self.ecg.epoch_timestamps[start_index:end_index],
                         self.ecg.epoch_hr[start_index:end_index], label="HR", color='red')
                ax3.set_ylabel("HR (bpm)")

            plt.show()