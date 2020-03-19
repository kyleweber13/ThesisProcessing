from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import csv
import statistics as stats
import scipy.stats


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
        self.wrist_intensity = None
        self.hr_intensity = None
        self.hracc_intensity = None

        # Data used to determine what epochs are valid
        self.hr_validity = None
        self.sleep_validity = None

        # Data that only contains valid epochs
        self.ankle = None
        self.wrist = None
        self.hr = None
        self.hr_acc = None

        # Dictionaries for activity totals using valid data
        self.ankle_totals = None
        self.wrist_totals = None
        self.hr_totals = None
        self.hracc_totals = None

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
        if self.subject_object.sleeplog_folder is not None:
            self.remove_invalid_sleep()

        self.recalculate_activity_totals()

        self.generate_validity_report()

        self.check_ecgvalidity_activitylevel()

        if self.write_results:
            self.write_activity_totals()
            self.write_validity_report()
            self.write_valid_epochs()

        # self.plot_validity_data()

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

        # Error handling if ankle model not run (no treadmill protocol?)
        except AttributeError:
            self.ankle_intensity = None

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
        if self.subject_object.sleeplog_folder is not None:
            self.sleep_validity = self.subject_object.sleep.sleep_status if \
                self.subject_object.sleep.sleep_status is not None else None

    def remove_invalid_hr(self):
        """Removes invalid epochs from Ankle and Wrist data based on HR validity."""

        print("\n" + "Removing invalid HR epochs...")

        self.hr = self.hr_intensity

        if self.ankle_intensity is not None:
            self.ankle = [self.ankle_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]

        if self.wrist_intensity is not None:
            self.wrist = [self.wrist_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]

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

        # If ankle data available and invalid data was removed
        if self.ankle_intensity is not None and self.ankle is not None:
            self.ankle = [self.ankle[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

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

        self.validity_dict = {"Valid ECG %": 100 - self.subject_object.ecg.quality_report["Percent invalid"],
                              "ECG Hours Lost": self.subject_object.ecg.quality_report["Hours lost"],
                              "Sleep %": self.subject_object.sleep.sleep_report["Sleep%"],
                              "Sleep Hours Lost": round(self.subject_object.sleep.sleep_report["SleepDuration"] / 60,
                                                        2),
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
        ax1.set_title("Participant {}: Valid Data ({}% valid) (grey = sleep)".format(self.subject_object.subjectID,
                                                                                     self.percent_valid))

        # Fills in region where participant was asleep
        for day1, day2 in zip(self.subject_object.sleep.sleep_data[:], self.subject_object.sleep.sleep_data[1:]):
            try:
                # Overnight sleep
                ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax3.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax4.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)

                # Daytime naps
                ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax3.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax4.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)

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

        if self.ankle is None:
            self.ankle_totals = {"Model": "Ankle",
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

        with open("{}Model Output/OND07_WTL_{}_01_Valid_EpochIntensityData.csv".format(self.subject_object.output_dir,
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

    def write_validity_report(self):

        with open(self.subject_object.output_dir + "Validity Check/" + self.subject_object.subjectID +
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
              self.subject_object.subjectID + "_ValidityData.csv")


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
        self.wrist_intensity = None

        # Data used to determine what epochs are valid
        self.sleep_validity = None

        # Data that only contains valid epochs
        self.ankle = None
        self.wrist = None

        # Dictionaries for activity totals using valid data
        self.ankle_totals = None
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
        if self.subject_object.sleeplog_folder is not None:
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

        # Error handling if ankle model not run (no treadmill protocol?)
        except AttributeError:
            self.ankle_intensity = None

        # Wrist intensity data
        if self.subject_object.wrist_filepath is not None:
            self.wrist_intensity = self.subject_object.wrist.model.epoch_intensity if \
                self.subject_object.wrist.model.epoch_intensity is not None else None

        # Sleep validity data
        if self.subject_object.sleeplog_folder is not None:
            self.sleep_validity = self.subject_object.sleep.sleep_status if \
                self.subject_object.sleep.sleep_status is not None else None

    def remove_invalid_sleep(self):
        """Removes invalid epochs from Ankle, Wrist, and HR data based on sleep validity.
           If invalid data has been removed using HR data, this method further removes invalid periods due to sleep.
           If invalid data has not been removed using HR data, this method removes invalid periods due to sleep from
           the "raw" data.
        """

        print("\n" + "Removing epochs during sleep...")

        # Ankle -------------------------------------------------------------------------------------------------------
        if self.ankle_intensity is not None and self.ankle is None:
            self.ankle = [self.ankle_intensity[i] if self.sleep_validity[i] == 0 else None
                          for i in range(self.data_len)]

        # Wrist ------------------------------------------------------------------------------------------------------
        if self.wrist_intensity is not None and self.wrist is None:
            self.wrist = [self.wrist_intensity[i] if self.sleep_validity[i] == 0 else None
                          for i in range(self.data_len)]

        print("Complete.")

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
                              "Sleep %": self.subject_object.sleep.sleep_report["Sleep%"],
                              "Sleep Hours Lost": round(self.subject_object.sleep.sleep_report["SleepDuration"] / 60,
                                                        2),
                              "Total Valid %": self.percent_valid,
                              "Total Hours Valid": self.hours_valid}

    def plot_validity_data(self):
        """Generates 4 subplots for each activity model with invalid data removed."""

        # x-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))
        ax1.set_title("Participant {}: Accel-Only Valid Data ({}% valid) "
                      "(grey = sleep)".format(self.subject_object.subjectID, self.percent_valid))

        # Fills in region where participant was asleep
        for day1, day2 in zip(self.subject_object.sleep.sleep_data[:], self.subject_object.sleep.sleep_data[1:]):
            try:
                # Overnight sleep
                ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)

                # Daytime naps
                ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)

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

        with open("{}Model Output/OND07_WTL_{}"
                  "_01_Valid_Activity_Totals_AccelOnly.csv".format(self.subject_object.output_dir,
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

            print()
            print("Saved activity profiles from valid data to file "
                  "{}Model Output/OND07_WTL_{}"
                  "_01_Valid_Activity_Totals_AccelOnly.csv".format(self.subject_object.output_dir,
                                                                   self.subject_object.subjectID))

    def write_valid_epochs(self):

        with open("{}Model Output/OND07_WTL_{}"
                  "_01_Valid_EpochIntensityData_AccelOnly.csv".format(self.subject_object.output_dir,
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

            writer.writerows(zip(self.epoch_timestamps, self.final_epoch_validity,
                                 wrist_intensity, ankle_intensity))

        print("\n" + "Saved epoch-by-epoch intensity data to file "
                     "{}Model Output/OND07_WTL_{}_01_Valid_EpochIntensityData_AccelOnly.csv"
              .format(self.subject_object.output_dir, self.subject_object.subjectID))

    def write_validity_report(self):

        with open(self.subject_object.output_dir + "Validity Check/" + self.subject_object.subjectID +
                  "_ValidityData_AccelOnly.csv", "w") as outfile:
            fieldnames = ['Valid ECG %', 'ECG Hours Lost',
                          'Sleep %', 'Sleep Hours Lost',
                          'Total Valid %', "Total Hours Valid"]

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(self.validity_dict)

        print("\n" + "Saved validity summary data to file {}".format(self.subject_object.output_dir) +
              "Validity Check/" + self.subject_object.subjectID + "_ValidityData_AccelOnly.csv")
