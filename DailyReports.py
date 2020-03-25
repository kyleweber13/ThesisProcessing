import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import csv
import statistics


class DailyReport:

    def __init__(self, subject_object=None, n_resting_hrs=30):

        self.subjectID = subject_object.subjectID
        self.output_dir = subject_object.output_dir
        self.process_sleep = subject_object.sleeplog_file is not None

        if subject_object.load_ankle:
            self.timestamps = subject_object.ankle.epoch.timestamps

            self.ankle = subject_object.ankle.epoch.svm
            self.ankle_intensity = subject_object.ankle.model.epoch_intensity

        if not subject_object.load_ankle and subject_object.load_wrist:
            self.timestamps = subject_object.wrist.epoch.timestamps
            self.ankle = None
            self.ankle_intensity = None

        if not subject_object.load_ankle and not subject_object.load_wrist and subject_object.load_ecg:
            self.timestamps = subject_object.ecg.epoch_timestamps

            self.wrist = None
            self.wrist_intensity = None

        if subject_object.load_wrist:
            self.wrist = subject_object.wrist.epoch.svm
            self.wrist_intensity = subject_object.wrist.model.epoch_intensity

        if subject_object.load_ecg:
            self.hr = subject_object.ecg.epoch_hr
            self.valid_hr = subject_object.ecg.valid_hr
            self.hr_intensity = subject_object.ecg.epoch_intensity

        if self.process_sleep:
            self.sleep_status = [i for i in subject_object.sleep.status]
            self.sleep_timestamps = subject_object.sleep.data
        if not self.process_sleep:
            self.sleep_status = None
            self.sleep_timestamps = None

        self.epoch_len = subject_object.epoch_len
        self.n_resting_hrs = n_resting_hrs

        self.roll_avg_hr = []
        self.max_hr_timeofday = []
        self.max_hr_indexes = []

        self.day_indexes = []
        self.sleep_indexes = []

        self.report_df = None
        self.report_sleep_df = None

        self.create_index_list()

        self.create_activity_report()

        if self.process_sleep:
            self.create_sleep_indexes()
            self.create_activity_report_sleep()

    def create_index_list(self):

        # Creates a list of dates from start of first day to last day of collection
        day_list = pd.date_range(start=self.timestamps[0].date(), end=self.timestamps[-1].date(),
                                 periods=(self.timestamps[-1].date()-self.timestamps[0].date()).days+1)

        # Gets indexes that correspond to start of days
        for day in day_list:
            for i, stamp in enumerate(self.timestamps):
                if stamp > day:
                    self.day_indexes.append(i)
                    break

        # Adds index of last datapoint to list
        self.day_indexes.append(len(self.timestamps)-1)

    def create_sleep_indexes(self):

        sleep_timestamps = []

        for i in self.sleep_timestamps:
            sleep_timestamps.append(i[0])

        self.sleep_timestamps = sleep_timestamps

        # Gets indexes that correspond to each sleep event
        for i in self.sleep_timestamps:
            if i != "N/A":
                for j, stamp in enumerate(self.timestamps):
                    if stamp > i:
                        self.sleep_indexes.append(j)
                        break

        self.sleep_indexes.insert(0, 0)
        self.sleep_indexes.append(len(self.timestamps) - 1)

    def create_activity_report(self):
        """Generates activity report using 12:00am-11:59pm clock."""

        # Calculates rolling average of 4 consecutive epochs without invalid epochs
        for index in range(0, len(self.valid_hr) - 4):
            window = [i for i in self.valid_hr[index:index + 4] if i is not None]

            if len(window) == 4:
                self.roll_avg_hr.append(statistics.mean(window))
            else:
                self.roll_avg_hr.append(0)

        # Pads with zeros due to lost epochs with averaging
        for i in range(4):
            self.roll_avg_hr.append(0)

        day_num_list = []
        day_start_list = []

        max_hr_list = []
        max_hr_time_list = []
        mean_hr_list = []
        rest_hr_list = []

        wrist_active_minutes_list = []
        wrist_mvpa_minutes_list = []

        sleep_minutes_list = []

        period_length_list = []
        for start, end, day_num in zip(self.day_indexes[:], self.day_indexes[1:],
                                       np.arange(1, len(self.day_indexes))):

            day_num_list.append(day_num)
            day_start_list.append(self.timestamps[start])

            period_length_list.append(((end - start) / (60 / self.epoch_len) / 60))

            # HR data: max, mean, resting
            non_zero_hr = [i for i in self.hr[start:end] if i > 0]

            if self.process_sleep:
                awake_hr = [value for i, value in enumerate(self.hr[start:end])
                            if self.sleep_status[start + i] == 0 and value > 0]
            if not self.process_sleep:
                awake_hr = non_zero_hr

            max_hr_list.append(round(max(self.roll_avg_hr[start:end]), 1))
            max_hr_time_list.append(
                self.timestamps[self.roll_avg_hr[start:end].index(max(self.roll_avg_hr[start:end])) + start])
            mean_hr_list.append(round(sum(non_zero_hr) / len(non_zero_hr), 1))
            rest_hr_list.append(round(sum(sorted(awake_hr)[:self.n_resting_hrs]) / self.n_resting_hrs, 1))

            wrist_active_minutes_list.append(
                ((end - start) - self.wrist_intensity[start:end].count(0)) / (60 / self.epoch_len))
            wrist_mvpa_minutes_list.append(
                (self.wrist_intensity[start:end].count(2) + self.wrist_intensity[start:end].count(3)) / (
                            60 / self.epoch_len))

            # Sleep report
            if self.process_sleep:
                sleep_minutes_list.append(
                    (end - start - self.sleep_status[start:end].count(0)) / (60 / self.epoch_len))

            if not self.process_sleep:
                sleep_minutes_list.append(0)

        dataframe_dict = {"Day Definition": ["Calendar Date" for i in range(len(day_start_list))],
                          "Day Start": day_start_list,
                          "Max HR": max_hr_list,
                          "Max HR Time": max_hr_time_list,
                          "Mean HR": mean_hr_list,
                          "Resting HR": rest_hr_list,
                          "Wrist Active Minutes": wrist_active_minutes_list,
                          "Wrist MVPA Minutes": wrist_mvpa_minutes_list,
                          "Sleep Minutes": sleep_minutes_list,
                          "Period Length (H)": period_length_list}

        self.report_df = pd.DataFrame(dataframe_dict, index=day_num_list)

    def create_activity_report_sleep(self):
        """Generates activity report using days as defined by when participant went to bed."""

        # ADAM START -------------------------------------------------------------------------------------------------
        day_num_list = []
        day_start_list = []

        max_hr_list = []
        max_hr_time_list = []
        mean_hr_list = []
        rest_hr_list = []

        wrist_active_minutes_list = []
        wrist_mvpa_minutes_list = []

        sleep_minutes_list = []

        period_length_list = []
        for start, end, day_num in zip(self.sleep_indexes[:], self.sleep_indexes[1:],
                                       np.arange(1, len(self.sleep_indexes))):

            day_num_list.append(day_num)
            day_start_list.append(self.timestamps[start])

            period_length_list.append(((end - start) / (60 / self.epoch_len) / 60))

            # HR data: max, mean, resting
            non_zero_hr = [i for i in self.hr[start:end] if i > 0]

            if self.process_sleep:
                awake_hr = [value for i, value in enumerate(self.hr[start:end])
                            if self.sleep_status[start + i] == 0 and value > 0]
            if not self.process_sleep:
                awake_hr = non_zero_hr

            max_hr_list.append(round(max(self.roll_avg_hr[start:end]), 1))
            max_hr_time_list.append(
                self.timestamps[self.roll_avg_hr[start:end].index(max(self.roll_avg_hr[start:end])) + start])
            mean_hr_list.append(round(sum(non_zero_hr) / len(non_zero_hr), 1))
            rest_hr_list.append(round(sum(sorted(awake_hr)[:self.n_resting_hrs]) / self.n_resting_hrs, 1))

            wrist_active_minutes_list.append(
                ((end - start) - self.wrist_intensity[start:end].count(0)) / (60 / self.epoch_len))
            wrist_mvpa_minutes_list.append(
                (self.wrist_intensity[start:end].count(2) + self.wrist_intensity[start:end].count(3)) / (
                        60 / self.epoch_len))

            # Sleep report
            if self.process_sleep:
                sleep_minutes_list.append(
                    (end - start - self.sleep_status[start:end].count(0)) / (60 / self.epoch_len))

            if not self.process_sleep:
                sleep_minutes_list.append(0)

        dataframe_dict = {"Day Definition": ["Sleep Cycle" for i in range(len(day_start_list))],
                          "Day Start": day_start_list,
                          "Max HR": max_hr_list,
                          "Max HR Time": max_hr_time_list,
                          "Mean HR": mean_hr_list,
                          "Resting HR": rest_hr_list,
                          "Wrist Active Minutes": wrist_active_minutes_list,
                          "Wrist MVPA Minutes": wrist_mvpa_minutes_list,
                          "Sleep Minutes": sleep_minutes_list,
                          "Period Length (H)": period_length_list}

        self.report_sleep_df = pd.DataFrame(dataframe_dict, index=day_num_list)

    def plot_hr_data(self, day_definition="sleep"):

        if day_definition == "sleep" or day_definition == "Sleep":
            df = self.report_sleep_df
        if day_definition == "Date" or day_definition == "date":
            df = self.report_df

        if not self.process_sleep and df == self.report_sleep_df:
            print("No sleep data to process.")
            return None

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))

        plt.suptitle("Participant {}: Heart Rate Data (Day = {})".format(self.subjectID,
                                                                         df.iloc[0]["Day Definition"]))

        ax1.bar(x=["Day " + str(i) for i in np.arange(1, df.shape[0]+1)],
                height=df["Max HR"],
                label="Max HR", color='grey', edgecolor='black')
        ax1.set_ylabel("HR (bpm)")
        ax1.legend()

        ax2.bar(x=["Day " + str(i) for i in np.arange(1, df.shape[0]+1)], height=df["Mean HR"],
                label="Mean HR", color='grey', edgecolor='black')
        ax2.set_ylabel("HR (bpm)")
        ax2.legend()

        ax3.bar(x=["Day " + str(i) for i in np.arange(1, df.shape[0]+1)], height=df["Resting HR"],
                label="Rest HR", color='grey', edgecolor='black')
        ax3.set_ylabel("HR (bpm)")
        ax3.legend()

        plt.show()

    def plot_wrist_data(self, day_definition="sleep"):

        if day_definition == "sleep" or day_definition == "Sleep":
            df = self.report_sleep_df
        if day_definition == "Date" or day_definition == "date":
            df = self.report_df

        if not self.process_sleep and df == self.report_sleep_df:
            print("No sleep data to process.")
            return None

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))

        plt.suptitle("Participant {}: Wrist Activity (Day = {})".format(self.subjectID,
                                                                        df.iloc[0]["Day Definition"]))
        ax1.bar(x=["Day " + str(i) for i in np.arange(1, df.shape[0]+1)],
                height=df["Wrist Active Minutes"],
                label="Non-Sedentary", color='green', edgecolor='black')
        ax1.set_ylabel("Minutes")
        ax1.legend()

        ax2.bar(x=["Day " + str(i) for i in np.arange(1, df.shape[0]+1)], height=df["Wrist MVPA Minutes"],
                label="MVPA", color='#EA890C', edgecolor='black')
        ax2.set_ylabel("Minutes")
        ax2.legend()

        plt.show()

    def plot_day_splits(self):

        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))
        plt.suptitle("Subject {}: Day Divisions (blue = sleep; red = calendar date)")

        ax1.plot(self.timestamps, self.wrist, color='black', label='Wrist')
        ax1.set_ylabel("Counts")
        ax1.legend(loc='upper left')

        ax2.plot(self.timestamps, self.valid_hr[:len(self.timestamps)],
                 color='red', label='HR')
        ax2.set_ylabel("HR (bpm)")
        ax2.legend(loc='upper left')

        for day_index1, day_index2 in zip(self.day_indexes[:], self.day_indexes[1:]):
            ax1.fill_between(x=self.timestamps[day_index1:day_index2 - 100],
                             y1=0, y2=max(self.wrist) / 2, color='red', alpha=0.35)

            ax2.fill_between(x=self.timestamps[day_index1:day_index2 - 100],
                             y1=min([i for i in self.hr if i > 1]), y2=max(self.hr) / 2, color='red', alpha=0.35)

        for sleep_index1, sleep_index2 in zip(self.sleep_indexes[:], self.sleep_indexes[1:]):
            ax1.fill_between(x=self.timestamps[sleep_index1:sleep_index2 - 100],
                             y1=max(self.wrist) / 2, y2=max(self.wrist), color='blue', alpha=0.35)

            ax2.fill_between(x=self.timestamps[sleep_index1:sleep_index2 - 100],
                             y1=max([i for i in self.hr if i > 1]) / 2, y2=max(self.hr), color='blue', alpha=0.35)

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

    def write_summary(self):

        self.report_df.to_csv(path_or_buf=self.output_dir + "{}_DailyReport.csv".format(self.subjectID), sep=",")

        if self.process_sleep:
            self.report_sleep_df.to_csv(path_or_buf=self.output_dir + "{}_DailyReportSleep.csv".format(self.subjectID),
                                        sep=",")

        """with open(file="{}{}_DailySummaries.csv".format(self.output_dir, self.subjectID), mode="w") as outfile:

            fieldnames = []

            for key in self.hr_report.keys():
                fieldnames.append(key)
            for key in self.wrist_report.keys():
                if key != "Day Definition":
                    fieldnames.append(key)
            for key in self.sleep_report.keys():
                fieldnames.append(key)
            for key in self.period_lengths.keys():
                fieldnames.append(key)
                    
            date_output = []
            
            for value in self.hr_report.values():
                date_output.append(value)
            for key, value in zip(self.wrist_report.keys(), self.wrist_report.values()):
                if key != "Day Definition":
                    date_output.append(value)
            for value in self.sleep_report.values():
                date_output.append(value)
            for value in self.period_lengths.values():
                date_output.append(value)

            sleep_output = []

            for value in self.hr_report_sleep.values():
                sleep_output.append(value)
            for key, value in zip(self.wrist_report_sleep.keys(), self.wrist_report_sleep.values()):
                if key != "Day Definition":
                    sleep_output.append(value)
            for value in self.sleep_report_sleep.values():
                sleep_output.append(value)
            for value in self.period_lengths_sleep.values():
                sleep_output.append(value)

            writer = csv.writer(outfile, lineterminator="\n", delimiter=',')

            writer.writerow(fieldnames)
            writer.writerow(date_output)
            writer.writerow(sleep_output)"""
