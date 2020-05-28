import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import csv
import statistics
from datetime import datetime


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
        try:
            day_list = pd.date_range(start=self.timestamps[0].date(), end=self.timestamps[-1].date(),
                                     periods=(self.timestamps[-1].date()-self.timestamps[0].date()).days+1)

        # Handles stupid timestamp formatting when reading epoched data from raw as opposed to from processed
        except AttributeError:
            start_stamp = datetime.strptime(str(self.timestamps[0])[:-3], "%Y-%m-%dT%H:%M:%S.%f")
            end_stamp = datetime.strptime(str(self.timestamps[-1])[:-3], "%Y-%m-%dT%H:%M:%S.%f")

            day_list = pd.date_range(start=start_stamp.date(), end=end_stamp.date(),
                                     periods=(end_stamp.date()-start_stamp.date()).days+1)

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
            if len(non_zero_hr) == 0:
                non_zero_hr = [1]

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
            if len(non_zero_hr) == 0:
                non_zero_hr = [1]

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

    def plot_week(self, day_definition="sleep"):

        # Which data to use
        if day_definition == "Sleep" or day_definition == "sleep":
            df = self.report_sleep_df
        if day_definition == "Date" or day_definition == "date":
            df = self.report_df

        # DATA LABELS ------------------------------------------------------------------------------------------------
        def gen_sleep_label(rects, value):
            for rect in rects:
                height = rect.get_height()
                plt.annotate('{} \n hours'.format(round(value, 2)),
                             xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                             xytext=(0, 0),
                             textcoords="offset points", color='white',
                             ha='center', va='center')

        def gen_activity_label(rects, value):
            for rect in rects:
                height = rect.get_height()
                plt.annotate('{} \n hours'.format(round(value, 2)),
                             xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                             xytext=(0, 0),
                             textcoords="offset points", color='black',
                             ha='center', va='center')

        # PLOTTING ---------------------------------------------------------------------------------------------------
        plt.subplots(1, 2, figsize=(10, 7))

        plt.suptitle("Subject {}: Weekly Report".format(self.subjectID))

        # SLEEP DATA
        plt.subplot(1, 2, 1)

        for day in range(0, df.shape[0]):
            sleep_plot = plt.bar(x="Day {}".format(day + 1), height=df.iloc[day]["Sleep Minutes"] / 60,
                                 color="dimgray", edgecolor='black')
            gen_sleep_label(rects=sleep_plot, value=df.iloc[day]["Sleep Minutes"] / 60)

            plt.ylabel("Hours")
            plt.title("Sleep Durations")
            plt.yticks(np.arange(0, max(df["Sleep Minutes"]) / 60 * 1.1, 1))
            plt.ylim(0, max(df["Sleep Minutes"]) / 60 * 1.1)

        plt.subplot(1, 2, 2)

        # WRIST ACTIVITY DATA
        for day in range(0, df.shape[0]):

            awake_minutes = df.iloc[day]["Period Length (H)"] * 60 - df.iloc[day]["Sleep Minutes"]
            active_percent = 100 * df.iloc[day]["Wrist Active Minutes"] / awake_minutes
            inactive_percent = 100 - active_percent

            if awake_minutes < 60:
                plt.bar(x="Day {}".format(day + 1), height=0)
                break

            inactive = plt.bar(x="Day {}".format(day + 1), height=inactive_percent,
                               color='darkgrey', edgecolor='black', bottom=active_percent)
            gen_activity_label(rects=inactive, value=awake_minutes)

            active = plt.bar(x="Day {}".format(day + 1), height=active_percent,
                             color='green', edgecolor='black')
            gen_activity_label(rects=active, value=df.iloc[day]["Wrist Active Minutes"])

            plt.title("Activity by Day")
            plt.yticks(np.arange(0, 110, 10))
            plt.ylabel("% of waking hours")

    def plot_week_activity_only(self, day_definition="sleep"):

        # Which data to use
        if day_definition == "Sleep" or day_definition == "sleep":
            df = self.report_sleep_df
        if day_definition == "Date" or day_definition == "date":
            df = self.report_df

        # DATA LABELS ------------------------------------------------------------------------------------------------
        def gen_sleep_label(rects, value):
            for rect in rects:
                height = rect.get_height()

                hours = int(np.floor(value))
                minutes = (value - hours) * 60
                # plt.annotate('{} \n hours'.format(round(value, 1)),
                plt.annotate('{} hours\n{} mins'.format(int(hours), int(minutes)),
                             xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                             xytext=(0, 0),
                             textcoords="offset points", color='white',
                             ha='center', va='center')

        def gen_light_label(rects, value):
            for rect in rects:
                delta_height = rect.get_height() - df.iloc[day]["Wrist MVPA Minutes"]

                if rect.get_height() == 0:
                    return None

                plt.annotate('{}\nmins'.format(int(value)),
                             xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() - delta_height / 2),
                             xytext=(0, 0),
                             textcoords="offset points", color='black',
                             ha='center', va='center')

        def gen_mvpa_label(rects, value):
            for rect in rects:
                height = rect.get_height()

                if height == 0:
                    return None

                plt.annotate('{}\nmins'.format(int(value)),
                             xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                             xytext=(0, 0),
                             textcoords="offset points", color='black',
                             ha='center', va='center')

        def gen_total_label(rects, value):
            for rect in rects:
                height = rect.get_height()

                if height == 0:
                    return None

                plt.annotate('{}\ntotal\nmins'.format(int(value)),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 20),
                             textcoords="offset points", color='black',
                             ha='center', va='center')

        def gen_invalid_label(rects):
            for rect in rects:
                height = max(df["Wrist Active Minutes"] / 2)

                plt.annotate('Not \n enough \n data \n today',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 16),
                             textcoords="offset points", color='black',
                             ha='center', va='center')

        # PLOTTING ---------------------------------------------------------------------------------------------------
        plt.subplots(1, 2, figsize=(12, 7))
        plt.rcParams.update({'font.size': 12})

        plt.subplots_adjust(bottom=0.16, wspace=0.29)
        plt.suptitle("Sleep and Physical Activity Report")

        # SLEEP DATA -------------------------------------------------------------------------------------------------
        plt.subplot(1, 2, 1)

        for day in range(0, df.shape[0]):
            day_label = datetime.strftime(df.iloc[day]["Day Start"].date(), "%a., %b. %d")

            # Doesn't plot final day if value is 0
            if df.iloc[day]["Sleep Minutes"] == 0 and day == df.shape[0]-1:
                break

            sleep_plot = plt.bar(x=day_label, height=df.iloc[day]["Sleep Minutes"]/60,
                                 color="#314A97", edgecolor='black', linewidth=1.5)
            gen_sleep_label(rects=sleep_plot, value=df.iloc[day]["Sleep Minutes"]/60)

        plt.ylabel("Hours per Night")
        plt.title("Sleep")
        plt.yticks(np.arange(0, max(df["Sleep Minutes"]) * 1.1, 1))
        plt.ylim(0, max(df["Sleep Minutes"]) / 60 * 1.1)
        plt.xticks(rotation=45, fontsize=12)

        # WRIST ACTIVITY DATA -----------------------------------------------------------------------------------------
        plt.subplot(1, 2, 2)
        plt.rcParams.update({'font.size': 12})

        for day in range(0, df.shape[0]):

            day_label = datetime.strftime(df.iloc[day]["Day Start"].date(), "%a., %b. %d")

            light_hours = df.iloc[day]["Wrist Active Minutes"] - df.iloc[day]["Wrist MVPA Minutes"]

            # Plots blank spot if day was < 6 hours
            if df.iloc[day]["Period Length (H)"] < 6:
                # Doesn't plot final at all day if day was < 6 hours
                if day == df.shape[0]-1:
                    break

                # Blank spot with "Not enough data today" label
                light = plt.bar(x=day_label, height=0, color='white', alpha=0.0, edgecolor='black')
                gen_invalid_label(rects=light)

            # Days longer than 6 hours
            if df.iloc[day]["Period Length (H)"] >= 6:

                # Light activity -----------------------------------------------------------
                light = plt.bar(x=day_label, height=df.iloc[day]["Wrist Active Minutes"],
                                color="darkgrey", alpha=.75, edgecolor='black', linewidth=1.5)
                gen_light_label(rects=light, value=light_hours)

                # MVPA ---------------------------------------------------------------------
                mvpa = plt.bar(x=day_label, height=df.iloc[day]["Wrist MVPA Minutes"],
                               color="gold", edgecolor='black', linewidth=1.5)
                gen_mvpa_label(rects=mvpa, value=df.iloc[day]["Wrist MVPA Minutes"])

                # Total - for label; no bar seen
                total = plt.bar(x=day_label, height=df.iloc[day]["Wrist Active Minutes"],
                                color='white', edgecolor='black', alpha=0)
                gen_total_label(rects=total, value=df.iloc[day]["Wrist Active Minutes"])

            plt.legend(loc='upper right', labels=["Light", "Mod/Vig"])
            plt.title("Daily Physical Activity")
            plt.xticks(rotation=45, fontsize=12)
            plt.ylim(0, max(df["Wrist Active Minutes"])*1.3)
            plt.ylabel("Minutes per Day")
