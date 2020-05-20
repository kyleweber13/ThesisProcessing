import pandas as pd
from datetime import datetime
from datetime import timedelta
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt


class GGIR:

    def __init__(self, file, subject_object):

        self.data = pd.read_excel(file)
        self.subject_object = subject_object
        self.formatted_stamps = None

        self.data = self.data.loc[self.data["FILE"] == self.subject_object.subjectID]

        self.format_stamps()

    def format_stamps(self):

        ggir_stamps = []
        ptct_stamps = []

        for date, ggir_time, ptct_time in zip(self.data["DATE"], self.data["ONSET - GGIR"], self.data["ONSET - PTCT"]):
            try:
                if ggir_time.hour <= 10:
                    stamp = str((datetime.strptime(date, "%d/%m/%Y") + timedelta(days=1)).date()) + " " + str(ggir_time)
                    ggir_stamps.append(stamp)
                else:
                    stamp = str((datetime.strptime(date, "%d/%m/%Y")).date()) + " " + str(ggir_time)
                    ggir_stamps.append(stamp)
            except AttributeError:
                pass

            try:
                if ptct_time.hour <= 10:
                    stamp = str((datetime.strptime(date, "%d/%m/%Y") + timedelta(days=1)).date()) + " " + str(ptct_time)
                    ptct_stamps.append(stamp)
                else:
                    stamp = str((datetime.strptime(date, "%d/%m/%Y")).date()) + " " + str(ptct_time)
                    ptct_stamps.append(stamp)
            except AttributeError:
                pass

            ggir_wake = [str(self.data.iloc[i]["DATE"]) + " " + str(self.data.iloc[i]["WAKE - GGIR"]) for i in
                         range(0, self.data.shape[0]) if self.data.iloc[i]["WAKE - GGIR"] is not None]

            ptct_wake = [str(self.data.iloc[i]["DATE"]) + " " + str(self.data.iloc[i]["WAKE - PTCT"]) for i in
                         range(0, self.data.shape[0])]

            self.formatted_stamps = pd.DataFrame(list(zip_longest(np.arange(1, self.data.shape[0] + 1),
                                                                  ggir_stamps, ptct_stamps,
                                                          ggir_wake, ptct_wake)),
                                                 columns=["Day", "ONSET - GGIR", "ONSET - PTCT",
                                                          "WAKE - GGIR", "WAKE - PTCT"])

    def plot_events(self):

        plt.plot(self.subject_object.wrist.epoch.timestamps, self.subject_object.wrist.epoch.svm, color='black')
        plt.title("Red = GGIR; Blue = Log")

        for day in range(self.formatted_stamps.shape[0]):

            # GGIR DATA
            """try:
                plt.axvline(ggir.formatted_stamps.iloc[day]["ONSET - GGIR"], linestyle='dashed', color='red')
            except TypeError:
                pass

            try:
                plt.axvline(datetime.strptime(ggir.formatted_stamps.iloc[day]["WAKE - GGIR"], "%d/%m/%Y %H:%M:%S"),
                            linestyle='dashed', color='red')
            except (TypeError, ValueError):
                pass"""

            try:
                plt.fill_betweenx(
                    x1=datetime.strptime(self.formatted_stamps.iloc[day]["ONSET - GGIR"], "%Y-%m-%d %H:%M:%S"),
                    x2=datetime.strptime(self.formatted_stamps.iloc[day + 1]["WAKE - GGIR"], "%d/%m/%Y %H:%M:%S"),
                    y=[max(self.subject_object.wrist.epoch.svm) / 4, max(self.subject_object.wrist.epoch.svm) / 2],
                    alpha=.5, color='red')
            except (ValueError, TypeError, IndexError):
                pass

            # LOG DATA
            """try:
                plt.axvline(ggir.formatted_stamps.iloc[day]["ONSET - PTCT"], linestyle='dashed', color='blue')
            except TypeError:
                pass

            try:
                plt.axvline(datetime.strptime(ggir.formatted_stamps.iloc[day]["WAKE - PTCT"], "%d/%m/%Y %H:%M:%S"),
                            linestyle='dashed', color='blue')
            except (TypeError, ValueError):
                pass"""

            try:
                plt.fill_betweenx(
                    x1=datetime.strptime(self.formatted_stamps.iloc[day]["ONSET - PTCT"], "%Y-%m-%d %H:%M:%S"),
                    x2=datetime.strptime(self.formatted_stamps.iloc[day + 1]["WAKE - PTCT"], "%d/%m/%Y %H:%M:%S"),
                    y=[0, max(self.subject_object.wrist.epoch.svm) / 4], alpha=.5, color='blue')
            except (ValueError, TypeError, IndexError):
                pass


# ggir = GGIR(file="/Users/kyleweber/Desktop/UPDATED DATA.xlsx", subject_object=x)
