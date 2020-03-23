from Subject import Subject
import ModelStats
import ValidData
import ImportEDF
import HRAccTesting
import LocateUsableParticipants
import DailyReports

import random
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
warnings.filterwarnings("ignore")


"""
usable_subjs = LocateUsableParticipants.find_usable(check_file="/Users/kyleweber/Desktop/Data/OND07/"
                                                               "Tabular Data/OND07_ProcessingStatus.xlsx",
                                                    require_ecg=False, require_wrist=False, require_ankle=False,
                                                    require_all=False, require_ecg_and_one_accel=False,
                                                    require_ecg_and_ankle=True)

rand_part = random.choice(usable_subjs)
"""

x = Subject(raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
            subjectID=3043,
            load_ecg=True, load_ankle=True, load_wrist=True,
            load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,

            from_processed=True,

            rest_hr_window=30,
            n_epochs_rest_hr=30,
            hracc_threshold=24,

            filter_ecg=True,
            epoch_len=15,
            crop_index_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/CropIndexes_All.csv",
            treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",
            demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",
            sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/",
            output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
            write_results=False)
