from Subject import Subject
import ModelStats
import FindValidEpochs
import ImportEDF

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


x = Subject(raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
            subjectID=3030,
            load_ecg=True, load_ankle=True, load_wrist=True,
            load_raw_ecg=True, load_raw_ankle=True, load_raw_wrist=True,
            from_processed=False,
            rest_hr_window=30,
            n_epochs_rest_hr=30,
            filter_ecg=True,
            epoch_len=15,
            crop_index_file="/Users/kyleweber/Desktop/Data/OND07/CropIndexes_All.csv",
            treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Treadmill_Log.csv",
            demographics_file="/Users/kyleweber/Desktop/Data/OND07/Participant Information/Demographics_Data.csv",
            sleeplog_folder="/Users/kyleweber/Desktop/Data/OND07/Sleep Logs/",
            output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
            write_results=False,
            plot_data=False)


x.valid = FindValidEpochs.ValidData(subject_object=x, write_results=x.write_results)
x.stats = ModelStats.Stats(subject_object=x)
"""x.valid.write_validity_report()
x.valid.write_valid_epochs()

x.ankle.treadmill.plot_treadmill_protocol(x.ankle)
x.valid.plot_validity_data()"""

"""x.wrist.write_model()
x.ankle.model.write_anklemodel()
x.ecg.write_output()"""

w_start, w_end = ImportEDF.check_file(x.wrist_filepath)
a_start, a_end = ImportEDF.check_file(x.ankle_filepath)
e_start, e_end = ImportEDF.check_file(x.ecg_filepath)