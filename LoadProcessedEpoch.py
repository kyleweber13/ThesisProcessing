import numpy as np
from datetime import datetime


def import_processed_accel(GENEActiv_object, processed_data_folder):

    filename = GENEActiv_object.filepath.split("/")[-1].split(".")[0] + "_IntensityData.csv"

    print("\n" + "Imported data processed from {}.".format(GENEActiv_object.filepath))

    # Data import from .csv
    if "Wrist" in filename:
        epoch_timestamps, svm = np.loadtxt(fname=processed_data_folder + filename, delimiter=",", skiprows=1,
                                           usecols=(0, 1), unpack=True, dtype="str")

        stamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in epoch_timestamps]
        epoch_len = (stamps[1] - stamps[0]).seconds
        counts = [float(i) for i in svm]

        return stamps, epoch_len, counts

    if "Ankle" in filename:
        epoch_timestamps, svm, pred_speed, \
        pred_mets, epoch_intensity = np.loadtxt(fname=processed_data_folder + filename, delimiter=",", skiprows=1,
                                                usecols=(0, 1, 2, 3, 4), unpack=True, dtype="str")

        # stamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in epoch_timestamps]
        stamps = [datetime.strptime(i[:-3], "%Y-%m-%dT%H:%M:%S.%f") for i in epoch_timestamps]

        epoch_len = (stamps[1] - stamps[0]).seconds
        counts = [float(i) for i in svm]

        return stamps, epoch_len, counts, pred_speed, pred_mets, epoch_intensity
