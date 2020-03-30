import pandas as pd


def find_usable(check_file=None, require_ecg=False, require_wrist=False, require_ankle=False,
                require_all=False, require_ecg_and_one_accel=False, require_ecg_and_ankle=False,
                require_wrist_and_ankle=False):
    """Function that reads in data from check_file to determine which participants meet the criteria set by
    arguments. NOTE: ONLY SET ONE ARGUMENT TO TRUE

    :returns
    -list of participant IDs that meet criteria
    """

    if require_ecg:
        print("\n" + "Locating all participants with valid ECG data...")

    if require_wrist:
        print("\n" + "Locating all participants with valid wrist data...")

    if require_ankle:
        print("\n" + "Locating all participants with valid ankle data...")

    if require_all:
        print("\n" + "Locating all participants with valid ECG, wrist, and ankle data...")

    if require_ecg_and_one_accel:
        print("\n" + "Locating all participants with valid ECG and at least one accelerometer data...")

    if require_ecg_and_ankle:
        print("\n" + "Locating all participants with valid ECG and ankle data...")

    if require_wrist_and_ankle:
        print("\n" + "Locating all participants with valid wrist and ankle data...")

    data_file = pd.read_excel(io=check_file, header=0,
                              usecols=["ID", "Wrist_usable_alone", "Ankle_usable_alone", "HRAcc_usable",
                                       "ECG_and_one_usable", "ECG_usable_alone", "AccelsOnly_usable",
                                       "All_usable", "All_dur_valid"])

    usable_list = []

    for i in range(data_file.shape[0]):

        # All participants with valid ECG data: no accelerometer requirements
        if require_ecg and data_file.iloc[i]["ECG_usable_alone"] == int(require_ecg):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid wrist data: no other requirements
        if require_wrist and data_file.iloc[i]["Wrist_usable_alone"] == int(require_wrist):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ankle data: no other requirements
        if require_ankle and data_file.iloc[i]["Ankle_usable_alone"] == int(require_ankle):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ECG, wrist, and ankle data: requires all three
        if require_all and data_file.iloc[i]["All_usable"] == int(require_all):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ECG and either wrist/ankle data: only requires one accelerometer
        if require_ecg_and_one_accel and data_file.iloc[i]["ECG_and_one_usable"] == int(require_ecg_and_one_accel):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ECG and ankle data: no wrist requirement
        if require_ecg_and_ankle and data_file.iloc[i]["HRAcc_usable"] == int(require_ecg_and_ankle):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid wrist and ankle data: no ECG requirement
        if require_wrist_and_ankle and data_file.iloc[i]["AccelsOnly_usable"] == int(require_wrist_and_ankle):
            usable_list.append(data_file.iloc[i]["ID"])

    print("Found {} participants that meet criteria.".format(len(usable_list)))

    usable_list = [i.split("_")[2] for i in usable_list]

    # Returns unique values as a list
    usable_list = [i for i in set(usable_list)]

    return usable_list
