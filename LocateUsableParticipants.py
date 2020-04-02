import pandas as pd


def find_usable(check_file=None, require_ecg=False, require_wrist=False, require_ankle=False, require_treadmill=False,
                require_all=False, require_ecg_and_one_accel=False, require_ecg_and_ankle=False,
                require_wrist_and_ankle=False):
    """Function that reads in data from check_file to determine which participants meet the criteria set by
    arguments. NOTE: ONLY SET ONE ARGUMENT TO TRUE

    :returns
    -list of participant IDs that meet criteria
    """

    if require_ecg:
        print("\n" + "Locating all participants with valid ECG data (≥ 30 total valid hours)...")

    if require_wrist:
        print("\n" + "Locating all participants with valid wrist data (≥ 30 total waking hours)...")

    if require_ankle:
        print("\n" + "Locating all participants with valid ankle data (≥ 30 total waking hours)...")

    if require_treadmill:
        print("\n" + "Locating all participants who performed the treadmill protocol...")

    if require_all:
        print("\n" + "Locating all participants with valid ECG, wrist, and ankle data (w/ treadmill protocol)...")

    if require_ecg_and_one_accel:
        print("\n" + "Locating all participants with valid ECG and at least "
                     "one accelerometer data (≥ 30 total valid hours)...")

    if require_ecg_and_ankle and require_treadmill:
        print("\n" + "Locating all participants with valid ECG and ankle data (w/ treadmill protocol)...")

    if require_ecg_and_ankle and not require_treadmill:
        print("\n" + "Locating all participants with valid ECG and ankle data (with or without treadmill protocol)...")

    if require_wrist_and_ankle and require_treadmill:
        print("\n" + "Locating all participants with valid wrist and ankle data (w/ treadmill protocol...")

    if require_wrist_and_ankle and not require_treadmill:
        print("\n" + "Locating all participants with valid wrist and ankle data (with or without treadmill protocol...")

    data_file = pd.read_excel(io=check_file, header=0,
                              usecols=["ID", "Treadmill_performed", "Wrist_usable_alone", "Ankle_usable_alone_noTM",
                                       "Ankle_usable_alone_TM",
                                       "HRAcc_usable", "ECG_and_one_usable", "ECG_usable_alone",
                                       "AccelsOnly_usable", "All_usable", "All_dur_valid"])

    usable_list = []

    for i in range(data_file.shape[0]):

        # All participants with valid ECG data: no accelerometer requirements
        if require_ecg and data_file.iloc[i]["ECG_usable_alone"] == int(require_ecg):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid wrist data: no other requirements
        if require_wrist and data_file.iloc[i]["Wrist_usable_alone"] == int(require_wrist):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ankle data: no other requirements
        if require_ankle and data_file.iloc[i]["Ankle_usable_alone_noTM"] == int(require_ankle):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ankle data AND treadmill protocol: no other requirements
        if require_treadmill and data_file.iloc[i]["Treadmill_performed"] == int(require_treadmill):
            usable_list.append(data_file.iloc[i]["ID"])

        # All participants with valid ECG, wrist, and ankle data (w/ treadmill): requires all three
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

    print("Found {} participants that meet criteria.".format(len(set(usable_list))))

    usable_list = [i.split("_")[2] for i in set(usable_list)]

    # Returns unique values as a list
    usable_list = sorted([int(i) for i in set(usable_list)])

    return usable_list


class SubjectSubset:

    def __init__(self, check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/OND07_ProcessingStatus.xlsx",
                 wrist_ankle=False, wrist_hr=False, wrist_hracc=False, hr_hracc=False,
                 ankle_hr=False, ankle_hracc=False, require_treadmill=False,
                 wrist_only=False, ankle_only=False, hr_only=False, hracc_only=False,
                 require_all=False):

        self.check_file = check_file
        self.wrist_ankle = wrist_ankle
        self.wrist_hr = wrist_hr
        self.wrist_hracc = wrist_hracc
        self.hr_hracc = hr_hracc
        self.ankle_hr = ankle_hr
        self.ankle_hracc = ankle_hracc
        self.require_treadmill = require_treadmill  # WHETHER INDIVIDUAL TREADMILL PROTOCOL REQUIRED
        self.wrist_only = wrist_only
        self.ankle_only = ankle_only
        self.hr_only = hr_only
        self.hracc_only = hracc_only
        self.require_all = require_all  # Wrist, ankle w/ treadmill, HR, HR-Acc

        self.data = None
        self.performed_treadmill = []
        self.participant_list = []

        self.import_data()
        self.find_participants()

    def import_data(self):
        self.data = pd.read_excel(io=self.check_file, header=0,
                                  usecols=["ID", "Wrist_file", "Ankle_file",
                                           "ECG_dur_valid", "Treadmill_performed", "Accelonly_dur_valid",
                                           "All_dur_valid", "All_usable"])

    def find_participants(self):

        for i in range(self.data.shape[0]):

            # ======================================== PERFORMED TREADMILL ============================================

            # Who performed treadmill
            if self.data.iloc[i]["Treadmill_performed"] == 1:
                self.performed_treadmill.append(self.data.iloc[i]["ID"])

            # ========================================= INDIVIDUAL SENSORS ============================================

            # Wrist only
            if self.wrist_only:
                if self.data.iloc[i]["Wrist_file"] == int(self.wrist_only):
                    self.participant_list.append(self.data.iloc[i]["ID"])

            # Ankle only
            if self.ankle_only:
                if self.data.iloc[i]["Ankle_file"] == int(self.ankle_only):
                    self.participant_list.append(self.data.iloc[i]["ID"])

            # HR only
            if self.hr_only:
                if self.data.iloc[i]["ECG_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

            # HR-Acc only
            if self.hracc_only:
                # Requires ankle file and more than 30 hours valid ECG + ankle data
                if self.data.iloc[i]["Ankle_file"] == 1 and self.data.iloc[i]["All_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

        # ============================================ ALL DEVICES ====================================================

            if self.require_all:
                # Requires wrist and ankle files and more than 30 hours valid ECG + wrist + ankle data
                if self.data.iloc[i]["Ankle_file"] == 1 and \
                        self.data.iloc[i]["Wrist_file"] == 1 and \
                        self.data.iloc[i]["All_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

        # ========================================= CHOOSE TWO DEVICES ================================================

            # Wrist and Ankle comparison
            if self.wrist_ankle:
                # Requires ≥ 30 hours of wrist + ankle data
                if self.data.iloc[i]["Ankle_file"] == 1 and \
                        self.data.iloc[i]["Wrist_file"] == 1 and \
                        self.data.iloc[i]["Accelonly_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

            # Wrist and HR comparison
            if self.wrist_hr:
                # Requires ≥ 30 hours of wrist + HR data
                if self.data.iloc[i]["Wrist_file"] == 1 and self.data.iloc[i]["All_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

            # Wrist and HR-Acc comparison
            if self.wrist_hracc:
                # Requires wrist file, ankle file, and ≥ 30 hours valid wrist + ankle + HR data
                if self.data.iloc[i]["Wrist_file"] == 1 and self.data.iloc[i]["Ankle_file"] == 1 and \
                        self.data.iloc[i]["All_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

            # Ankle and HR OR Ankle and HR-Acc comparison OR HR and HR-Acc (same requirements)
            if self.ankle_hr or self.ankle_hracc or self.hr_hracc:
                # Requires ankle file and ≥ 30 hours valid ankle + HR data
                if self.data.iloc[i]["Ankle_file"] == 1 and self.data.iloc[i]["All_dur_valid"] >= 30:
                    self.participant_list.append(self.data.iloc[i]["ID"])

        # ====================================== CHECK FOR REQUIRE_TREADMILL ==========================================

        # Removes participants who did not perform treadmill if it was required
        if self.require_treadmill:

            return_list = []

            for participant in self.participant_list:
                if participant in self.performed_treadmill:
                    return_list.append(participant)

            self.participant_list = return_list

        print("\nFound {} participants that meet criteria.".format(len(self.participant_list)))
        print(self.participant_list)
