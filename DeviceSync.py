import pyedflib
from datetime import timedelta


def crop_start(subject_object):
    """Function that checks device starttimes and calculates the number of data points to skip at the start of the file
       so all devices begin at the same time (within 1 second; EDF resolution).

    -Checks which files exist and which starts first --> calculates values accordingly.

    :argument
    -subject_object: object of Subject class

    :returns
    -start_crop_dict: dictionary of values for each device that correspond to number of data points to skip.
    """

    # Skips device synchronization if only one device is loaded
    if subject_object.load_ecg + subject_object.load_ankle + subject_object.load_wrist < 2:
        start_crop_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

        return start_crop_dict

    # Booleans for whether data exists or not
    ankle_exists = subject_object.ankle_filepath is not None
    wrist_exists = subject_object.wrist_filepath is not None
    ecg_exists = subject_object.ecg_filepath is not None

    # Loads EDF file headers to retrieve sample rates and start times
    if ankle_exists:
        ankle_file = pyedflib.EdfReader(subject_object.ankle_filepath)
        ankle_samplerate = ankle_file.getSampleFrequencies()[1]
        ankle_start = ankle_file.getStartdatetime()
        ankle_duration = ankle_file.getFileDuration()
    if not ankle_exists:
        ankle_samplerate = 1

    if wrist_exists:
        wrist_file = pyedflib.EdfReader(subject_object.wrist_filepath)
        wrist_samplerate = wrist_file.getSampleFrequencies()[1]
        wrist_start = wrist_file.getStartdatetime()
        wrist_duration = wrist_file.getFileDuration()

    if not wrist_exists:
        wrist_samplerate = 1

    if ecg_exists:
        ecg_file = pyedflib.EdfReader(subject_object.ecg_filepath)
        ecg_samplerate = ecg_file.getSampleFrequencies()[0]
        ecg_start = ecg_file.getStartdatetime()
        ecg_duration = ecg_file.getFileDuration()
    if not ecg_exists:
        ecg_samplerate = 1

    # Values by which data is offset (indexes): default values
    ankle_start_offset = 0
    wrist_start_offset = 0
    ecg_start_offset = 0

    # Checks which device started last

    # IF ALL DEVICES ARE USED ========================================================================================
    if ankle_exists and wrist_exists and ecg_exists:

        # If already synced at start
        if ankle_start == wrist_start == ecg_start:
            ankle_start_offset, wrist_start_offset, ecg_start_offset = 0, 0, 0

        # Wrist first
        if ankle_start <= wrist_start and ecg_start <= wrist_start:

            ecg_start_offset = ((wrist_start - ecg_start).days * 86400 +
                                (wrist_start - ecg_start).seconds) * ecg_samplerate

            ankle_start_offset = ((wrist_start - ankle_start).days * 86400 +
                                  (wrist_start - ankle_start).seconds) * ankle_samplerate

        # Ankle first
        if wrist_start <= ankle_start and ecg_start <= ankle_start:

            ecg_start_offset = ((ankle_start - ecg_start).days * 86400 +
                                (ankle_start - ecg_start).seconds) * ecg_samplerate

            wrist_start_offset = ((ankle_start - wrist_start).days * 86400 +
                                  (ankle_start - wrist_start).seconds) * wrist_samplerate

        # ECG first
        if wrist_start <= ecg_start and ankle_start <= ecg_start:

            wrist_start_offset = ((ecg_start - wrist_start).days * 86400 +
                                  (ecg_start - wrist_start).seconds) * wrist_samplerate

            ankle_start_offset = ((ecg_start - ankle_start).days * 86400 +
                                  (ecg_start - ankle_start).seconds) * ankle_samplerate

    # IF ONLY WRIST AND ANKLE ARE USED ===============================================================================
    if ankle_exists and wrist_exists and not ecg_exists:

        # If ankle started first
        if ankle_start < wrist_start:

            ankle_start_offset = ((wrist_start - ankle_start).days * 86400 +
                                  (wrist_start - ankle_start).seconds) * ankle_samplerate

        # If wrist started first
        if wrist_start < ankle_start:

            wrist_start_offset = ((ankle_start - wrist_start).days * 86400 +
                                  (ankle_start - wrist_start).seconds) * wrist_samplerate

    # IF ONLY ECG AND ANKLE ARE USED =================================================================================
    if ankle_exists and ecg_exists and not wrist_exists:

        # If ankle started first
        if ankle_start < ecg_start:

            ankle_start_offset = ((ecg_start - ankle_start).days * 86400 +
                                  (ecg_start - ankle_start).seconds) * ankle_samplerate

        # If ECG started first
        if ecg_start < ankle_start:

            ecg_start_offset = ((ankle_start - ecg_start).days * 86400 +
                                (ankle_start - ecg_start).seconds) * ecg_samplerate

    # IF ONLY ECG AND WRIST ARE USED =================================================================================
    if wrist_exists and ecg_exists and not ankle_exists:

        # If wrist started first
        if wrist_start < ecg_start:

            wrist_start_offset = ((ecg_start - wrist_start).days * 86400 +
                                  (ecg_start - wrist_start).seconds) * wrist_samplerate

        # If ECG started first
        if ecg_start < wrist_start:

            ecg_start_offset = ((wrist_start - ecg_start).days * 86400 +
                                (wrist_start - ecg_start).seconds) * ecg_samplerate

    start_crop_dict = {"Ankle": ankle_start_offset,
                       "Wrist": wrist_start_offset,
                       "ECG": ecg_start_offset}

    return start_crop_dict


def crop_end(subject_object):
    """Function that determines how many data points to read in so that all files are the same duration.

    :returns
    -end_crop_dict: dictionary of values for each device of how many data points to read in
                    so the files are the same duration

    :argument
    -subject_object: object of class Subject. Needs to contain start_offset_dict from crop_start function.
    """

    # Skips device synchronization if only one device is loaded
    if subject_object.load_ecg + subject_object.load_ankle + subject_object.load_wrist < 2:
        end_crop_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

        return end_crop_dict

    # Booleans for whether data exists or not
    ankle_exists = subject_object.ankle_filepath is not None
    wrist_exists = subject_object.wrist_filepath is not None
    ecg_exists = subject_object.ecg_filepath is not None

    # Start time offset dictionary
    start_offset_dict = subject_object.start_offset_dict

    # Loads EDF file headers to retrieve sample rates and start times
    if ankle_exists:
        ankle_file = pyedflib.EdfReader(subject_object.ankle_filepath)
        ankle_samplerate = ankle_file.getSampleFrequencies()[1]
        ankle_start = ankle_file.getStartdatetime() + timedelta(seconds=start_offset_dict["Ankle"] / ankle_samplerate)
        ankle_duration = ankle_file.getFileDuration()
        ankle_end = ankle_start + timedelta(seconds=ankle_duration)
    if not ankle_exists:
        ankle_samplerate = 1
        ankle_duration = 0
        ankle_start = None
        ankle_end = None

    if wrist_exists:
        wrist_file = pyedflib.EdfReader(subject_object.wrist_filepath)
        wrist_samplerate = wrist_file.getSampleFrequencies()[1]
        wrist_start = wrist_file.getStartdatetime() + timedelta(seconds=start_offset_dict["Wrist"] / wrist_samplerate)
        wrist_duration = wrist_file.getFileDuration()
        wrist_end = wrist_start + timedelta(seconds=wrist_duration)

    if not wrist_exists:
        wrist_samplerate = 1
        wrist_duration = 0
        wrist_start = None
        wrist_end = None

    if ecg_exists:
        ecg_file = pyedflib.EdfReader(subject_object.ecg_filepath)
        ecg_samplerate = ecg_file.getSampleFrequencies()[0]
        ecg_start = ecg_file.getStartdatetime() + timedelta(seconds=start_offset_dict["ECG"] / ecg_samplerate)
        ecg_duration = ecg_file.getFileDuration()
        ecg_end = ecg_start + timedelta(seconds=ecg_duration)
    if not ecg_exists:
        ecg_samplerate = 1
        ecg_duration = 0

    # Values by which data is cropped at the end (indexes): default values
    ankle_end_offset = 0
    wrist_end_offset = 0
    ecg_end_offset = 0

    # IF ALL DEVICES ARE USED ========================================================================================
    if ankle_exists and wrist_exists and ecg_exists:

        # Wrist ends first
        if ankle_end >= wrist_end and ecg_end >= wrist_end:
            ankle_end_offset = ((ankle_end - wrist_end).days * 86400 +
                                (ankle_end - wrist_end).seconds) * ankle_samplerate

            ecg_end_offset = ((ecg_end - wrist_end).days * 86400 + (ecg_end - wrist_end).seconds) * ecg_samplerate

        # Ankle ends first
        if wrist_end >= ankle_end and ecg_end >= ankle_end:

            ecg_end_offset = ((ecg_end - ankle_end).days * 86400 + (ecg_end - ankle_end).seconds) * ecg_samplerate

            wrist_end_offset = ((wrist_end - ankle_end).days * 86400 +
                                (wrist_end - ankle_end).seconds) * wrist_samplerate

        # ECG ends first
        if wrist_end >= ecg_end and ankle_end >= ecg_end:

            wrist_end_offset = ((wrist_end - ecg_end).days * 86400 + (wrist_end - ecg_end).seconds) * wrist_samplerate

            ankle_end_offset = ((ankle_end - ecg_end).days * 86400 + (ankle_end - ecg_end).seconds) * ankle_samplerate

    # IF ANKLE AND WRIST USED =========================================================================================

    if ankle_exists and wrist_exists and not ecg_exists:

        # Wrist ends first
        if ankle_end >= wrist_end:

            ankle_end_offset = ((ankle_end - wrist_end).days * 86400 +
                                (ankle_end - wrist_end).seconds) * ankle_samplerate

        # Ankle ends first
        if wrist_end >= ankle_end:

            wrist_end_offset = ((wrist_end - ankle_end).days * 86400 +
                                (wrist_end - ankle_end).seconds) * wrist_samplerate

    # IF ANKLE AND ECG USED ===========================================================================================

    if ankle_exists and ecg_exists and not wrist_exists:

        # ECG ends first
        if ankle_end >= ecg_end:

            ankle_end_offset = ((ankle_end - ecg_end).days * 86400 + (ankle_end - ecg_end).seconds) * ankle_samplerate

        # Ankle ends first
        if ecg_end >= ankle_end:

            ecg_end_offset = ((ecg_end - ankle_end).days * 86400 + (ecg_end - ankle_end).seconds) * ecg_samplerate

    # IF WRIST AND ECG USED ===========================================================================================

    if wrist_exists and ecg_exists and not ankle_exists:

        # ECG ends first
        if wrist_end >= ecg_end:

            wrist_end_offset = ((wrist_end - ecg_end).days * 86400 + (wrist_end - ecg_end).seconds) * wrist_samplerate

        # Wrist ends first
        if ecg_end >= wrist_end:

            ecg_end_offset = ((ecg_end - wrist_end).days * 86400 + (ecg_end - wrist_end).seconds) * ecg_samplerate

    # FINAL VALUES ===================================================================================================

    # Values correspond to number of data points to read in
    # Number of all data points (duration in seconds * sample rate - offset from end)
    end_crop_dict = {"Ankle": ankle_duration * ankle_samplerate - ankle_end_offset,
                     "Wrist": wrist_duration * wrist_samplerate - wrist_end_offset,
                     "ECG": ecg_duration * ecg_samplerate - ecg_end_offset}

    return end_crop_dict
