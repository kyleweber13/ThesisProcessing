import pyedflib


def check_starttimes(subject_object):
    """Function that checks device starttimes and calculates raw data offset index
    to ensure devices start at the same time.

    :argument
    -starttime_dict: dictionary that contains starttimes for wrist, ankle, and ECG devices

    :returns
    -wrist_index_offset, ankle_index_offset: number of data points by which device should be offset
    """

    print()
    print("======================================= DEVICE SYNCRONIZATION =======================================")

    print("\n" + "Performing device sync...")

    # Booleans for whether data exists or not
    ankle_exists = subject_object.ankle_filepath is not None
    wrist_exists = subject_object.wrist_filepath is not None
    ecg_exists = subject_object.ecg_filepath is not None

    # Loads EDF file headers to retrieve sample rates and start times
    if ankle_exists:
        ankle_file = pyedflib.EdfReader(subject_object.ankle_filepath)
        ankle_samplerate = ankle_file.getSampleFrequencies()[1]
        ankle_start = ankle_file.getStartdatetime()
    if not ankle_exists:
        ankle_samplerate = 1

    if wrist_exists:
        wrist_file = pyedflib.EdfReader(subject_object.wrist_filepath)
        wrist_samplerate = wrist_file.getSampleFrequencies()[1]
        wrist_start = wrist_file.getStartdatetime()
    if not wrist_exists:
        wrist_samplerate = 1

    if ecg_exists:
        ecg_file = pyedflib.EdfReader(subject_object.ecg_filepath)
        ecg_samplerate = ecg_file.getSampleFrequencies()[0]
        ecg_start = ecg_file.getStartdatetime()
    if not ecg_exists:
        ecg_samplerate = 1

    # Values by which data is offset (indexes): default values
    ankle_index_offset = 0
    wrist_index_offset = 0
    ecg_index_offset = 0

    # Checks which device started last

    # IF ALL DEVICES ARE USED ========================================================================================
    if ankle_exists and wrist_exists and ecg_exists:

        # If already synced
        if ankle_start == wrist_start == ecg_start:
            print("-Devices synced.")
            ankle_index_offset, wrist_index_offset, ecg_index_offset = 0, 0, 0

            return ankle_index_offset, wrist_index_offset, ecg_index_offset

        # Wrist first
        if ankle_start <= wrist_start and ecg_start <= wrist_start:
            ecg_index_offset = (wrist_start - ecg_start).seconds * ecg_samplerate
            ankle_index_offset = (wrist_start - ankle_start).seconds * ankle_samplerate
            print("-Wrist started first." + "\n" + "-Ankle offset = {}; ECG offset = {}.".format(ankle_index_offset,
                                                                                                 ecg_index_offset))

        # Ankle first
        if wrist_start <= ankle_start and ecg_start <= ankle_start:
            ecg_index_offset = (ankle_start - ecg_start).seconds * ecg_samplerate
            wrist_index_offset = (ankle_start - wrist_start).seconds * wrist_samplerate
            print("-Ankle started first." + "\n" + "-Wrist offset = {}; ECG offset = {}.".format(wrist_index_offset,
                                                                                                 ecg_index_offset))

        # ECG first
        if wrist_start <= ecg_start and ankle_start <= ecg_start:
            wrist_index_offset = (ecg_start - wrist_start).seconds * wrist_samplerate
            ankle_index_offset = (ecg_start - ankle_start).seconds * ankle_samplerate
            print("-ECG started first." + "\n" + "-Wrist offset = {}; ankle offset = {}.".format(wrist_index_offset,
                                                                                                 ankle_index_offset))

    # IF ONLY WRIST AND ANKLE ARE USED ===============================================================================
    if ankle_exists and wrist_exists and not ecg_exists:

        # If ankle started first
        if ankle_start < wrist_start:
            print("-Ankle started before wrist.")

            ankle_index_offset = (wrist_start - ankle_start).seconds * ankle_samplerate

        # If wrist started first
        if wrist_start < ankle_start:
            print("-Wrist started before ankle.")

            wrist_index_offset = (ankle_start - wrist_start).seconds * wrist_samplerate

    # IF ONLY ECG AND ANKLE ARE USED =================================================================================
    if ankle_exists and ecg_exists and not wrist_exists:

        # If ankle started first
        if ankle_start < ecg_start:
            print("-Ankle started before ECG.")

            ankle_index_offset = (ecg_start - ankle_start).seconds * ankle_samplerate

        # If ECG started first
        if ecg_start < ankle_start:
            print("-ECG started before ankle.")

            ecg_index_offset = (ankle_start - ecg_start).seconds * ecg_samplerate

    # IF ONLY ECG AND WRIST ARE USED =================================================================================
    if wrist_exists and ecg_exists and not ankle_exists:

        # If wrist started first
        if wrist_start < ecg_start:
            print("-Wrist started before ECG.")

            wrist_index_offset = (ecg_start - wrist_start).seconds * wrist_samplerate

        # If ECG started first
        if ecg_start < wrist_start:
            print("-ECG started before wrist.")

            ecg_index_offset = (wrist_start - ecg_start).seconds * ecg_samplerate

    offset_dict = {"Ankle": ankle_index_offset, "Wrist": wrist_index_offset, "ECG": ecg_index_offset}

    print("Index offsets:", offset_dict)

    return offset_dict
