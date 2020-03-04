import numpy as np


def import_crop_indexes(subject, crop_file):
    """Searches file for existing crop index data. Returns separate dictionaries for start/end indexes."""

    print()
    print("======================================= DEVICE SYNCHRONIZATION ======================================")

    print("\n" + "Searching {} for existing file crop indexes for subject {}...".format(crop_file, subject))
    print()

    # Reads in .csv file
    data = np.loadtxt(fname=crop_file, delimiter=",", skiprows=1, dtype="str")

    # Default values
    crop_indexes_found = False
    start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
    end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

    for row in data:
        # Only looks at data for correct subject
        if str(subject) == row[0]:

            data = []
            empty_tally = 0  # Tallies empty cells

            # Changes values to integers if possible. Empty cells become "N/A"
            # Tallies empty cells --> more than 2 empty cells and the data is not valid.
            for value in row:
                if value == "":
                    value = 0
                    empty_tally += 1
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = value
                data.append(value)

            if empty_tally >= 3:
                crop_indexes_found = False

            if empty_tally <= 2:
                start_offset_dict = {"Ankle": data[1], "Wrist": data[3], "ECG": data[5]}
                end_offset_dict = {"Ankle": data[2], "Wrist": data[4], "ECG": data[6]}
                crop_indexes_found = True

    if crop_indexes_found:
        print("Start offsets: ", start_offset_dict)
        print("End offsets: ", end_offset_dict)
    if not crop_indexes_found:
        print("No indexes found. Checking raw data files..." + "\n")

    return start_offset_dict, end_offset_dict, crop_indexes_found
