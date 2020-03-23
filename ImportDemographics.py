import numpy as np
import os


def import_demographics(subject_object=None):
    """Function that imports demographics data from spreadsheet for desired participants.

    :returns
    -demos_dict: dictionary containing demographics information
    """

    demos_file = subject_object.demographics_file

    if demos_file is None:
        print("No demographics file input.")
        return None
    if not os.path.exists(demos_file):
        print("Demographics file does not exist.")
        return None

    data = np.loadtxt(fname=demos_file, delimiter=",", skiprows=1, dtype="str")

    for row in data:
        if str(subject_object.subjectID) in row[0]:

            # Sets resting VO2 according to Kwan et al. (2004) values based on age/sex
            age = int(row[4])
            sex = row[6]

            if int(age) < 65 and sex == "Male":
                rvo2 = 3.03
            if int(age) < 65 and sex == "Female":
                rvo2 = 3.32
            if int(age) >= 65 and sex == "Male":
                rvo2 = 2.84
            if int(age) >= 65 and sex == "Female":
                rvo2 = 2.82

            demos_dict = {"Age": int(row[4]),
                          "Sex": row[6],
                          "Weight": float(row[7]),
                          "Height": float(row[8]),
                          "Hand": row[9],
                          "RestVO2": rvo2}

    subject_object.demographics = demos_dict
