import numpy as np


def import_demographics(subjectID, demos_filename):
    """Imports age and sex demographics for participant."""

    data = np.loadtxt(fname=demos_filename, delimiter=",", skiprows=1, usecols=(0, 4, 6), dtype="str")

    for row in data:
        if subjectID in row[0]:
            return int(row[1]), row[2]


def set_rvo2(age, sex):
    """Sets Kwan et al. (2004) resting VO2 value based on age and sex."""

    if int(age) < 65 and sex == "Male":
        rvo2 = 3.03
    if int(age) < 65 and sex == "Female":
        rvo2 = 3.32
    if int(age) >= 65 and sex == "Male":
        rvo2 = 2.84
    if int(age) >= 65 and sex == "Female":
        rvo2 = 2.82

    return rvo2
