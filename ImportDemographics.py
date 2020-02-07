import numpy as np


def import_demographics(demographics_file, subjectID):
    """Function that imports demographics data from spreadsheet for desired participants.

    :returns
    -demos_dict: dictionary containing demographics information
    """

    data = np.loadtxt(fname=demographics_file, delimiter=",", skiprows=1, dtype="str")

    for row in data:
        if str(subjectID) in row[0]:

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

    return demos_dict
