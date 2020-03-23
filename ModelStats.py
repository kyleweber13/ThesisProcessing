import sklearn


class Stats:

    def __init__(self, subject_object):
        """Class that contains results from statistical analysis."""

        print()
        print("======================================= STATISTICAL ANALYSIS ========================================")

        self.subject_object = subject_object

        if subject_object.valid_all is not None:
            print("\n" + "ALL DEVICES")
            self.kappa_all = self.cohens_kappa(subject_object.valid_all)
        if subject_object.valid_accelonly is not None:
            print("\n" + "ACCELEROMETER-ONLY DATA")
            self.kappa_accelonly = self.cohens_kappa(subject_object.valid_accelonly)

    def cohens_kappa(self, validity_data):
        """Calculates Cohen's kappa for all available model comparisons. Returns results in a dictionary.

        :argument
        -subject_object: class instance of Subject class
        """

        kappa_dict = {"AnkleAccel-WristAccel": None,
                      "AnkleAccel-HR": None,
                      "AnkleAccel-HRAcc": None,
                      "WristAccel-HR": None,
                      "WristAccel-HRAcc": None,
                      "HR-HRAcc": None}

        # Creates data sets excluding Nones
        if self.subject_object.wrist_filepath is not None:
            wrist = [i for i in validity_data.wrist if i is not None]
        if self.subject_object.wrist_filepath is None:
            wrist = None

        if self.subject_object.ankle_filepath is not None:
            ankle = [i for i in validity_data.ankle if i is not None]
        if self.subject_object.ankle_filepath is None:
            ankle = None

        if self.subject_object.ecg_filepath is not None:
            hr = [i for i in validity_data.hr if i is not None]
        if self.subject_object.ecg_filepath is None:
            hr = None

        if self.subject_object.ankle_filepath is not None and self.subject_object.ecg_filepath is not None:
            hr_acc = [i for i in validity_data.hr_acc if i is not None]
        if self.subject_object.ankle_filepath is None or self.subject_object.ecg_filepath is None:
            hr_acc = None

        # Ankle-Wrist comparison
        try:
            kappa = round(sklearn.metrics.cohen_kappa_score(y1=ankle, y2=wrist), 4)
            kappa_dict["AnkleAccel-WristAccel"] = kappa

        except (AttributeError, ValueError, TypeError):
            kappa_dict["AnkleAccel-WristAccel"] = None

        # Ankle-HR comparison
        try:
            kappa = round(sklearn.metrics.cohen_kappa_score(y1=ankle, y2=hr), 4)
            kappa_dict["AnkleAccel-HR"] = kappa

        except (AttributeError, ValueError, TypeError):
            kappa_dict["AnkleAccel-HR"] = None

        # Ankle-HRAcc comparison
        try:
            kappa = round(sklearn.metrics.cohen_kappa_score(y1=ankle, y2=hr_acc), 4)
            kappa_dict["AnkleAccel-HRAcc"] = kappa

        except (AttributeError, ValueError, TypeError):
            kappa_dict["AnkleAccel-HRAcc"] = None

        # Wrist-HR comparison
        try:
            kappa = round(sklearn.metrics.cohen_kappa_score(y1=wrist, y2=hr), 4)
            kappa_dict["WristAccel-HR"] = kappa

        except (AttributeError, ValueError, TypeError):
            kappa_dict["WristAccel-HR"] = None

        # Wrist-HRAcc comparison
        try:
            kappa = round(sklearn.metrics.cohen_kappa_score(y1=wrist, y2=hr_acc), 4)
            kappa_dict["WristAccel-HRAcc"] = kappa

        except (AttributeError, ValueError, TypeError):
            kappa_dict["WristAccel-HRAcc"] = None

        # HR-HRAcc comparison
        try:
            kappa = round(sklearn.metrics.cohen_kappa_score(y1=hr, y2=hr_acc), 4)
            kappa_dict["HR-HRAcc"] = kappa

        except (AttributeError, ValueError, TypeError):
            kappa_dict["HR-HRAcc"] = None

        print("\n" + "Epoch-by-epoch agreement: Cohen's Kappa")
        print("-Ankle-Wrist: {}".format(kappa_dict["AnkleAccel-WristAccel"]))
        print("-Ankle-HR: {}".format(kappa_dict["AnkleAccel-HR"]))
        print("-Ankle-HRAcc: {}".format(kappa_dict["AnkleAccel-HRAcc"]))
        print("-Wrist-HR: {}".format(kappa_dict["WristAccel-HR"]))
        print("-Wrist-HRAcc: {}".format(kappa_dict["WristAccel-HRAcc"]))
        print("-HR-HRAcc: {}".format(kappa_dict["HR-HRAcc"]))

        return kappa_dict
