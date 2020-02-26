import sklearn


class Stats:

    def __init__(self, subject_object):
        """Class that contains results from statistical analysis."""

        print()
        print("======================================= STATISTICAL ANALYSIS ========================================")

        self.subject_object = subject_object

        self.kappa = self.cohens_kappa()

    def cohens_kappa(self):
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
        wrist = [i for i in self.subject_object.valid.wrist if i is not None]
        ankle = [i for i in self.subject_object.valid.ankle if i is not None]
        hr = [i for i in self.subject_object.valid.hr if i is not None]
        # hr_acc = [i for i in self.subject_object.valid.hr_acc if i is not None]
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
