import csv


class HRAcc:

    def __init__(self, subject_object=None, use_ankle_during_invalid_hr=False):
        """Class that uses data from subject_object.ecg and subject_object.ankle to combine data into a HR-Acc model.

        :argument
        -subject_object: object of class Subject
        -hrr_threshold: threshold as %HRR that marks use of ankle vs. HR data
        """

        self.output_dir = subject_object.output_dir
        self.subjectID = subject_object.subjectID
        self.filename = subject_object.ecg.filename[:-3]
        self.write_results = subject_object.write_results
        self.epoch_len = subject_object.epoch_len

        self.hrr_threshold = subject_object.hracc_threshold
        self.use_ankle_during_invalid_hr = use_ankle_during_invalid_hr

        self.ankle = subject_object.ankle
        self.hr = subject_object.ecg

        self.ankle_intensity = self.ankle.model.epoch_intensity
        self.hr_intensity = self.hr.epoch_intensity
        self.perc_hrr = self.hr.perc_hrr

        # Converts list of 1 (invalid) and 0 (valid) to boolean values: True == valid
        self.hr_validity = [not(bool(i)) for i in subject_object.ecg.epoch_validity]

        self.model = HRAccModel(hracc_object=self, use_ankle_during_invalid_hr=self.use_ankle_during_invalid_hr)


class HRAccModel:

    def __init__(self, hracc_object=None, use_ankle_during_invalid_hr=False):
        """Class that creates HR-Acc model."""

        self.hracc = hracc_object
        self.use_ankle_during_invalid_hr = use_ankle_during_invalid_hr
        self.epoch_intensity = None
        self.model_used = []

        self.combine_models(self.use_ankle_during_invalid_hr)
        self.intensity_totals, self.model_usage = self.calculate_intensity_totals()

    def combine_models(self, use_ankle_during_invalid_hr):

        self.epoch_intensity = []

        for hrr, hr_int, ankle_int in zip(self.hracc.perc_hrr, self.hracc.hr_intensity, self.hracc.ankle_intensity):

            if hrr is None:
                if use_ankle_during_invalid_hr:
                    self.epoch_intensity.append(ankle_int)
                    self.model_used.append("Ankle")
                if not use_ankle_during_invalid_hr:
                    self.epoch_intensity.append(None)
                    self.model_used.append("Invalid ECG")

            if hrr is not None:
                if hrr >= self.hracc.hrr_threshold:
                    self.epoch_intensity.append(hr_int)
                    self.model_used.append("HR")

                if hrr < self.hracc.hrr_threshold:
                    self.epoch_intensity.append(ankle_int)
                    self.model_used.append("Ankle")

    def calculate_intensity_totals(self):

        intensity = [i for i in self.epoch_intensity if i is not None]

        if self.use_ankle_during_invalid_hr:
            n_valid_epochs = len(intensity)
        if not self.use_ankle_during_invalid_hr:
            n_valid_epochs = len(intensity) - intensity.count(None)

        # Calculates time spent in each intensity category
        intensity_totals = {"Sedentary": intensity.count(0) / (60 / self.hracc.epoch_len),
                            "Sedentary%": round(intensity.count(0) / n_valid_epochs, 3),
                            "Light": intensity.count(1) / (60 / self.hracc.epoch_len),
                            "Light%": round(intensity.count(1) / n_valid_epochs, 3),
                            "Moderate": intensity.count(2) / (60 / self.hracc.epoch_len),
                            "Moderate%": round(intensity.count(2) / n_valid_epochs, 3),
                            "Vigorous": intensity.count(3) / (60 / self.hracc.epoch_len),
                            "Vigorous%": round(intensity.count(3) / n_valid_epochs, 3)}

        print("\n" + "HEART RATE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        model_usage_dict = {"Ankle epochs": self.model_used.count("Ankle"),
                            "Ankle %)": round(self.model_used.count("Ankle") / n_valid_epochs, 5),
                            "HR epochs": self.model_used.count("HR"),
                            "HR %": round(self.model_used.count("HR") / n_valid_epochs, 5)}

        return intensity_totals, model_usage_dict

    def write_output(self):
        """Writes csv of epoched timestamps, validity category."""

        with open(self.hracc.output_dir + "Model Output/" +
                  self.hracc.filename + "_HRAcc_IntensityData.csv", "w") as outfile:
            writer = csv.writer(outfile, delimiter=',', lineterminator="\n")

            writer.writerow(["Timestamp", "ValidHR", "ModelUsed", "IntensityCategory"])
            writer.writerows(zip(self.hracc.hr.epoch_timestamps, self.hracc.hr_validity,
                                 self.model_used, self.epoch_intensity))

        print("\n" + "Complete. File {} saved.".format(self.hracc.output_dir + "Model Output/" +
                                                       self.hracc.filename + "_HRAcc_IntensityData.csv"))
