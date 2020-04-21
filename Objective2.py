import LocateUsableParticipants
import pandas as pd
import scipy
import os
import pingouin as pg


usable_subjs = LocateUsableParticipants.SubjectSubset(check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/"
                                                                 "OND07_ProcessingStatus.xlsx",
                                                      wrist_ankle=False, wrist_hr=False,
                                                      wrist_hracc=False, hr_hracc=False,
                                                      ankle_hr=False, ankle_hracc=False,
                                                      wrist_only=False, ankle_only=False,
                                                      hr_only=False, hracc_only=False,
                                                      require_treadmill=True, require_all=True)


class Objective2:

    def __init__(self, data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Kappas_AllData.xlsx'):

        os.chdir("/Users/kyleweber/Desktop/")

        self.data_file = data_file
        self.df = None
        self.oneway_rm_aov = None
        self.ttests_unpaired = None
        self.ttests_paired = None

        """RUNS METHODS"""
        self.load_data()
        self.pairwise_ttests_paired()
        self.pairwise_ttests_unpaired()

    def load_data(self):
        self.df = pd.read_excel(self.data_file, usecols=["ID", "Ankle-Wrist", "Wrist-HR", "Wrist-HRAcc",
                                                         "Ankle-HR", "Ankle-HRAcc", "HR-HRAcc"])

    def check_normality(self):
        for col_name in self.df.keys():
            result = scipy.stats.shapiro(self.df[col_name].dropna())
            print(col_name, ":", "W =", round(result[0], 3), "p =", round(result[1], 3))

    def pairwise_ttests_unpaired(self):

        df = self.df.melt(id_vars="ID")

        self.ttests_unpaired = pg.pairwise_ttests(dv="value", subject='ID',
                                                  between='variable', data=df,
                                                  padjust="bonf", effsize="cohen", parametric=True)

    def pairwise_ttests_paired(self):

        df = self.df.melt(id_vars="ID")

        self.oneway_rm_aov = pg.rm_anova(data=df, dv="value", within="variable", subject='ID')

        self.ttests_paired = pg.pairwise_ttests(dv="value", subject='ID',
                                                within='variable', data=df,
                                                padjust="bonf", effsize="CLES", parametric=True)


# a = Objective2(data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Kappas_AllData.xlsx')
# b = Objective2(data_file='/Users/kyleweber/Desktop/Data/OND07/Processed Data/Kappas_RepeatedOnly.xlsx')
