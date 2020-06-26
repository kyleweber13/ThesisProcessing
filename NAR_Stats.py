import pandas as pd
import os
import pingouin as pg
import scipy.stats
import matplotlib.pyplot as plt


class Data:

    def __init__(self, data_folder=None):

        self.data_folder = data_folder

        self.files = self.get_filenames()

        self.df_activity = self.create_df()

        self.activity_anova = self.activity_stats()
        self.activity_stats = self.activity_descriptive_stats()

    def get_filenames(self):

        filenames = [i for i in os.listdir(self.data_folder) if "csv" in i]

        return filenames

    def create_df(self):

        for i, file in enumerate(self.files):
            if i == 0:
                output_df = pd.read_csv(filepath_or_buffer=self.data_folder + file, delimiter=",")
                output_df.insert(0, "ID", [int(file.split("_")[2]) for i in range(4)])

            if i > 1:
                df = pd.read_csv(filepath_or_buffer=self.data_folder + file, delimiter=",")
                df["ID"] = [int(file.split("_")[2]) for i in range(4)]

                output_df = output_df.append(df)

        output_df["MVPA"] = output_df["Moderate"] + output_df["Vigorous"]
        output_df["MVPA%"] = output_df["Moderate%"] + output_df["Vigorous%"]

        return output_df

    def activity_stats(self, data_type='percent'):

        if data_type == 'percent':
            intensity_list = ["Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]
            df = self.df_activity[["ID", "Model", "Sedentary%", "Light%", "Moderate%", "Vigorous%", "MVPA%"]]

        if data_type == 'minutes':
            intensity_list = ["Sedentary", "Light", "Moderate", "Vigorous", "MVPA"]
            df = self.df_activity[["ID", "Model", "Sedentary", "Light", "Moderate", "Vigorous", "MVPA"]]

        for i, intensity in enumerate(intensity_list):

            if i == 0:
                aov_df = pg.rm_anova(data=df, dv=intensity, within="Model", subject="ID",
                                     correction=True, detailed=True)
                aov_df.insert(0, "Intensity", [intensity for i in range(2)])

                """post_df = pg.pairwise_ttests(dv=intensity, subject='ID', within="Model",
                                             data=df, padjust="none", effsize="hedges", parametric=False)
                post_df.insert(0, "Intensity", [intensity for i in range(6)])"""

            if i > 0:
                aov = pg.rm_anova(data=df, dv=intensity, within="Model", subject="ID",
                                  correction=True, detailed=True)
                aov["Intensity"] = [intensity for i in range(2)]

                """post = pg.pairwise_ttests(dv=intensity, subject='ID', within="Model",
                                          data=df, padjust="none", effsize="hedges", parametric=False)
                post["Intensity"] = [intensity for i in range(6)]"""

                aov_df = aov_df.append(aov)

                # post_df = post_df.append(post)

            aov_df["Significant"] = ["Yes" if p < .05 else "No" for p in aov_df["p-unc"]]

        return aov_df

    def activity_descriptive_stats(self):

        model_names = [i for i in set(self.df_activity["Model"])]
        n_subjs = len(set(self.df_activity["ID"]))

        for i, model in enumerate(model_names):
            if i == 0:
                output_df = self.df_activity.groupby("Model").get_group(model).describe()
                output_df = output_df.loc[["mean", "std"]].iloc[:, 1:]  # Removes ID column
                ci = [i / (n_subjs ** (1 / 2)) * scipy.stats.t.ppf(.95, n_subjs - 1) for i in output_df.loc["std"]]
                output_df.loc["CI"] = ci
                output_df.insert(0, "Model", [model for i in range(3)])

            if i > 0:
                df = self.df_activity.groupby("Model").get_group(model).describe()
                df = df.loc[["mean", "std"]].iloc[:, 1:]  # Removes ID column
                ci = [i / (n_subjs ** (1 / 2)) * scipy.stats.t.ppf(.95, n_subjs - 1) for i in df.loc["std"]]
                df.loc["CI"] = ci
                df["Model"] = [model for i in range(3)]

                output_df = output_df.append(df)

        return output_df

    def plot_activity_stats(self, data_type='percent', show_error=True):

        if data_type == "percent":
            df = self.activity_stats[["Model", 'Sedentary%', 'Light%', 'Moderate%', 'Vigorous%']]
        if data_type == "minutes":
            df = self.activity_stats[["Model", 'Sedentary', 'Light', 'Moderate', 'Vigorous']]

        n_subjs = len(set(self.df_activity["ID"]))

        plt.subplots(2, 2, figsize=(12, 6))
        plt.subplots_adjust(wspace=.25, hspace=.25)
        plt.suptitle("Model Means ± 95% CI (n={})".format(n_subjs))

        # Means with 95%CI error bars ---------------------------------------------------------------------------------
        for i, intensity in enumerate([i for i in df.keys()[1:]]):
            plt.subplot(2, 2, i+1)

            if show_error:
                plt.bar(x=[i for i in set(df["Model"])],
                        height=[i * 100 if data_type == "percent" else i for i in df.loc["mean"][intensity]],
                        color=['red', 'blue', 'green', 'purple'], edgecolor='black', alpha=.75,
                        yerr=[i * 100 if data_type == "percent" else i for i in df.loc["CI"][intensity]], capsize=4)
            if not show_error:
                plt.bar(x=[i for i in set(df["Model"])],
                        height=[i * 100 if data_type == "percent" else i for i in df.loc["mean"][intensity]],
                        color=['red', 'blue', 'green', 'purple'], edgecolor='black', alpha=.75)

            if i % 2 == 0:
                plt.ylabel(data_type.capitalize())
            plt.title(intensity)

            # Scatterplot of all data points
            for model in [i for i in set(df["Model"])]:
                plt.scatter(x=[model for i in range(n_subjs)],
                            y=[i * 100 if data_type == "percent" else i
                               for i in self.df_activity.groupby("Model").get_group(model)[intensity]],
                            color='black', marker="x")


x = Data("/Users/kyleweber/Desktop/Data from Vanessa/Outputs/AllSensors/")
# x.plot_activity_stats(data_type="percent", show_error=False)
