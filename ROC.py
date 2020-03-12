import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


# Function to generate outcome measure
def run_algorithm(threshold):

    output = []

    for i in range(len(data)):

        if data["Valid HR"][i] and data["Valid RR Interval"][i] and data["Valid Voltage"][i] \
                and data['Valid Correlation'][i] and data["RR Ratio"][i] <= threshold:
            output.append(True)
        else:
            output.append(False)

    return output


# Data file
results_file = "/Users/kyleweber/Desktop/Data/OND07/Tabular Data/QualityControl_Testing.xlsx"

# Reads in data file
data = pd.read_excel(io=results_file, header=0, index_col=None, sheet_name="ThresholdTesting",
                     usecols=(1, 2, 4, 5, 6, 7))

expert_decision = np.array(data["ExpertDecision"])

# Threshold value, false positive rate, true positive rate, area under curve
thresh, fpr, tpr, auc, distance = [], [], [], [], []

# Use np.arange to set thresholds to test
for i in np.arange(1, 20, 0.1):

    thresh.append(i)

    cur_fpr, cur_tpr, _ = roc_curve(y_score=run_algorithm(threshold=i), y_true=expert_decision)

    fpr.append(cur_fpr[1])
    tpr.append(cur_tpr[1])

    sens = cur_tpr[1]
    spec = 1 - cur_fpr[1]

    distance.append(((1 - sens) ** 2 + (1 - spec) ** 2) ** (1/2))

    auc.append(roc_auc_score(y_score=run_algorithm(threshold=i), y_true=expert_decision))

# Finds index that corresponds to largest AUC
max_auc_index = auc.index(max(auc))

plt.plot(fpr, tpr, color='black')
plt.plot(fpr[max_auc_index], tpr[max_auc_index],
         marker="o", color='green', label="(FPR = {}, TPR = {})".format(round(fpr[max_auc_index], 3),
                                                                        round(tpr[max_auc_index], 3)))

plt.title("ROC Curve: sensitivity = {} , specificity = {}, "
          "AUC = {}, threshold = {}".format(round(tpr[max_auc_index], 3), round(1-fpr[max_auc_index], 3),
                                            round(auc[max_auc_index], 3), round(thresh[max_auc_index]), 5))

plt.plot(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05), linestyle='dashed', color='red')
plt.legend(loc='upper right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()
