import pandas as pd


def calculate_sens_spec(filename, predicted_value_colname, true_value_colname):

    data = pd.read_excel(io=filename)

    pred_value = data[predicted_value_colname]
    true_value = data[true_value_colname]

    outcome_list = []

    for pred, truth in zip(pred_value, true_value):

        if int(pred) == truth == 1:
            outcome_list.append("TP")
        if int(pred) == truth == 0:
            outcome_list.append("TN")
        if int(pred) == 1 and truth == 0:
            outcome_list.append("FP")
        if int(pred) == 0 and truth == 1:
            outcome_list.append("FN")

    sens = outcome_list.count("TP") / (outcome_list.count("TP") + outcome_list.count("FN"))
    spec = outcome_list.count("TN") / (outcome_list.count("TN") + outcome_list.count("FP"))

    return sens, spec
