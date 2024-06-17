from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import confusion_matrix


class BiasAnalyzer:
    """
    Provides automated capture of fairness metrics in a given AI workflow.
    """
    def __init__(self, dataset, true_labels, predicted_labels, sensitive_feature_name, sensitive_feature_data, model):
        self.dataset = dataset
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.sensitive_feature = sensitive_feature_name
        self.sensitive_data = sensitive_feature_data
        self.model = model

    def calculate_bias_metrics(self):
        demographic_parity = demographic_parity_difference(self.true_labels, self.predicted_labels,
                                                           sensitive_features=self.sensitive_data)
        equal_odds_diff = equalized_odds_difference(self.true_labels, self.predicted_labels,
                                                    sensitive_features=self.sensitive_data)

        tn, fp, fn, tp = confusion_matrix(self.true_labels, self.predicted_labels).ravel()

        # false_positive_rate = fp / (fp + tn)
        # false_negative_rate = fn / (fn + tp)
        # true_positive_rate = tp / (tp + fn)
        # true_negative_rate = tn / (tn + fp)

        return {"demographic_parity_diff": demographic_parity,
                "equal_odds_difference": equal_odds_diff
                }

