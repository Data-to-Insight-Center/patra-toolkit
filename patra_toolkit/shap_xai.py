import shap
import pandas as pd
import re

class ExplainabilityAnalyser:
    """
    Provides automated capture of xai information in a given AI workflow.
    """
    def __init__(self, train_dataset, column_names, model):
        self.dataset = train_dataset
        self.column_names = column_names
        self.model = model

    def calculate_xai_features(self, n_features=10, pytorch = False):
        # calculate the shap values
        if pytorch:
            explainer = shap.DeepExplainer(self.model, self.dataset)
        else:
            explainer = shap.Explainer(self.model, self.dataset)

        values = explainer(self.dataset)
        shap_values = values.values

        if len(shap_values.shape) == 3:
            shap_values = abs(shap_values).mean(axis=2)

        feature_importance_df = pd.DataFrame(shap_values, columns=self.column_names).abs().mean(axis=0)
        top_features = feature_importance_df.sort_values(ascending=False).head(n_features)

        result_dict = {}

        for name, importance in top_features.items():

            filtered_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
            result_dict[filtered_name] = float(importance)

        return result_dict
