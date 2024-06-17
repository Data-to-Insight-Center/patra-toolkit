import shap
import pandas as pd

class ExplainabilityAnalyser:
    """
    Provides automated capture of xai information in a given AI workflow.
    """
    def __init__(self, train_dataset, column_names, model):
        self.dataset = train_dataset
        self.column_names = column_names
        self.model = model

    def calculate_xai_features(self, n_features=10):
        # calculate the shap values
        explainer = shap.Explainer(self.model, self.dataset)
        shap_values = explainer.shap_values(self.dataset)

        # calculate the feature importance
        feature_importance = pd.DataFrame(shap_values, columns=self.column_names).abs().mean(axis=0)

        # Get the top n features
        top_features = feature_importance.sort_values(ascending=False).head(n_features)

        result_dict = {}
        for name, importance in top_features.items():
            result_dict[name] = float(importance)

        return result_dict