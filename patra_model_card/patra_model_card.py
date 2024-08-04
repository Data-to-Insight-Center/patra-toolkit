from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
import json
from json import JSONEncoder
from jsonschema import validate
import os.path
from patra_model_card.fairlearn_bias import BiasAnalyzer
from patra_model_card.shap_xai import ExplainabilityAnalyser


SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'schema.json')

@dataclass
class Metric:
    key: str
    value: str



@dataclass
class AIModel:
    name: str
    version: str
    description: str
    owner: str
    location: str
    license: str
    framework: str
    model_type: str
    test_accuracy: float
    foundational_model: Optional[str] = ""
    model_structure: Optional[str] = ""
    metrics: Dict[str, str] = field(default_factory=dict)

    def add_metric(self, key: str, value: str) -> None:
        self.metrics[key] = value

@dataclass
class BiasAnalysis:
    demographic_parity_difference: float
    equal_odds_difference: float

@dataclass
class ExplainabilityAnalysis:
    name: str
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class ModelCard:
    name: str
    version: str
    short_description: str
    full_description: str
    keywords: str
    author: str
    input_type: str
    category: str
    input_type: str
    input_data: Optional[str] = ""
    output_data: Optional[str] = ""
    ai_model: Optional[AIModel] = None
    bias_analysis: Optional[BiasAnalysis] = None
    xai_analysis: Optional[ExplainabilityAnalysis] = None

    def __str__(self):
        """
        Overriding the __str__ to pretty print the model card in Json format.
        :return:
        """
        return json.dumps(self.__dict__, cls=ModelCardJSONEncoder, indent=4, separators=(',', ': '))

    def populate_bias(self, dataset, true_labels, predicted_labels, sensitive_feature_name, sensitive_feature_data, model):
        bias_analyzer = BiasAnalyzer(dataset, true_labels, predicted_labels, sensitive_feature_name,
                                          sensitive_feature_data, model)

        self.bias_analysis = bias_analyzer.calculate_bias_metrics()

    def populate_xai(self, train_dataset, column_names, model, n_features=10):
        xai_analyzer = ExplainabilityAnalyser(train_dataset, column_names, model)
        self.xai_analysis = xai_analyzer.calculate_xai_features(n_features)


def validate_mc(model_card):
    """
    Validates the current model against the Model Card schema
    :return:
    """
    # Convert the dataclass object to JSON string using the custom encoder
    mc_json = json.dumps(model_card, cls=ModelCardJSONEncoder, indent=4)

    try:
        with open(SCHEMA_JSON, 'r') as schema_file:
            schema = json.load(schema_file)

        validate(json.loads(mc_json), schema)
        return True
    except Exception as e:
        print(e)
        return False


def save_mc(model_card, file_location):
    """
    Saves the model card as a json file.
    """
    with open(file_location, 'w') as json_file:
        json.dump(model_card, json_file, cls=ModelCardJSONEncoder, indent=4, separators=(',', ': '))


class ModelCardJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (ModelCard, Metric, AIModel, ExplainabilityAnalysis, BiasAnalysis)):
            return obj.__dict__
        return super().default(obj)

