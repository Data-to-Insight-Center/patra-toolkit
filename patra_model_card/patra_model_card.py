import hashlib
import json
import os.path
from dataclasses import dataclass
from dataclasses import field
from json import JSONEncoder
from typing import List, Optional, Dict

import jsonschema
import pkg_resources
import requests

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
    model_structure: Optional[object] = field(default_factory=dict)
    metrics: Dict[str, str] = field(default_factory=dict)

    def add_metric(self, key: str, value: str) -> None:
        self.metrics[key] = value

    def remove_nulls(self, model_structure):
        """
        Removes the null values from the model structure.

        Args:
            model_structure (object): The model structure

        Returns:
            object: The model structure without null values
        """
        if isinstance(model_structure, dict):
            return {k: self.remove_nulls(v) for k, v in model_structure.items() if v is not None and self.remove_nulls(v)}
        elif isinstance(model_structure, list):
            return [self.remove_nulls(v) for v in model_structure if v is not None and self.remove_nulls(v) != []]
        else:
            return model_structure

    def populate_model_structure(self, trained_model):
        """
        Populates the model structure from the trained model.

        Args:
            trained_model (object): The trained model

        Returns:
            None
        """
        if self.framework is not None and self.framework == 'tensorflow':
            json_structure = json.loads(trained_model.to_json())
            cleaned_structure = self.remove_nulls(json_structure)
            self.model_structure = cleaned_structure
        else:
            self.model_structure = {}


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
    foundational_model: Optional[str] = ""
    ai_model: Optional[AIModel] = None
    bias_analysis: Optional[BiasAnalysis] = None
    xai_analysis: Optional[ExplainabilityAnalysis] = None
    model_requirements: Optional[List] = None
    id: Optional[str] = field(init=False, default=None)

    def __str__(self):
        """
        Converts the model card to a JSON string.

        Returns:
            str: The JSON string representation of the model card
        """
        return json.dumps(self.__dict__, cls=ModelCardJSONEncoder, indent=4, separators=(',', ': '))

    def populate_bias(self, dataset, true_labels, predicted_labels, sensitive_feature_name, sensitive_feature_data, model):
        """
        Calculates the fairness metrics and adds it to the model card.

        Args:
            dataset (object): The dataset
            true_labels (list): The true labels
            predicted_labels (list): The predicted labels
            sensitive_feature_name (str): The sensitive feature name
            sensitive_feature_data (list): The sensitive feature data
            model (object): The model

        Returns:
            None
        """
        bias_analyzer = BiasAnalyzer(dataset, true_labels, predicted_labels, sensitive_feature_name,
                                          sensitive_feature_data, model)

        self.bias_analysis = bias_analyzer.calculate_bias_metrics()

    def populate_xai(self, train_dataset, column_names, model, n_features=10):
        """
        Calculates the top n_features in terms of feature importance and adds it to the model card.

        Args:
            train_dataset (object): The training dataset
            column_names (list): The column names
            model (object): The model
            n_features (int): The number of features to calculate

        Returns:
            None
        """
        xai_analyzer = ExplainabilityAnalyser(train_dataset, column_names, model)
        self.xai_analysis = xai_analyzer.calculate_xai_features(n_features)

    def populate_requirements(self):
        """
        Gets all the package requirements for this model.

        Returns:
            None
        """
        # Patra related packages. Remove this from the requirement list.
        exclude_packages = {"shap", "fairlearn"}

        # Get all the installed packages
        installed_packages = pkg_resources.working_set
        packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

        # all packages except Patra packages
        filtered_packages_list = [pkg for pkg in packages_list if pkg.split("==")[0] not in exclude_packages]
        self.model_requirements = filtered_packages_list

    def validate(self):
        """
        Validates the current model against the Model Card schema.

        Returns:
            bool: True if the model card is valid, False
        """
        # Convert the dataclass object to JSON string using the custom encoder
        mc_json = self.__str__()

        try:
            with open(SCHEMA_JSON, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(instance=json.loads(mc_json), schema=schema)
            return True
        except jsonschema.ValidationError as e:
            # Print the error message only
            print(e.message)  # This will print only the specific error message
            return False
        except Exception as e:
            # For any other exception, print the error message
            print(f"An unexpected error occurred: {e}")
            return False

    def submit(self, patra_server_url):
        """
        Validates and submits the model card to the Patra Server.

        Args:
            patra_server_url (str): The Patra Server URL
        """
        if self.validate():
            try:
                if patra_server_url:
                    self.id = self._get_hash_id(patra_server_url)
                    patra_submit_url = patra_server_url + "/upload_mc"
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(patra_submit_url, json=json.loads(str(self)), headers=headers)
                    response.raise_for_status()
                    return response.json()
                else:
                    return {"An error occurred: valid patra_server_url not provided. Unable to upload."}
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                return None

    def _get_hash_id(self, patra_server_url):
        """
        Retrieve a unique hash generated from the provided name, version, and author. If patra_server_url is null or the server is down, generate the hash ID locally.

        Args:
            patra_server_url (str): The Patra Server URL
        """
        combined_string = f"{self.name}:{self.version}:{self.author}"
        try:
            if patra_server_url:
                patra_hash_url = patra_server_url + "/get_hash_id"
                headers = {'Content-Type': 'application/json'}
                response = requests.get(patra_hash_url,params={"combined_string": combined_string}, headers=headers)
                response.raise_for_status()
                return response.json()
            else:
                id_hash = hashlib.sha256(combined_string.encode()).hexdigest()
                return id_hash
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            id_hash = hashlib.sha256(combined_string.encode()).hexdigest()
            return  id_hash

    def save(self, file_location):
        """
        Saves the model card as a json file.
        """
        with open(file_location, 'w') as json_file:
            json_file.write(str(self))


class ModelCardJSONEncoder(JSONEncoder):
    """
    JSON encoder for the model card.

    Args:
        JSONEncoder (object): The JSON encoder

    Returns:
        object: The JSON encoder object
    """
    def default(self, obj):
        if isinstance(obj, (ModelCard, Metric, AIModel, ExplainabilityAnalysis, BiasAnalysis)):
            return obj.__dict__
        return super().default(obj)

