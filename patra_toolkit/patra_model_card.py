import json
import logging
import os.path
import tempfile
from dataclasses import dataclass, field
from json import JSONEncoder
from typing import List, Optional, Dict

import tensorflow as tf
import jsonschema
import pkg_resources
import requests
import torch

from .exceptions import PatraIDGenerationError, PatraSubmissionError
from .fairlearn_bias import BiasAnalyzer
from .shap_xai import ExplainabilityAnalyser
from .model_store import get_model_store

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'schema.json')
logging.basicConfig(level=logging.INFO)


@dataclass
class Metric:
    """
    Data class for storing metric key-value pairs.
    """
    key: str
    value: str


@dataclass
class AIModel:
    """
    Represents and stores AI model metadata and performance metrics.
    """
    name: str
    version: str
    description: str
    owner: str
    location: str
    license: str
    framework: str
    model_type: str
    test_accuracy: float
    inference_label: Optional[str] = None
    model_structure: Optional[object] = field(default_factory=dict)
    metrics: Dict[str, str] = field(default_factory=dict)

    def add_metric(self, key: str, value: str) -> None:
        """
        Adds a performance metric to the model's metrics.
        """
        self.metrics[key] = value

    def remove_nulls(self, model_structure):
        """
        Recursively removes null values from the model structure.
        """
        if isinstance(model_structure, dict):
            return {k: self.remove_nulls(v) for k, v in model_structure.items() if v is not None}
        elif isinstance(model_structure, list):
            return [self.remove_nulls(v) for v in model_structure if v is not None]
        return model_structure

    def populate_model_structure(self, trained_model):
        """
        Populates the model_structure from a trained model object.
        """
        if self.framework == 'tensorflow':
            json_structure = json.loads(trained_model.to_json())
            self.model_structure = self.remove_nulls(json_structure)
        else:
            self.model_structure = {}


@dataclass
class BiasAnalysis:
    """
    Stores fairness metrics (e.g., demographic parity, equal odds).
    """
    demographic_parity_difference: float
    equal_odds_difference: float


@dataclass
class ExplainabilityAnalysis:
    """
    Stores feature importance or interpretability metrics.
    """
    name: str
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class ModelCard:
    """
    Describes the model's documentation, analyses, and requirements.
    """
    name: str
    version: str
    short_description: str
    full_description: str
    keywords: str
    author: str
    input_type: str
    category: str
    citation: Optional[str] = None
    input_data: Optional[str] = ""
    output_data: Optional[str] = ""
    foundational_model: Optional[str] = ""
    ai_model: Optional[object] = None
    bias_analysis: Optional[object] = None
    xai_analysis: Optional[object] = None
    model_requirements: Optional[List[str]] = None
    id: Optional[str] = field(init=False, default=None)

    def __str__(self) -> str:
        return json.dumps(self.__dict__, cls=ModelCardJSONEncoder, indent=4, separators=(',', ': '))

    def populate_bias(self, dataset, true_labels, predicted_labels,
                      sensitive_feature_name, sensitive_feature_data, model) -> None:
        """
        Calculates bias metrics.
        """
        bias_analyzer = BiasAnalyzer(dataset, true_labels, predicted_labels,
                                     sensitive_feature_name, sensitive_feature_data, model)
        self.bias_analysis = bias_analyzer.calculate_bias_metrics()

    def populate_xai(self, train_dataset, column_names, model, n_features: int = 10) -> None:
        """
        Calculates XAI features.
        """
        xai_analyzer = ExplainabilityAnalyser(train_dataset, column_names, model)
        self.xai_analysis = xai_analyzer.calculate_xai_features(n_features)

    def populate_requirements(self) -> None:
        """
        Gathers environment dependencies.
        """
        exclude_packages = {"shap", "fairlearn"}
        installed_packages = pkg_resources.working_set
        packages_list = sorted([f"{p.key}=={p.version}" for p in installed_packages])
        self.model_requirements = [pkg for pkg in packages_list
                                   if pkg.split("==")[0] not in exclude_packages]

    def validate(self) -> bool:
        """
        Validates the model card against a JSON schema.
        """
        mc_json_str = str(self)
        try:
            with open(SCHEMA_JSON, 'r', encoding='utf-8') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(instance=json.loads(mc_json_str), schema=schema)
            logging.info("Model card validated successfully.")
            return True
        except jsonschema.ValidationError as val_err:
            logging.error(f"Model card validation error: {val_err.message}")
            return False
        except Exception as exc:
            logging.error(f"Unexpected error during validation: {exc}")
            return False

    def _submit_to_store(self, patra_server_url: str, model_store: str, upload_func) -> dict:
        """
        Common submission logic for both model and artifact uploads.
        Validates the model card, checks that the Patra server is reachable,
        generates a unique ID, invokes the upload function, and submits the model card.
        """
        if not patra_server_url:
            return {"error": "No Patra server URL provided."}

        self.id = self._generate_unique_id(patra_server_url)
        logging.info(f"PID generated: {self.id}")

        upload_location = None
        try:
            upload_location = upload_func()
            self.output_data = upload_location.rsplit('/', 3)[0]

            if not self.validate():
                raise Exception("Model card validation failed.")

            response = requests.post(
                f"{patra_server_url}/upload_mc",
                json=json.loads(str(self)),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logging.info("Model card submitted successfully.")
            return response.json()
        except (PatraIDGenerationError, requests.exceptions.RequestException) as e:
            logging.error(f"Submission failed: {e}")
            if upload_location and model_store.lower() in ("huggingface", "github"):
                try:
                    backend = get_model_store(model_store.lower())
                    backend.delete_repo(self.id, patra_server_url)
                    logging.info("Rollback successful: repository deleted.")
                except Exception as rollback_err:
                    logging.error(f"Rollback failed: {rollback_err}")
            return {"error": f"Submission failed: {str(e)}"}

    def submit_model(self,
                     patra_server_url: str,
                     model,
                     file_format: str = "pt",
                     model_store: str = "huggingface") -> dict:
        """
        Uploads a trained model and submits this ModelCard to the Patra server.
        Supports serialization of PyTorch models (as .pt or ONNX) and TensorFlow models (as H5).
        Rolls back repository creation if submission fails.
        """

        def upload_func(model=model):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_name = f"{self.id}.{file_format}"
                path_in_tmp = os.path.join(temp_dir, file_name)

                # If it's a PyTorch model
                if isinstance(model, torch.nn.Module):
                    if file_format.lower() == "pt":
                        # Save state_dict (works for Sequential and other modules)
                        torch.save(model.state_dict(), path_in_tmp)
                    elif file_format.lower() == "onnx":
                        # Define a dummy input. This default is for models accepting (N,3,224,224) images.
                        # For other models, you may need to adjust this.
                        dummy_input = torch.randn(1, 3, 224, 224)
                        torch.onnx.export(model, dummy_input, path_in_tmp)
                    elif file_format.lower() == "h5":
                        raise Exception("h5 format is not supported for PyTorch models. Use 'pt' or 'onnx' instead.")
                    else:
                        raise Exception(f"Unsupported format: {file_format}")
                # Else, if it's a TensorFlow (Keras) model and TensorFlow is installed
                elif tf is not None and isinstance(model, tf.keras.Model):
                    if file_format.lower() == "h5":
                        model.save(path_in_tmp, save_format='h5')
                    else:
                        raise Exception("For TensorFlow models, only 'h5' format is supported.")
                else:
                    raise Exception("Unsupported model type or missing required framework.")

                backend = get_model_store(model_store.lower())
                model_location = backend.upload(path_in_tmp, self.id, patra_server_url)
                logging.info(f"Model stored at: {model_location}")
                self.ai_model.location = model_location
                return model_location

        return self._submit_to_store(patra_server_url, model_store, upload_func)

    def submit_artifact(self,
                        patra_server_url: str,
                        artifact_path: str,
                        model_store: str = "huggingface") -> dict:
        """
        Uploads an artifact file and updates the Patra KG with the new artifact location in the model card.
        Checks for an existing model card to prevent duplicate nodes in Neo4j.
        """

        def upload_func():
            backend = get_model_store(model_store.lower())
            artifact_location = backend.upload(artifact_path, self.id, patra_server_url)
            logging.info(f"Artifact stored at: {artifact_location}")
            return artifact_location

        return self._submit_to_store(patra_server_url, model_store, upload_func)

    def _generate_unique_id(self, patra_server_url: str) -> str:
        """
        Retrieves a unique PID from the Patra server based on model metadata.
        """
        if not patra_server_url:
            raise PatraIDGenerationError("No server URL provided for PID generation.")

        try:
            response = requests.get(
                f"{patra_server_url}/get_pid",
                params={"author": self.author, "name": self.name, "version": self.version},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            msg = f"HTTP {response.status_code} error from server: {response.reason}"
            logging.error(msg)
            raise PatraIDGenerationError(msg)
        except requests.exceptions.ConnectionError:
            raise PatraIDGenerationError("Connection to Patra server failed.")
        except requests.exceptions.Timeout:
            raise PatraIDGenerationError("Request to Patra server timed out.")
        except requests.exceptions.RequestException as req_exc:
            raise PatraIDGenerationError(f"Unexpected error: {req_exc}")

    def save(self, file_location: str) -> None:
        """
        Serializes the model card as JSON to the specified path.
        """
        try:
            with open(file_location, 'w', encoding='utf-8') as json_file:
                json_file.write(str(self))
            logging.info(f"Model card saved to {file_location}.")
        except IOError as io_err:
            logging.error(f"Failed to save model card: {io_err}")


class ModelCardJSONEncoder(JSONEncoder):
    """
    Custom JSON Encoder for ModelCard to handle complex objects.
    """

    def default(self, obj):
        if isinstance(obj, (ModelCard, Metric, AIModel, ExplainabilityAnalysis, BiasAnalysis)):
            return obj.__dict__
        return super().default(obj)
