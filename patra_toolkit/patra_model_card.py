import json
import logging
import os.path
import tempfile
from dataclasses import dataclass, field
from json import JSONEncoder
from typing import List, Optional, Dict
from urllib.parse import urlparse, urlunparse

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
    inference_label: Optional[str] = ""
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
    citation: Optional[str] = ""
    input_data: Optional[str] = ""
    output_data: Optional[str] = ""
    foundational_model: Optional[str] = ""
    ai_model: Optional[object] = None
    bias_analysis: Optional[object] = None
    xai_analysis: Optional[object] = None
    model_requirements: Optional[List[str]] = None
    id: Optional[str] = field(init=False, default=None)
    credentials: Optional[Dict[str, str]] = field(init=False, default=None)

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

    def submit_model(self,
                     patra_server_url: str,
                     model,
                     file_format: str = "pt",
                     model_store: str = "huggingface",
                     inference_label: Optional[str] = "") -> dict:
        if not self.validate():
            return {"error": "ModelCard validation failed."}
        try:
            self.id = self._get_model_id(patra_server_url)
            logging.info(f"Model ID retrieved: {self.id}")
        except Exception as e:
            logging.error(f"Model ID creation failed: {e}")
            return {"error": f"Model ID creation failed: {str(e)}"}
        try:
            creds = self._get_credentials(patra_server_url, model_store)
            self.credentials = {"token": creds.get("token"), "username": creds.get("username", self.author)}
            logging.info("Repository credentials stored.")
        except Exception as e:
            logging.error(f"Credential retrieval failed: {e}")
            return {"error": f"Credential retrieval failed: {str(e)}"}
        try:
            serialized_model = self._serialize_model(model, file_format)
        except Exception as e:
            logging.error(f"Model serialization failed: {e}")
            return {"error": f"Model serialization failed: {str(e)}"}
        try:
            backend = get_model_store(model_store.lower())
            model_upload_location = backend.upload(serialized_model, self.id, self.credentials)
            logging.info(f"Model uploaded at: {model_upload_location}")
            self.ai_model.location = model_upload_location
            self.output_data = self._extract_repository_link(model_upload_location, model_store)
        except Exception as e:
            logging.error(f"Model upload failed: {e}")
            try:
                backend = get_model_store(model_store.lower())
                backend.delete_repo(self.id, self.credentials)
                logging.info("Rollback successful: repository deleted.")
            except Exception as rollback_err:
                logging.error(f"Rollback failed: {rollback_err}")
            return {"error": f"Model upload failed: {str(e)}"}
        if inference_label:
            try:
                backend = get_model_store(model_store.lower())
                inference_url = backend.upload(inference_label, self.id, self.credentials)
                self.ai_model.inference_label = inference_url
                logging.info(f"Inference label uploaded at: {inference_url}")
            except Exception as e:
                logging.error(f"Inference label upload failed: {e}")
                return {"error": f"Inference label upload failed: {str(e)}"}
        try:
            response = requests.post(
                f"{patra_server_url}/upload_mc",
                json=json.loads(str(self)),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logging.info("Model Card submitted successfully.")
            return response.json()
        except Exception as e:
            logging.error(f"Model Card update failed: {e}")
            try:
                backend = get_model_store(model_store.lower())
                backend.delete_repo(self.id, self.credentials)
                requests.post(
                    f"{patra_server_url}/rollback",
                    json={"id": self.id},
                    headers={'Content-Type': 'application/json'}
                )
                logging.info("Rollback successful after ModelCard update failure.")
            except Exception as rollback_err:
                logging.error(f"Rollback failed: {rollback_err}")
            return {"error": f"ModelCard update failed: {str(e)}"}

    def submit_artifact(self, artifact_path: str) -> dict:
        if not self.output_data:
            raise ValueError("No repository location available. Ensure submit_model() is successful.")
        if "huggingface.co" in self.output_data:
            store = "huggingface"
        elif "github.com" in self.output_data:
            store = "github"
        else:
            raise ValueError("Unsupported repository location. Expected HuggingFace or GitHub URL.")
        backend = get_model_store(store)
        try:
            artifact_location = backend.upload(artifact_path, self.id, self.credentials)
            logging.info(f"Artifact stored at: {artifact_location}")
            return {"artifact_location": artifact_location}
        except Exception as e:
            logging.error(f"Artifact upload failed: {e}")
            raise e

    def _get_model_id(self, patra_server_url: str) -> str:
        url = f"{patra_server_url}/get_model_id"
        response = requests.get(
            url,
            params={"author": self.author, "name": self.name, "version": self.version},
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()

    def _get_credentials(self, patra_server_url: str, model_store: str) -> Dict[str, str]:
        endpoint = "/get_huggingface_credentials" if model_store.lower() == "huggingface" else "/get_github_credentials"
        url = f"{patra_server_url}{endpoint}"
        response = requests.get(
            url,
            params={"author": self.author, "name": self.name, "version": self.version},
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()

    def _serialize_model(self, model, file_format: str) -> str:
        # Use the system temporary directory and construct the filename using self.id.
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"{self.id}.{file_format}")
        if isinstance(model, torch.nn.Module):
            if file_format.lower() == "pt":
                torch.save(model.state_dict(), path)
            elif file_format.lower() == "onnx":
                dummy_input = torch.randn(1, 3, 224, 224)
                torch.onnx.export(model, dummy_input, path)
            elif file_format.lower() == "h5":
                raise Exception("h5 format is not supported for PyTorch models. Use 'pt' or 'onnx'.")
            else:
                raise Exception(f"Unsupported format: {file_format}")
        elif tf is not None and isinstance(model, tf.keras.Model):
            if file_format.lower() == "h5":
                model.save(path, save_format='h5')
            else:
                raise Exception("For TensorFlow models, only 'h5' format is supported.")
        else:
            raise Exception("Unsupported model type or missing required framework.")
        logging.info("Model serialized successfully.")
        return path

    def _extract_repository_link(self, model_upload_location: str, model_store: str) -> str:
        parsed = urlparse(model_upload_location)
        if model_store.lower() == "huggingface":
            repo_path = parsed.path.split('/blob/')[0] if '/blob/' in parsed.path else parsed.path
        elif model_store.lower() == "github":
            repo_path = parsed.path.split('/tree/')[0] if '/tree/' in parsed.path else parsed.path
        else:
            repo_path = parsed.path
        return urlunparse((parsed.scheme, parsed.netloc, repo_path, '', '', ''))

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
