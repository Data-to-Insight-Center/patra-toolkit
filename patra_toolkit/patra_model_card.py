import importlib
import json
import logging
import os.path
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from json import JSONEncoder
from typing import List, Optional, Dict
from urllib.parse import urlparse, urlunparse

import jsonschema
import pkg_resources
import requests

from .exceptions import PatraIDGenerationError
from .fairlearn_bias import BiasAnalyzer
from .model_store import get_model_store
from .shap_xai import ExplainabilityAnalyser

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'schema.json')
logging.basicConfig(level=logging.INFO)


def ensure_package_installed(package_name: str, import_name: Optional[str] = None):
    """
    Tries to import a package by name and if not found, installs it using pip.
    Returns the imported module.
    """
    import_name = import_name or package_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        logging.info(f"Package '{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(import_name)


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

    def submit(
            self,
            patra_server_url: str,
            model: Optional[object] = None,
            file_format: Optional[str] = "h5",
            model_store: Optional[str] = "huggingface",
            inference_label: Optional[str] = None,
            artifacts: Optional[List[str]] = None
    ):
        """
        Submits the ModelCard along with optional components to the Patra server
        and the specified model store.
        """
        if not self.validate():
            logging.error("ModelCard validation failed.")
            return None

        # Determine if a model is provided (uploading_model flag is not used for suppressing conflicts)
        is_uploading_model = True if model is not None else False

        try:
            self.id = self._get_model_id(patra_server_url, is_uploading_model)
            logging.info(f"Model ID retrieved: {self.id}")
        except PatraIDGenerationError as pid_exc:
            logging.error(f"Model submission failed during model ID creation: {pid_exc}")
            return None
        except Exception as e:
            logging.error(f"Model submission failed during model ID creation: {e}")
            return None

        model_upload_location = None
        inference_url = None
        artifact_locations = []
        upload_requested = any([model, inference_label, artifacts])

        if upload_requested:
            try:
                creds = self._get_credentials(patra_server_url, model_store)
                self.credentials = {"token": creds.get("token"), "username": creds.get("username")}
            except Exception as e:
                logging.error(f"Model submission failed during credential retrieval: {e}")
                return None

            if model is not None:
                try:
                    if file_format.lower() == "h5":
                        ensure_package_installed("tensorflow")
                    elif file_format.lower() in ["pt", "onnx"]:
                        ensure_package_installed("torch")
                    serialized_model = self._serialize_model(model, file_format)
                except Exception as e:
                    logging.error(f"Model submission failed during model serialization: {e}")
                    return None

                backend = get_model_store(model_store.lower())
                try:
                    model_upload_location = backend.upload(serialized_model, self.id, self.credentials)
                    logging.info(f"Model uploaded at: {model_upload_location}")
                    self.ai_model.location = model_upload_location
                    self.output_data = self._extract_repository_link(model_upload_location, model_store)
                except Exception as e:
                    logging.error(f"Model submission failed during model upload: {e}")
                    try:
                        backend.delete_repo(self.id, self.credentials)
                        logging.info("Rollback successful: repository deleted.")
                    except Exception as rollback_err:
                        logging.error(f"Rollback failed: {rollback_err}")
                    return None

            if inference_label is not None:
                try:
                    backend = get_model_store(model_store.lower())
                    if model_store.lower() == "huggingface":
                        ensure_package_installed("huggingface_hub")
                    elif model_store.lower() == "github":
                        ensure_package_installed("PyGithub", "github")
                    inference_url = backend.upload(inference_label, self.id, self.credentials)
                    self.ai_model.inference_label = inference_url
                    logging.info(f"Inference label uploaded at: {inference_url}")
                except Exception as e:
                    logging.error(f"Model submission failed during inference label upload: {e}")
                    return None

            if artifacts is not None:
                try:
                    backend = get_model_store(model_store.lower())
                    for artifact in artifacts:
                        loc = backend.upload(artifact, self.id, self.credentials)
                        logging.info(f"Artifact '{artifact}' uploaded at: {loc}")
                        artifact_locations.append(loc)
                except Exception as e:
                    logging.error(f"Model submission failed during artifact upload: {e}")
                    return None

        try:
            submission_payload = json.loads(str(self))
            response = requests.post(
                f"{patra_server_url}/upload_mc",
                json=submission_payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logging.info("Model Card submitted successfully.")
            return "success"
        except Exception as e:
            logging.error(f"Model submission failed during ModelCard submission: {e}")
            if upload_requested:
                try:
                    backend = get_model_store(model_store.lower())
                    backend.delete_repo(self.id, self.credentials)
                    logging.info("Rollback successful after ModelCard submission failure.")
                except Exception as rollback_err:
                    logging.error(f"Rollback failed: {rollback_err}. Manual cleanup required.")
            return None

    def _get_model_id(self, patra_server_url: str, is_uploading_model: bool) -> str:
        """
        Attempts to retrieve a new model ID from the Patra server.
        If the ID (based on author, name, version) already exists, the function
        always raises an error to force the user to update the version, regardless
        of whether they intend to upload a model or just submit artifacts.
        """
        if not patra_server_url:
            raise PatraIDGenerationError("No server URL provided for PID generation.")

        try:
            response = requests.get(
                f"{patra_server_url}/get_model_id",
                params={"author": self.author, "name": self.name, "version": self.version},
                headers={'Content-Type': 'application/json'}
            )
            if is_uploading_model and response.status_code == 409:
                raise ValueError(
                    "Model ID already exists. Please update the version."
                )
            response.raise_for_status()
            id_data = response.json()
            return id_data["pid"]
        except requests.exceptions.ConnectionError:
            raise PatraIDGenerationError("Patra server is unreachable.")
        except requests.exceptions.Timeout:
            raise PatraIDGenerationError("Patra server connection timed out.")
        except requests.exceptions.RequestException as req_exc:
            raise PatraIDGenerationError(f"Request failed: {req_exc}")

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
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"{self.id}.{file_format}")

        # Lazy imports for torch and tensorflow
        torch_module = None
        tf_module = None
        try:
            import torch
            torch_module = torch
        except ImportError:
            torch_module = ensure_package_installed("torch")
        try:
            import tensorflow as tf_mod
            tf_module = tf_mod
        except ImportError:
            tf_module = ensure_package_installed("tensorflow")

        # If using PyTorch, perform a lazy import of torch.nn as torch_nn
        if torch_module is not None:
            try:
                from torch import nn as torch_nn
            except ImportError:
                torch_nn = None
        else:
            torch_nn = None

        # PyTorch model branch
        if torch_module is not None and torch_nn is not None and isinstance(model, torch_nn.Module):
            # Allow saving state_dict to either "pt" or "h5"
            if file_format.lower() in ["pt", "h5"]:
                torch_module.save(model.state_dict(), path)
            elif file_format.lower() == "onnx":
                dummy_input = torch_module.randn(1, 3, 224, 224)
                torch_module.onnx.export(model, dummy_input, path)
            else:
                raise Exception(f"Unsupported format for PyTorch models: '{file_format}'")
        # TensorFlow model branch
        elif tf_module is not None and isinstance(model, tf_module.keras.Model):
            if file_format.lower() == "h5":
                model.save(path, save_format='h5')
            else:
                raise Exception("For TensorFlow models, only 'h5' format is supported.")
        else:
            raise Exception("Unsupported model type or missing required framework.")

        logging.info("Model serialized successfully.")
        return path

    @staticmethod
    def _extract_repository_link(model_upload_location: str, model_store: str) -> str:
        """
        Extracts the repository link from the model upload location.
        """
        parsed = urlparse(model_upload_location)
        if model_store.lower() == "huggingface":
            repo_path = parsed.path.split('/blob/')[0] if '/blob/' in parsed.path else parsed.path
        elif model_store.lower() == "github":
            repo_path = parsed.path.split('/tree/')[0] if '/tree/' in parsed.path else parsed.path
        else:
            repo_path = parsed.path
        return urlunparse((parsed.scheme, parsed.netloc, repo_path, '', '', ''))


class ModelCardJSONEncoder(JSONEncoder):
    """
    Custom JSON Encoder for ModelCard to handle complex objects.
    """

    def default(self, obj):
        if isinstance(obj, (ModelCard, Metric, AIModel, ExplainabilityAnalysis, BiasAnalysis)):
            return obj.__dict__
        return super().default(obj)
