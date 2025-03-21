<div align="center">

# Patra Model Card Toolkit

[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://patra-toolkit.readthedocs.io/en/latest/)
[![Build Status](https://github.com/Data-to-Insight-Center/patra-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Data-to-Insight-Center/patra-toolkit/actions)
[![PyPI version](https://badge.fury.io/py/patra-toolkit.svg)](https://pypi.org/project/patra-toolkit/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Example Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-to-Insight-Center/patra-toolkit/blob/main/examples/notebooks/GettingStarted.ipynb)

</div>

The Patra Toolkit is a component of the Patra ModelCards framework designed to simplify the process of creating and documenting AI/ML models. It provides a structured schema that guides users in providing essential information about their models, including details about the model's purpose, development process, and performance. The toolkit also includes features for semi-automating the capture of key information, such as fairness and explainability metrics, through integrated analysis tools. By reducing the manual effort involved in creating model cards, the Patra Toolkit encourages researchers and developers to adopt best practices for documenting their models, ultimately contributing to greater transparency and accountability in AI/ML development.

---

## Features

1. **Encourages Accountability**  
   - Incorporate essential model information (metadata, dataset details, fairness, explainability) at training time, ensuring AI models remain transparent from development to deployment.

2. **Semi-Automated Capture**  
   - Automated *Fairness* and *Explainability* scanners compute demographic parity, equal odds, SHAP-based feature importances, etc., for easy integration into Model Cards.

3. **Machine-Actionable Model Cards**  
   - Produce a structured JSON representation for ingestion into the Patra Knowledge Base. Ideal for advanced queries on model selection, provenance, versioning, or auditing.

4. **Flexible Repository Support**  
   - Pluggable backends for storing models/artifacts on **Hugging Face** or **GitHub**, unifying the model publishing workflow.

5. **Versioning & Model Relationship Tracking**  
   - Maintain multiple versions of a model with recognized edges (e.g., `revisionOf`, `alternateOf`) using embedding-based similarity. This ensures clear lineages and easy forward/backward provenance.

---

## 1. Create a Model Card

```python
from patra_toolkit import ModelCard

mc = ModelCard(
    name="MyModel",
    version="0.1",
    short_description="A demonstration Model Card",
    full_description="Trains a basic PyTorch classification model.",
    keywords="classification, demonstration",
    author="my-user",
    input_type="Tabular",
    category="classification"
)
```
Use the `ModelCard` constructor to capture high-level information: model name, version, short/full descriptions, domain category, and references.

---

## 2. Initialize an AI/ML Model

```python
from patra_toolkit import AIModel

ai_model = AIModel(
    name="MyTorchModel",
    version="0.1",
    description="PyTorch DNN",
    owner="my-user",
    location="",  # Will be filled upon upload
    license="Apache-2.0",
    framework="pytorch",
    model_type="dnn",
    test_accuracy=0.83
)

# Optionally, add more performance or training metrics
ai_model.add_metric("Epochs", 10)
ai_model.add_metric("BatchSize", 64)
ai_model.add_metric("Optimizer", "Adam")
```

Attach the `AIModel` object to your `ModelCard`:

```python
mc.ai_model = ai_model
```

---

## 3. Populate Fairness and Explainability

### 3.1 Fairness (Bias) Analysis

```python
y_pred = trained_model.predict(X_test)
y_pred = (y_pred >= 0.5).flatten()

mc.populate_bias(
    X_test,
    y_test,
    y_pred,
    "ProtectedFeatureName",
    X_test[:, <index_of_sensitive_feature>],
    trained_model
)

print("Bias Analysis:", mc.bias_analysis)
```
Often includes demographic parity difference and equal odds difference.

### 3.2 Explainability (XAI)

```python
column_names = df.columns.tolist()
column_names.remove('target')

mc.populate_xai(
    X_test[:10],
    column_names,
    trained_model
)

print("Explainability Analysis:", mc.xai_analysis)
```
Leverages SHAP (by default) to compute feature importance, stored in `mc.xai_analysis`.

---

## 4. Validate and Save the Model Card

```python
mc.populate_requirements()

if mc.validate():
    print("Model Card validated successfully.")
    mc.save("my_model_card.json")
else:
    print("Validation failed.")
```
`s mc.save()` writes the final JSON to disk for version control or further editing.

---

## 5. Submit

`mc.submit(...)` lets you post your Model Card and optionally your trained model, inference labels, or artifacts to a repository (Hugging Face or GitHub) while registering the card on the Patra server.

```python
mc.submit(
    patra_server_url=<patra_server_url>,
    model=<trained_model>,
    file_format="pt",
    model_store="huggingface",
    inference_label="labels.txt",
    artifacts=["data/train.csv", "docs/config.yaml"]
)
```

If an ID conflict arises (model already exists), increment `mc.version` and resubmit.

---

## Examples
Explore the following example notebooks and model cards to learn more about how to use the Patra Model Card Toolkit:
[Notebook Example](./examples/notebooks/GettingStarted.ipynb), [Model Card Example](./examples/model_cards/tesorflow_adult_nn_MC.json)

## License
The Patra Model Card toolkit is developed by Indiana University and distributed under the BSD 3-Clause License. See `LICENSE.txt` for more details.

## Acknowledgements
This research is funded in part through the National Science Foundation under award #2112606, AI Institute for Intelligent CyberInfrastructure with Computational Learning in the Environment (ICICLE), and in part through Data to Insight Center (D2I) at Indiana University.

## Reference
S. Withana and B. Plale, "Patra ModelCards: AI/ML Accountability in the Edge-Cloud Continuum," 2024 IEEE 20th International Conference on e-Science (e-Science), Osaka, Japan, 2024, pp. 1-10, doi: 10.1109/e-Science62913.2024.10678710. Keywords: Analytical models, Vectors, Edge-cloud continuum, Model cards, AI/ML accountability, Provenance
