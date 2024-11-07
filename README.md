# Patra Model Card Toolkit

The **Patra Model Card Toolkit** is designed to simplify and accelerate the creation of AI/ML model cards, automating the addition of essential descriptive information. This toolkit streamlines the integration of standardized details through a schema that captures key characteristics of AI/ML models, supporting transparency, accountability, and ease of use in model documentation.

The toolkit's semi-automated pipeline reduces the time and effort required to develop model cards by populating descriptive fields such as fairness metrics and explainability insights through automated scanners.

## Usage

### [Notebook Examples](./examples/notebooks)
### [Model Card Examples](./examples/model_cards)

---

## Getting Started

### Installation

#### Colab Installation
To install the Patra Model Card toolkit in a Google Colab notebook, run the following:
```shell
!pip install 'git+https://github.com/Data-to-Insight-Center/patra-toolkit.git'
```

#### Local Installation
For local installation, clone the repository and install the toolkit:
```shell
pip install -e <local_git_dir>/patra_model_card
```

### Creating a Model Card

#### Step 1: Import and Initialize

- **Import the Model Card toolkit**:
    ```python
    from patra_model_card import ModelCard, AIModel
    ```

- **Initialize the Model Card**:
    ```python
    mc = ModelCard(
            name="UCI Adult Data Analysis model using Tensorflow",
            version="0.1",
            short_description="UCI Adult Data analysis using Tensorflow for demonstration of Patra Model Cards.",
            full_description="We have trained a ML model using the tensorflow framework to predict income for the UCI Adult Dataset. We leverage this data to run the Patra model cards to capture metadata about the model as well as fairness and explainability metrics.",
            keywords="uci adult, tensorflow, explainability, fairness, patra",
            author="Sachith Withana",
            input_type="Tabular",
            category="classification",
            foundational_model="None"
        )
    ```

   **Model Card Parameters**:
   - `name`: Name of the model card
   - `version`: Version of the model card
   - `short_description`: Brief description of the model card
   - `full_description`: Full description of the model card
   - `keywords`: Keywords for model discoverability
   - `author`: Author of the model card
   - `foundational_model`: Foundational model used, if any

#### Step 2: Add Model Metadata

Add input and output data locations:
```python
mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
mc.output_data = 'https://github.iu.edu/swithana/mcwork/rf_sklearn/adult_model.pkl'
```

#### Step 3: Define Model Details

- **Initialize the AI Model**:
   ```python
   ai_model = AIModel(
       name="UCI Adult Random Forest model",
       version="0.1",
       description="Census classification problem using Random Forest",
       owner="Sachith Withana",
       location="https://github.iu.edu/swithana/mcwork/randomforest/adult_model.pkl",
       license="BSD-3 Clause",
       framework="sklearn",
       model_type="random_forest",
       test_accuracy=accuracy
   )
   ```

   **AI Model Parameters**:
   - `name`: Name of the AI/ML model
   - `version`: Version of the AI/ML model
   - `description`: Description of the AI/ML model
   - `owner`: Owner of the AI/ML model
   - `location`: Location of the stored AI/ML model
   - `license`: License for the AI/ML Model (e.g., BSD-3)
   - `framework`: Framework used to build AI/ML model (tensorflow, sklearn, pytorch, other)
   - `model_type`: Type of AI/ML model (cnn, decision_tree, dnn, rnn, svm, kmeans, random_forest, llm, lstm, other)
   - `test_accuracy`: Accuracy of AI/ML model for the test dataset


- **Populate Model Structure**:
   ```python
   ai_model.populate_model_structure(trained_model)
   mc.ai_model = ai_model
   ```

- **Add Custom Metrics**:
   Additional metrics can be added as needed:
   ```python
   ai_model.add_metric("Test loss", loss)
   ai_model.add_metric("Epochs", 100)
   ai_model.add_metric("Batch Size", 32)
   ai_model.add_metric("Optimizer", "Adam")
   ai_model.add_metric("Learning Rate", 0.0001)
   ai_model.add_metric("Input Shape", "(26048, 100)")
   ```

#### Step 4: Run Fairness and Explainability Scanners

- **Fairness Scanner**:
   To assess fairness, provide the sensitive feature, test data, labels, and predictions:
   ```python
   mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)
   ```

- **Explainability (XAI) Scanner**:
   To generate explainability metrics, specify the dataset, column names, model, and number of features:
   ```python
   mc.populate_xai(X_test, x_columns, model, top_n=10)
   ```

#### Step 5: Validate and Save the Model Card

- **Validation**:
   Verify the model card content against the schema:
   ```python
   mc.validate()
   ```

- **Save the Model Card**:
   Print and save the finalized model card:
   ```python
   print(mc)
   mc.save(<file_path>)
   ```

- **Add Package Requirements**:
   Capture Python package dependencies and versions:
   ```python
   mc.populate_requirements()
   ```

---

## License
The Patra Model Card toolkit is developed by Indiana University and distributed under the BSD 3-Clause License. See `LICENSE.txt` for more details.

## Reference
S. Withana and B. Plale, "Patra ModelCards: AI/ML Accountability in the Edge-Cloud Continuum," 2024 IEEE 20th International Conference on e-Science (e-Science), Osaka, Japan, 2024, pp. 1-10, doi: 10.1109/e-Science62913.2024.10678710. Keywords: Analytical models, Vectors, Edge-cloud continuum, Model cards, AI/ML accountability, Provenance