# Patra Model Card Toolkit

The Patra Toolkit is a component of the Patra ModelCards framework designed to simplify the process of creating and documenting AI/ML models. It provides a structured schema that guides users in providing essential information about their models, including details about the model's purpose, development process, and performance. The toolkit also includes features for semi-automating the capture of key information, such as fairness and explainability metrics, through integrated analysis tools. By reducing the manual effort involved in creating model cards, the Patra Toolkit encourages researchers and developers to adopt best practices for documenting their models, ultimately contributing to greater transparency and accountability in AI/ML development.

The features of the Patra Toolkit's includes:

- **Structured Schema:** The Patra Toolkit offers a structured schema to guide users in providing crucial model information. This includes details such as the model's intended use, development process, and performance metrics.
  
- **Semi-Automated Information Capture:** The toolkit supports semi-automatic capture of certain descriptive fields. It achieves this by running a variety of automated scanners, with the results incorporated into the Model Card.  These include,
    - **Fairness Scanner** evaluates the model's fairness by examining its predictions across different groups. 
    - **Explainability Scanner** generates explainability metrics to help understand the model's decision-making process.
    - **Model Requirements Scanner** captures the Python packages and versions required to run the model.

- **Validation and JSON Generation:** Once a Model Card is created using the Toolkit, it validates the data against the defined schema to ensure completeness and accuracy. It then generates the Model Card as a JSON file, ready for integration into the Patra Knowledge Base.
  
- **Integration with Patra Knowledge Base:** The Model Cards created using the Patra Toolkit are designed to be added to the [Patra Knowledge Base](https://github.com/Data-to-Insight-Center/patra-kg), which is a graph database that stores and manages these cards.

The Patra Toolkit plays a crucial role in promoting transparency and accountability in AI/ML development by making it easier for developers to create comprehensive and informative Model Cards. By automating certain aspects of the documentation process and providing a structured schema, the Toolkit reduces the barriers to entry for creating high-quality model documentation.

For more information, please refer to the [Patra ModelCards paper](https://ieeexplore.ieee.org/document/10678710).

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

- **Submit**: Upload the model card to the Patra server:
   ```python
   mc.submit(<patra_server_url>)
   ```
---

## License
The Patra Model Card toolkit is developed by Indiana University and distributed under the BSD 3-Clause License. See `LICENSE.txt` for more details.

## Reference
S. Withana and B. Plale, "Patra ModelCards: AI/ML Accountability in the Edge-Cloud Continuum," 2024 IEEE 20th International Conference on e-Science (e-Science), Osaka, Japan, 2024, pp. 1-10, doi: 10.1109/e-Science62913.2024.10678710. Keywords: Analytical models, Vectors, Edge-cloud continuum, Model cards, AI/ML accountability, Provenance