# Patra Model Card toolkit

The **Patra Model Card toolkit** is designed to simplify and accelerate the creation of AI/ML model cards by automating the addition of essential descriptive information about AI/ML models. The toolkit streamlines the integration of standardized details through a schema that captures key characteristics of AI/ML models.

With a semi-automated pipeline, the toolkit reduces the time and effort required to develop model cards by populating a set of descriptive fields independently, with no need for user input. These fields include fairness metrics and explainability information, generated via automated scanners and directly added to the model card.

The copyright to the Patra Model Card toolkit is held by the Indiana University Board of Trustees. 

## Usage
#### Notebook Examples
You can find usage examples in the [Notebooks](./examples/notebooks) folder.

#### Model card Examples
You can find usage examples in the [Model Cards](./examples/model_cards) folder.

## Getting Started

### Installation
To install the Model Card in a Colab notebook, Run:
```shell
!pip install 'git+https://github.com/Data-to-Insight-Center/patra-toolkit.git'
```

To install the Model Card locally, Clone the repository and run:
```shell
pip install -e <local_git_dir>/patra_model_card
```

##### Adding the model metadata

1. Import the Model Card
```python
from patra_model_card.patra_model_card import ModelCard, AIModel
```

2. Initialize the Model Card
```python
mc = ModelCard(
            name="UCI Adult Data Analysis via Random Forest",
            version="0.1",
            short_description="UCI Adult Data analysis using SKLearn and Random Forest",
            full_description="Using a Random Forest to train on UCI Adult Data Analysis",
            keywords="uci adult, sklearn, random_forest, explainability, fairness, fairlearn, shap",
            author="Sachith Withana",
            foundational_model="None",
        )
```

### **Parameters**
  - **name**: Name of the model card
  - **version**: Version of the model card
  - **short_description**: Short description of the model card
  - **full_description**: Full description of the model card
  - **keywords**: Keywords for the model card
  - **author**: Author of the model card
  - **foundational_model**: Foundational model for the model card

4. Add details to the Model Card
```python
mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
mc.output_data = 'https://github.iu.edu/swithana/mcwork/rf_sklearn/adult_model.pkl'
```

4. Add model information
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
# Automatically populate model structure into the model
ai_model.populate_model_structure(trained_model)
# add the model information to the Model Card
mc.ai_model = ai_model
```
### **Parameters**
- **name**: Name of the AI Model
- **version**: Version of the AI Model
- **description**: Description of the AI Model
- **owner**: Owner of the AI Model
- **location**: Location of the AI Model
- **license**: License of the AI Model
- **framework**: Framework used to build the AI Model (tensorflow, sklearn, pytorch, other)
- **model_type**: Type of the AI Model (cnn, decision_tree, dnn, rnn, svm, kmeans, random_forest, llm, lstm, other)
- **test_accuracy**: Test accuracy of the AI Model

You can also add custom metrics: 
```python
ai_model.add_metric("Test loss", loss)
ai_model.add_metric("Epochs", 100)
ai_model.add_metric("Batch Size", 32)
ai_model.add_metric("Optimizer", "Adam")
ai_model.add_metric("Learning Rate", 0.0001)
ai_model.add_metric("Input Shape", "(26048, 100)")
```

##### Running fairness scanners

For this, the user needs to provide the sensitive feature as a column along with the following information
- test_dataset
- test_labels
- predictions for the test dataset
- sensitive feature name
- sensitive feature data as a column
- model

The following shows an example usage for fairness information population.
```python
mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)
```

##### Running xai scanners

For this, the user needs to provide the following:
- train_dataset
- column names
- model
- number of top n xai features (default=10)
The following shows an example usage for fairness information population.
```python
mc.populate_xai(X_test, x_columns, model, 10)
```

##### Validating the model card
This validates the content of the model card against the Patra Model Card schema.
```python
mc.validate()
```


##### Saving the model card
Verify the information in the model card
```python
print(mc)
```

Save the model card to a given file path. 
```python
mc.save(<file_path>)
```

##### Save python package requirements 
You can optionally capture the python package requirements and it's versions into the model card using the following function

```python
mc.populate_requirements()
```

## License
Distributed under the BSD 3-Clause License. See `LICENSE.txt` for more information.
