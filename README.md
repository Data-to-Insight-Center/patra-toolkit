# Patra Model Card toolkit

### Installation
In your colab notebook run the following command to install the Model Card
```shell
!pip install 'git+https://github.com/Data-to-Insight-Center/patra-toolkit.git'
```

For developer run the following instead to install locally:
```shell
pip install -e <local_git_dir>/patra_model_card
```

### How to use the library
Please refer to the tensorflow_adult_nn notebook for an example on how you can use this toolkit. 

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

3. Add details to the Model Card
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

Types of framework entries provided:
- tensorflow
- sklearn
- pytorch
- other

Types of model types provided:
- cnn
- decision_tree
- dnn
- rnn
- svm
- kmeans
- random_forest
- llm
- lstm
- other

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


