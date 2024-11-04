Arabic Named Entity Recognition Pipeline
======================================

This module provides a comprehensive implementation for Arabic Named Entity Recognition (NER) on the
One Thousand and One Nights dataset. It combines a hybrid approach using rule-based methods and
Conditional Random Fields (CRF), along with the ability to leverage pre-trained Transformer models.

The pipeline includes the following key components:
- Text Preprocessing: Handles Arabic text normalization and cleaning.
- Rule-Based NER: Extracts entities using predefined patterns.
- CRF-Based NER: Trains a CRF model with custom feature engineering.
- Transformer-Based NER: Utilizes pre-trained Transformer models for NER.
- Hybrid Approach: Combines the outputs of the above models.
- Evaluation: Calculates precision, recall, and F1 score metrics.
- Data Handling: Supports data annotation and model/dataset persistence.

Usage Scenarios
---------------

1. **Train and Evaluate the NER Models**
```python
# Initialize the pipeline
pipeline = NERPipeline()

# Load and preprocess the data
annotator = DataAnnotator('one_thousand_nights.csv')
training_data = annotator.prepare_data()

# Train the hybrid NER model
pipeline.train(training_data)

# Evaluate the model performance
text = "قال الملك شهريار لشهرزاد: حدثيني حديثاً عجيباً"
predictions = pipeline.predict(text)

metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1-score: {metrics['f1']:.2f}")
```

2. **Save and Load the Trained Model**
```python
# Save the trained model
with open('arabic_ner_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Load the saved model
with open('arabic_ner_model.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Use the loaded model for prediction
text = "قال الملك شهريار لشهرزاد: حدثيني حديثاً عجيباً"
predictions = loaded_pipeline.predict(text)
```

3. **Annotate the Dataset**
```python
# Initialize the data annotator
annotator = DataAnnotator('one_thousand_nights.csv')

# Prepare the data for annotation
raw_data = annotator.prepare_data()

# Annotate the data using the GUI
annotated_data = annotator.annotate_data(raw_data)

# Save the annotated data
annotator.save_annotations('annotated_data.json')
```

4. **Compare Model Performance**
```python
# Initialize the pipeline
pipeline = NERPipeline()

# Load and preprocess the data
annotator = DataAnnotator('one_thousand_nights.csv')
training_data = annotator.prepare_data()

# Train and evaluate the models
pipeline.train(training_data)
text = "قال الملك شهريار لشهرزاد: حدثيني حديثاً عجيباً"
predictions = pipeline.predict(text)

# Calculate and compare the metrics
rule_based_metrics = ModelEvaluator.calculate_metrics(y_true, rule_based_preds)
crf_metrics = ModelEvaluator.calculate_metrics(y_true, crf_preds)
transformer_metrics = ModelEvaluator.calculate_metrics(y_true, transformer_preds)

print("Rule-based Model:")
print(f"Precision: {rule_based_metrics['precision']:.2f}")
print(f"Recall: {rule_based_metrics['recall']:.2f}")
print(f"F1-score: {rule_based_metrics['f1']:.2f}")

print("CRF Model:")
print(f"Precision: {crf_metrics['precision']:.2f}")
print(f"Recall: {crf_metrics['recall']:.2f}")
print(f"F1-score: {crf_metrics['f1']:.2f}")

print("Transformer Model:")
print(f"Precision: {transformer_metrics['precision']:.2f}")
print(f"Recall: {transformer_metrics['recall']:.2f}")
print(f"F1-score: {transformer_metrics['f1']:.2f}")
```

Documentation
-------------

### `ArabicTextPreprocessor`
This class handles the preprocessing and normalization of Arabic text. It includes the following methods:
- `remove_diacritics`: Removes Arabic diacritical marks.
- `normalize_alef`: Normalizes different forms of the Alef character.
- `normalize_text`: Applies the full text normalization process.

### `DataAnnotator`
The `DataAnnotator` class is responsible for data annotation and handling. It includes the following methods:
- `prepare_data`: Converts the raw text data into a format suitable for annotation.
- `annotate_data`: Provides a GUI-based interface for manual data annotation.
- `save_annotations`: Saves the annotated data to a file.

### `RuleBasedNER`
The `RuleBasedNER` class implements the rule-based component of the hybrid NER model. It includes the following methods:
- `_compile_patterns`: Compiles the regular expression patterns for entity extraction.
- `extract_entities`: Applies the rule-based patterns to extract entities from the input text.

### `CRFModel`
The `CRFModel` class handles the CRF-based component of the NER pipeline. It includes the following methods:
- `extract_features`: Defines the feature extraction process for the CRF model.
- `train`: Trains the CRF model using the provided training data.
- `predict`: Makes predictions using the trained CRF model.

### `TransformerNER`
The `TransformerNER` class provides the Transformer-based NER functionality. It includes the following method:
- `predict`: Uses the pre-trained Transformer model to make predictions on the input text.

### `ModelEvaluator`
The `ModelEvaluator` class is responsible for evaluating the performance of the NER models. It includes the following static method:
- `calculate_metrics`: Calculates precision, recall, and F1 score for the NER predictions.

### `NERPipeline`
The `NERPipeline` class is the main entry point for the Arabic NER implementation. It integrates the various components and provides the following methods:
- `train`: Trains the hybrid NER model using the provided training data.
- `predict`: Makes predictions on the input text using the hybrid approach.
- `_get_rule_based_tag`: Extracts the tag from the rule-based predictions.
- `_get_transformer_tag`: Extracts the tag from the Transformer-based predictions.
