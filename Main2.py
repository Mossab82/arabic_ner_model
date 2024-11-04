# Required Libraries
import pandas as pd
import re
import nltk
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import precision_recall_fscore_support
from arabert.preprocess import ArabertPreprocessor  # for Arabic text preprocessing
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
from datasets import Dataset

# Installations
# Uncomment if needed
# !pip install pandas nltk sklearn-crfsuite spacy arabert pytorch-transformers datasets
# !python -m spacy download ar_core_news_sm

# Load the Arabic NLP Preprocessing model
model_name = "aubmindlab/bert-base-arabertv02"  # Example with AraBERT for Arabic NLP
arabert_prep = ArabertPreprocessor(model_name)

# Step 1: Data Preparation and Annotation
def preprocess_text(text):
    """ Preprocess Arabic text by removing diacritics and normalizing characters. """
    text = arabert_prep.preprocess(text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

# Load dataset
data = pd.read_csv("one_thousand_and_one_nights.csv")
data['text'] = data['text'].apply(preprocess_text)

# Annotation (Manual Annotation Example in Jupyter Cell)
# Define unique entities (e.g., mythical figures, historical figures)
entities = ['MYTHICAL_FIGURE', 'HISTORICAL_FIGURE', 'PLACE']

# Sample code for manual annotation in Jupyter Notebook
def annotate_data(text, entities):
    from IPython.display import display
    annotated_data = []
    for i, sentence in enumerate(text):
        display(f"Sentence {i+1}: {sentence}")
        labels = input(f"Enter entity labels for sentence {i+1} (separate by commas if multiple): ").split(',')
        labels = [label.strip() for label in labels]
        annotated_data.append({"text": sentence, "labels": labels})
    return annotated_data

# Uncomment below to run manual annotation in a Jupyter Notebook cell
# annotated_data = annotate_data(data['text'], entities)
# Save annotated data to CSV
# pd.DataFrame(annotated_data).to_csv("annotated_one_thousand_and_one_nights.csv", index=False)

# Step 2: Model Training Using Hybrid Approach (CRF and Rule-Based)

# Define custom features for CRF model
def extract_crf_features(sentence):
    """ Feature extraction for each word in a sentence for CRF training. """
    features = []
    for i, word in enumerate(sentence):
        word_features = {
            'word': word,
            'is_capitalized': word[0].isupper(),
            'is_digit': word.isdigit(),
            'prefix1': word[:1],
            'prefix2': word[:2],
            'suffix1': word[-1:],
            'suffix2': word[-2:],
        }
        if i > 0:
            word_features['prev_word'] = sentence[i - 1]
        else:
            word_features['BOS'] = True  # Beginning of sentence
        if i < len(sentence) - 1:
            word_features['next_word'] = sentence[i + 1]
        else:
            word_features['EOS'] = True  # End of sentence
        features.append(word_features)
    return features

# Prepare data for CRF training
sentences = [nltk.word_tokenize(text) for text in data['text']]
labels = data['labels']  # Assuming labels are provided in the dataset
X = [extract_crf_features(sentence) for sentence in sentences]
y = labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the CRF model
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Additional Models: spaCy and Transformers (BERT-based)
# spaCy Model Training
nlp_spacy = spacy.blank("ar")
# Further customization or fine-tuning with spaCy can be added here if training from scratch

# BERT-based Model Training (e.g., AraBERT)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(entities))

# Convert data to Hugging Face Dataset format
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

hf_dataset = Dataset.from_pandas(data)
tokenized_dataset = hf_dataset.map(tokenize_and_align_labels, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

# Step 3: Evaluation
# Evaluate CRF
y_pred = crf.predict(X_test)
print(flat_classification_report(y_test, y_pred, labels=entities))

# Evaluate BERT model
metrics = trainer.evaluate()
print(metrics)

# Step 4: Save the Model and Annotated Data
# Save CRF model
with open('crf_model.pkl', 'wb') as f:
    pickle.dump(crf, f)

# Save the BERT model and tokenizer
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

# Save annotated dataset
data.to_csv("annotated_one_thousand_and_one_nights.csv", index=False)

print("Models and data have been saved for future use.")
