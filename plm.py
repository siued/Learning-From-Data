from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import random
import numpy as np
random.seed(42)
np.random.seed(42)

def read_corpus(corpus_file):
    """
    Reads the corpus file with the label as the last token and provides tokenized documents and labels.

    :param corpus_file: The name of the corpus file to be processed
    :return: The tokenized documents and labels
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            label = tokens[-1]  # The label is the last token in the line
            document = ' '.join(tokens[:-1])  # The document content is everything except the last token
            documents.append(document)
            labels.append(label)
    return documents, labels

def to_categorical(y):
    """
    Converts a list of labels to a one-hot encoded matrix.
    """
    y = np.asarray(y)  # Ensure `y` is a NumPy array
    num_classes = len(np.unique(y))  # Get the number of classes
    return np.eye(num_classes)[y.reshape(-1)]  # Reshape to avoid extra dimensions


X_train, Y_train = read_corpus('data/train.tsv')
X_dev, Y_dev = read_corpus('data/dev.tsv')

encoder = LabelBinarizer()
Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
Y_dev_bin = encoder.transform(Y_dev)

Y_train_bin = to_categorical(Y_train_bin)
Y_dev_bin = to_categorical(Y_dev_bin)

lm = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(lm)
model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2, from_pt=True)
tokens_train = tokenizer(X_train, padding=True, max_length=64,
truncation=True, return_tensors="tf").data
tokens_dev = tokenizer(X_dev, padding=True, max_length=64,
truncation=True, return_tensors="tf").data


loss_function = CategoricalCrossentropy(from_logits=True)
optim = Adam(learning_rate=1e-6)

model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
model.fit(tokens_train, Y_train_bin, verbose=1, epochs=16, batch_size=64, validation_data=(tokens_dev, Y_dev_bin))

Y_pred = model.predict(tokens_dev)["logits"]

# print accuracy and fscore
print(classification_report(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1)))