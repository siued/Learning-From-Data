from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from collections import Counter
import random
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def read_corpus(corpus_file):
    """
    Reads the corpus file with the label as the last token and provides tokenized documents and labels.
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            label = tokens[-1]
            tokens = tokens[:-1]
            tokens = [token.lower().replace('#', '') for token in tokens if token.replace('#', '').isalnum()]
            random.shuffle(tokens)
            document = ' '.join(tokens)
            documents.append(document)
            labels.append(label)
    return documents, labels


def to_categorical(y):
    """
    Converts a list of labels to a one-hot encoded matrix.
    """
    y = np.asarray(y)
    num_classes = len(np.unique(y))
    return np.eye(num_classes)[y.reshape(-1)]


def create_variants(text, n=1):
    """
    Creates a list of text variants, each missing one or two words from the original text.
    Returns the variants and the corresponding sorted two-token combinations that were removed.
    """
    tokens = text.split()
    variants = []

    if n == 1:
        for i in range(len(tokens)):
            variant = tokens[:i] + tokens[i + 1:]  # Remove the ith token
            variants.append(' '.join(variant))
        return variants, tokens

    elif n == 2:
        removed_token_combinations = []  # To store the removed token combinations

        # Iterate over pairs of indices to remove two words at a time
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):  # Ensure j is always greater than i
                # Create a variant by removing the ith and jth tokens
                variant = [tokens[k] for k in range(len(tokens)) if k != i and k != j]
                variants.append(' '.join(variant))

                # Get the removed tokens, sort them alphabetically and join
                removed_tokens = sorted([tokens[i], tokens[j]])
                removed_token_combinations.append(' '.join(removed_tokens))

        return variants, removed_token_combinations  # Return variants and their corresponding removed token combinations


# Initialize counters for words causing label changes and overall word frequency
word_change_to_off = Counter()
word_change_to_not = Counter()
word_frequency = Counter()

# Load and process training and development data
X_train, Y_train = read_corpus('data/train.tsv')
X_dev, Y_dev = read_corpus('data/dev.tsv')

# Count word pair frequency in the entire corpus, do all 2-grams
for document in X_dev:
    tokens = document.split()
    for i in range(len(tokens) - 1):
        word_frequency[' '.join(sorted([tokens[i], tokens[i + 1]]))] += 1

for word, count in word_frequency.most_common()[:10]:
    print(f"{word}: {count}")

# Encode the labels
encoder = LabelBinarizer()
Y_train_bin = encoder.fit_transform(Y_train)
Y_dev_bin = encoder.transform(Y_dev)

# Store the class labels
classes = encoder.classes_

Y_train_bin = to_categorical(Y_train_bin)
Y_dev_bin = to_categorical(Y_dev_bin)

# Initialize tokenizer and model
lm = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(lm)
model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2, from_pt=True)

max_length = 64

# Tokenize the training and development data
tokens_train = tokenizer(X_train, padding=True, max_length=max_length, truncation=True, return_tensors="tf").data
tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length, truncation=True, return_tensors="tf").data

# Compile and train the model
loss_function = CategoricalCrossentropy(from_logits=True)
optim = Adam(learning_rate=1e-5)
model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
model.fit(tokens_train, Y_train_bin, verbose=1, epochs=2, batch_size=128, validation_data=(tokens_dev, Y_dev_bin))

ngram = 2
# Analyze variations of each document in the dev set for label changes
for i, text in tqdm(enumerate(X_dev), total=len(X_dev), desc="Processing Dev Set Variants"):
    original_label = Y_dev_bin[i].argmax()  # True label for the original text
    if len(text.split()) < 1 + ngram:
        continue
    variants, tokens = create_variants(text, n=ngram)
    text_tokens = tokenizer([text], padding=True, max_length=max_length, truncation=True, return_tensors="tf").data
    predicted_label = model.predict(text_tokens, verbose=0)["logits"].argmax(axis=1)

    # Tokenize all variants at once for batch processing
    variant_tokens = tokenizer(variants, padding=True, max_length=max_length, truncation=True, return_tensors="tf").data
    variant_predictions = model.predict(variant_tokens, verbose=0)["logits"].argmax(axis=1)  # Batch prediction

    # Track which words caused a label change
    for j, variant_label in enumerate(variant_predictions):
        if variant_label != predicted_label:
            if classes[variant_label] == 'OFF':
                word_change_to_off[tokens[j]] += 1
            else:  # Otherwise, it must be 'NOT'
                word_change_to_not[tokens[j]] += 1

# Normalize and sort words causing label changes
normalized_off = {word: count / word_frequency[word] for word, count in word_change_to_off.items() if word in word_frequency}
normalized_not = {word: count / word_frequency[word] for word, count in word_change_to_not.items() if word in word_frequency}

print("\nWords causing label changes to NOT the most frequently (normalized):")
for word, score in sorted(normalized_not.items(), key=lambda x: x[1], reverse=True):
    if word_change_to_not[word] > 0:  # Check original count for normalization condition
        print(f"{word}: {score:.4f}, count: {word_change_to_not[word]}")


# Final classification report on the original dev set
Y_pred = model.predict(tokens_dev)["logits"]
print(classification_report(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1)))
