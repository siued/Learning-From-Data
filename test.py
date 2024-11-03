"""
This script is used to train and evaluate a classifier on a set of product reviews.
The script takes in multiple commandline parameters to specify the training and evaluation data,
the classifier to use, and whether to perform sentiment analysis (2-class problem) or
product category classification (6-class problem).

To run the program with the best settings as found by the feature testing, use the following command:

python lfd_assignment1.py -t <path/to/training/file> -d <path/to/testing/file> -c all

The code was tested with Python 3.12, compatibility with other versions is not guaranteed.

The output will be a classification report showing the precision, recall, and f1-score for each class
as well as the overall accuracy. Note that the script can take over a minute to run because it uses
multiple machine learning models in an ensemble.
"""

import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# seed rng
np.random.seed(42)


def create_arg_parser():
    """
    Create argument parser with all necessary arguments.
    To see all arguments run the script with the -h flag.

    :return: The arguments for the current run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='data/train.tsv', type=str,
                        help="Train file to learn from (default data/train.tsv)")
    parser.add_argument("-d", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev file to evaluate on (default data/test.tsv)")
    args = parser.parse_args()
    return args


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


def main():
    """
    Main function to run the script.
    """
    args = create_arg_parser()

    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    with open('bad_words.json') as f:
        bad_words = json.load(f)

    Y_pred = []
    for doc in tqdm(X_test):
        off = False
        for word in bad_words:
            if word in doc:
                off = True
                break
        Y_pred.append('OFF' if off else 'NOT')

    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()
