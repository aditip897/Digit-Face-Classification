import numpy as np
import random
import time
from naivebay import NaiveBayesClassifier
from percep import PerceptronClassifier
from extract_features import extract_features
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from data_loader import load_images, load_labels
from typing import List, Dict, Tuple

DIGIT_TRAIN_LABELS_PATH = 'data/digitdata/traininglabels'
DIGIT_TRAIN_IMAGES_PATH = 'data/digitdata/trainingimages'

DIGIT_TEST_IMAGES_PATH = 'data/digitdata/testimages'
DIGIT_TEST_LABELS_PATH = 'data/digitdata/testlabels'

FACE_TRAIN_IMAGES_PATH = 'data/facedata/facedatatrain'
FACE_TRAIN_LABELS_PATH = 'data/facedata/facedatatrainlabels'

FACE_TEST_IMAGES_PATH = 'data/facedata/facedatatest'
FACE_TEST_LABELS_PATH = 'data/facedata/facedatatestlabels'


def load_data(image_file_path, label_file_path, data_type):
    images = load_images(image_file_path, image_type=data_type)
    labels = load_labels(label_file_path)
    features = []
    for img in images:
        extracted = extract_features(img, image_type=data_type)
        normalized = [float(val) / max(1.0, max(extracted)) for val in extracted]
        features.append({f'feature_{idx}': val for idx, val in enumerate(normalized)})
    return features, labels


def get_data_subset(data_features, data_labels, percent):
    subset_len = int(len(data_features) * percent / 100)
    selected_indices = random.sample(range(len(data_features)), subset_len)
    return ([data_features[i] for i in selected_indices], [data_labels[i] for i in selected_indices])


def evaluate_model(model, data_label, train_feats, train_labels, test_feats, test_labels, iterations=5):
    percentages = range(10, 101, 10)
    stats = []

    for pct in percentages:
        subset_len = int(len(train_feats) * pct / 100)
        errors = []
        total_duration = 0

        for _ in range(iterations):
            subset_feats, subset_labels = get_data_subset(train_feats, train_labels, pct)
            start = time.time()
            model.train(subset_feats, subset_labels, None, None)
            predictions = model.classify(subset_feats)
            accuracy = accuracy_score(subset_labels, predictions)
            errors.append(1 - accuracy)
            total_duration += time.time() - start

        stats.append([
            subset_len,
            f"{pct}%",
            f"{accuracy:.4f}",
            f"{np.mean(errors):.4f}",
            f"±{np.std(errors):.4f}",
            f"{total_duration / iterations:.2f}" ])

    return stats, data_label


def display_results(collected_stats):
    headers = ["Samples", "Percentage", "Accuracy", "Avg Error", "Std Dev", "Avg Time (s)"]

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for stats, label in collected_stats:
        print(f"\n{label}:")
        print("-" * 80)
        print(tabulate(stats, headers=headers, tablefmt="grid"))


def main():
    face_train_feats, face_train_labels = load_data(
        FACE_TRAIN_IMAGES_PATH, FACE_TRAIN_LABELS_PATH, 'face'
    )
    face_test_feats, face_test_labels = load_data(
        FACE_TEST_IMAGES_PATH, FACE_TEST_LABELS_PATH, 'face'
    )

    digit_train_feats, digit_train_labels = load_data(
        DIGIT_TRAIN_IMAGES_PATH, DIGIT_TRAIN_LABELS_PATH, 'digit'
    )
    digit_test_feats, digit_test_labels = load_data(
        DIGIT_TEST_IMAGES_PATH, DIGIT_TEST_LABELS_PATH, 'digit'
    )

    random.seed(42)

    all_stats = []

    digit_perceptron = PerceptronClassifier(legalLabels=list(set(digit_train_labels)), max_iterations=30)
    all_stats.append(evaluate_model(
        digit_perceptron, "Digit data with Perceptron", digit_train_feats, digit_train_labels, digit_test_feats, digit_test_labels
    ))

    digit_naive_bayes = NaiveBayesClassifier(legalLabels=list(set(digit_train_labels)))
    all_stats.append(evaluate_model(
        digit_naive_bayes, "Digit data with Naive Bayes", digit_train_feats, digit_train_labels, digit_test_feats, digit_test_labels
    ))

    face_perceptron = PerceptronClassifier(legalLabels=list(set(face_train_labels)), max_iterations=30)
    all_stats.append(evaluate_model(
        face_perceptron, "Face data with perceptron", face_train_feats, face_train_labels, face_test_feats, face_test_labels
    ))

    face_naive_bayes = NaiveBayesClassifier(legalLabels=list(set(face_train_labels)))
    all_stats.append(evaluate_model(
        face_naive_bayes, "Face data with naive bayes", face_train_feats, face_train_labels, face_test_feats, face_test_labels
    ))

    display_results(all_stats)
if __name__ == "__main__":
    main()