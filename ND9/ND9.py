import pandas as pd
import numpy as np

def calculate_probabilities(labels):
    total_samples = len(labels)
    unique_labels = labels.unique()
    probabilities = {}

    for label in unique_labels:
        count = (labels == label).sum()
        probabilities[label] = count / total_samples

    return probabilities

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def binarize_pixels(pixels, threshold=128):
    return (pixels > threshold).astype(int)

train_pixels = binarize_pixels(train_data.iloc[:, 1:])
test_pixels = binarize_pixels(test_data.iloc[:, 1:])

def train_naive_bayes(labels, pixels):
    probability = calculate_probabilities(train_data.iloc[:, 0])
    conditional = {}
    for digit in range(10):
        dig_px = pixels[labels == digit]
        conditional[digit] = (dig_px.sum(axis=0) + 1) / (dig_px.shape[0] + 2)
    return probability, conditional

probability, conditional = train_naive_bayes(train_data.iloc[:, 0], train_pixels)

def predict_naive_bayes(probability, conditional, pixels):
    predictions = []
    for i in range(len(pixels)):
        px_val = pixels.iloc[i]
        likelihoods = {digit: np.prod(conditional[digit][px_val != 0]) *
                                np.prod(1 - conditional[digit][px_val == 0])
                                for digit in range(10)}
        final_prob = {digit: likelihoods[digit] * probability[digit] for digit in range(10)}
        digit = max(final_prob, key=final_prob.get)
        predictions.append(digit)
    return predictions

prediction = predict_naive_bayes(probability, conditional, test_pixels)

def calculate_accuracy(predictions, label):
    accuracy = (predictions == label).mean()
    return accuracy

label = test_data.iloc[:, 0]
accuracy = calculate_accuracy(prediction, label)
print(f"Accuracy: {accuracy:.2%}")
