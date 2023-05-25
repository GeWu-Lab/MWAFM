import pandas as pd
import io
import numpy as np
import json


def load_vectors(embedding_file):
    fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def read_metadata(csv_file_path):
    csv_data = pd.read_csv(csv_file_path, encoding='latin1')
    audio_fnames = list(csv_data['file_name'])
    questions = list(csv_data['QuestionText'])
    answers = list(csv_data['answer'])
    return audio_fnames, questions, answers


def binary_classification_accuracy(pred, ground_truth):
    n_samples = pred.shape[0]
    # x = pred - ground_truth
    x = pred - ground_truth.reshape(n_samples, 1)
    n_wrong_predictions = np.count_nonzero(x)
    accuracy = (n_samples - n_wrong_predictions) / n_samples
    return accuracy


def multiclass_classification_accuracy(logits, ground_truth, k=1):   # b x 830  bx1
    n_samples = logits.shape[0]
    if k == 1:
        prediction = np.argmax(logits, axis=1)
        x = prediction - ground_truth
        n_wrong_predictions = np.count_nonzero(x)        
        accuracy = (n_samples - n_wrong_predictions) / n_samples
        return accuracy
    else:
        max_idx = np.argsort(-1*logits, 1)[:, :k]       # np.argsort() 返回数组值从小到大的索引值
        # x = max_idx - ground_truth
        x = max_idx - ground_truth.reshape(n_samples, 1)
        n_correct_predictions = np.count_nonzero(x == 0)
        top_k_accuracy = n_correct_predictions/n_samples
        return top_k_accuracy



def load_answers_dict(answers_dict_file):
    f = open(answers_dict_file)
    answers_dict = json.load(f)
    return answers_dict
