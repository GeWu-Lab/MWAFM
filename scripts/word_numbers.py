import os
import csv
import pandas as pd

def read_metadata(csv_file_path):
    csv_data = pd.read_csv(csv_file_path, encoding='latin1')
    audio_fnames = list(csv_data['file_name'])
    questions = list(csv_data['QuestionText'])
    answers = list(csv_data['answer'])
    return audio_fnames, questions, answers

def StatWorsNums(csv_file):

	words = []

	wave_name, question, answer = read_metadata(csv_file)
	# print(question)

	for index in range(len(question)):
		qst = question[index]
		qst = qst.replace(",", "").replace("?", "")
		qst = qst.split(" ")

		for wd in qst:
			if wd not in words:
				words.append(wd)

	print(words)
	print(len(words))




if __name__ == "__main__":

	csv_file = "../metadata/single_word_train.csv"
	StatWorsNums(csv_file)