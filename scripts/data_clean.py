import os
import csv
import pandas as pd
import json



def read_metadata(csv_file_path):
    csv_data = pd.read_csv(csv_file_path, encoding='latin1')
    audio_fnames = list(csv_data['file_name'])
    questions = list(csv_data['QuestionText'])
    answers = list(csv_data['answer'])
    return audio_fnames, questions, answers

def QAClean(csv_file):

	wave_name, question, answer = read_metadata(csv_file)

	cnt = 0
	for index in range(0,len(question),3):

		answer_tmp = []
		# for i in range(3)
		answer_tmp.append(answer[index])
		
		if answer[index+1] in answer_tmp:
			answer_tmp.append(answer[index+1])
			print(wave_name[index], ', "', question[index], '", ', answer[index+1])
			cnt += 1
		else:
			answer_tmp.append(answer[index+1])
			if answer[index+2] in answer_tmp:
				answer_tmp.append(answer[index+2])
				print(wave_name[index], ',"', question[index], '",', answer[index+2])
				cnt += 1
	print("cnt: ", cnt)


def AnswerGen(csv_file):

	csv_data = pd.read_csv(csv_file, encoding='latin1', usecols=['file_name', 'QuestionText', 'answer'])

	csv_data['answer'] = csv_data['answer'].str.upper()

	answers_set = set(list(csv_data['answer']))
	answers_dict = dict(zip(answers_set, range(len(answers_set))))

	with open("../metadata/output_classes_clean.json", "w") as outfile:
	    json.dump(answers_dict, outfile)



if __name__ == "__main__":

	csv_file = "../metadata/binary_test.csv"
	QAClean(csv_file)
	# AnswerGen(csv_file)