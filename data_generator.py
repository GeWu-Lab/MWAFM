import os
import torch
from torch.utils.data.dataset import Dataset
from utils import load_vectors, read_metadata, load_answers_dict
import numpy as np

import torch.nn.functional as F
from conifg import data_config, model_config


class DataGenerator(Dataset):

    def __init__(self, meta_file):
        super(DataGenerator, self).__init__()

        self.meta_file = meta_file

        self.feat_dir = data_config['feat_dir']
        # self.feat_ast_dir = data_config['feat_ast_dir']

        self.audio_fnames, self.qs, self.ans = read_metadata(self.meta_file)

        self.batch_size = model_config['batch_size']
        self.audio_length = data_config['audio_length']
        self.qust_max_len = data_config['quest_length']

        self.word_embedding_path = data_config['pre_trained_word_embeddings_file']
        self.word_embeddings = load_vectors(self.word_embedding_path)  # dict of all the {'word': [vector]} pairs
        self.answers_dict = load_answers_dict(data_config['output_classes_file'])

    def __getitem__(self, item):

        audio_feat = self.load_audio_features(item)

        audio_name = self.audio_fnames[item][:-3] + 'npy'
        # audio_ast_feat = np.load(os.path.join(self.feat_ast_dir, audio_name))

        question_text = self.qs[item]
        answer_text = self.ans[item]
        question_embedding = self.get_word_embeddings(question_text)

        if 'binary' in self.meta_file:
            if answer_text == 'YES':
                label = 0
            else:
                label = 1
        else:
            label = self.answers_dict[answer_text]

        # return audio_feat, audio_ast_feat, question_embedding, label
        return audio_feat, question_embedding, label

    def load_audio_features(self, idx):
        # audio_feat_file = self.audio_fnames[idx][:-3] + 'npz'
        audio_feat_file = self.audio_fnames[idx][:-3] + 'npy'
        data = np.load(os.path.join(self.feat_dir, audio_feat_file))
        # return data['embedding']

        ## -------------------------------------------------------------------------------
        ## ensure audio length equal
        if self.batch_size != 1:
            data1 = torch.from_numpy(data)
            data2 = data1.unsqueeze(0).permute(0, 2, 1).contiguous()
            data3 = F.interpolate(data2, size=self.audio_length, mode='linear', align_corners=False)
            data4 = data3.permute(0, 2, 1).contiguous()
            data5 = data4.squeeze()
            data = data5.numpy()
        ## -------------------------------------------------------------------------------

        return data

    def get_word_embeddings(self, input_text):

        words = input_text.split(' ')
        words[-1] = words[-1][:-1]  # removing '?' from the question, repetitive in all the Qs, so adds no value

        ## -------------------------------------------------------------------------------
        if len(words) < self.qust_max_len:
            dn = self.qust_max_len - len(words)
            for index in range(dn):
                words.append("0")
        else:
            words = words[0:self.qust_max_len]
        ## -------------------------------------------------------------------------------

        text_embedding = []
        for word in words:
            # word = word.split(",")[0]
            try:
                embedding = self.word_embeddings[word]
            except KeyError:
                continue
            text_embedding.append(embedding)

        text_embedding = np.array(text_embedding)

        ## -------------------------------------------------------------------------------
        # if text_embedding.shape[0] < self.qust_max_len:
        #     ddn = self.qust_max_len - text_embedding.shape[0]
        #     pad_value = np.repeat(text_embedding[-1], ddn)
        #     text_embedding = np.append(text_embedding, pad_value)
        #     text_embedding = text_embedding.reshape(self.qust_max_len, -1)
        ## -------------------------------------------------------------------------------

        ## -------------------------------------------------------------------------------
        text_embedding1 = torch.from_numpy(text_embedding)
        text_embedding2 = text_embedding1.unsqueeze(0).permute(0, 2, 1).contiguous()
        text_embedding3 = F.interpolate(text_embedding2, size=self.qust_max_len, mode='linear', align_corners=False)
        text_embedding4 = text_embedding3.permute(0, 2, 1).contiguous()
        text_embedding5 = text_embedding4.squeeze()
        text_embedding = text_embedding5.numpy()
        ## -------------------------------------------------------------------------------

        return text_embedding

    def __len__(self):
        return len(self.qs)
