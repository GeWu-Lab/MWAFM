data_config = {

    'train_metadata_path': 'metadata/clotho_aqa_train_clean.csv',  # CSV containing audio URLs, Questions, Answers,filenames
    'val_metadata_path': 'metadata/clotho_aqa_val_clean.csv',
    'test_metadata_path': 'metadata/clotho_aqa_test_clean.csv',
    'output_classes_file': 'metadata/output_classes_clean.json',

    'data_dir': '/home/data/clotho-aqa/audio_16kHz',  # path to store downloaded data
    'feat_dir': '/home/data/clotho-aqa/vggish',
    'feat_ast_dir': '/home/data/clotho-aqa/feats/ast',
    # 'feat_dir': '/home/guangyao_li/dataset/clotho-aqa/audio_spec',
    'question_dir': './metadata/questions.csv',
    'pre_trained_word_embeddings_file': './pretrained/wiki-news-300d-1M.vec',
    'audio_embedding_size': 512,

    # audio length
    'audio_length': 24,
    'quest_length': 22,
}

model_config = {

    'learning_rate': 1e-4,
    'batch_size': 64,
    'num_workers': 12,
    'num_epochs': 50,
    'log_interval': 10,


    # audio network
    'audio_input_size': data_config['audio_embedding_size'],
    'audio_lstm_n_layers': 2,
    'audio_lstm_hidden_size': 128,
    'audio_bidirectional': True,
    'audio_lstm_dropout': 0.2,


    # NLP network
    'text_input_size': 300,  # pretrained embedding size from fasttext
    'text_lstm_n_layers': 2,
    'text_lstm_hidden_size': 128,
    'text_bidirectional': True,
    'text_lstm_dropout': 0.2,

    # classification
    'n_dense1_units': 256,
    'n_dense2_units': 128,
}



if 'binary' in data_config['train_metadata_path']:
    model_config['n_classes'] = 1
else:
    model_config['n_classes'] = 828
    model_config['audio_lstm_hidden_size'] = 512
    model_config['text_lstm_hidden_size'] = 512

dense1_input = 0
if model_config['audio_bidirectional']:
    dense1_input = dense1_input + 2 * model_config['audio_lstm_hidden_size']
else:
    dense1_input = dense1_input + model_config['audio_lstm_hidden_size']

if model_config['text_bidirectional']:
    dense1_input = dense1_input + 2 * model_config['text_lstm_hidden_size']
else:
    dense1_input = dense1_input + model_config['text_lstm_hidden_size']

model_config['dense1_input'] = dense1_input
