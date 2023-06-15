import os
import time
import pickle
import utils
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

from data_generator import *
from nets.ours_MWAFM_Net import MWAFM_Net
from conifg import data_config, model_config


def train(model, train_iterator, optimizer, criterion, epoch):

    model.train()

    # for batch_idx, (audio_feat, audio_ast_feat, question, label) in enumerate(train_iterator):
    for batch_idx, (audio_feat, question, label) in enumerate(train_iterator):

        audio_feat = audio_feat.to(dtype=torch.float)
        # audio_ast_feat = audio_ast_feat.to(dtype=torch.float)
        question = question.to(dtype=torch.float)
        label = label.to('cuda', dtype=torch.long)

        question_len = torch.ones((question.size(0),), dtype=torch.int8).to('cuda')

        optimizer.zero_grad()
        # logits_output = model(audio_feat, audio_ast_feat, question)
        logits_output = model(audio_feat, question)

        loss = criterion(logits_output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % model_config['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(audio_feat), len(train_iterator.dataset), 100. * batch_idx / len(train_iterator), loss.item()))


def eval(model, val_iterator, optimizer, criterion, epoch):

    model.eval()
    val_acc_top_01 = 0
    val_acc_top_05 = 0
    val_acc_top_10 = 0

    total = 0
    correct_top_01 = 0
    correct_top_05 = 0
    correct_top_10 = 0

    with torch.no_grad():
        # for batch_idx, (audio_feat, audio_ast_feat, question, label) in enumerate(val_iterator):
        for batch_idx, (audio_feat, question, label) in enumerate(val_iterator):
        
            audio_feat = audio_feat.to(dtype=torch.float)
            # audio_ast_feat = audio_ast_feat.to(dtype=torch.float)
            question = question.to(dtype=torch.float)
            label = label.to('cuda', dtype=torch.long)
            question_len = torch.ones((question.size(0),), dtype=torch.int8).to('cuda')
        
            # logits_output = model(audio_feat, audio_ast_feat, question)
            logits_output = model(audio_feat, question)


            total += logits_output.size(0)

            # top-01 accuracy
            _, predicted_top_01 = torch.max(logits_output.data, 1)
            correct_top_01 += (predicted_top_01 == label).sum().item()

            # top-05 and top-20 accuracy
            _, predicted_top_n = torch.sort(logits_output.data, dim=1, descending=True)
            
            predicted_top_05 = predicted_top_n[:, :5].detach().cpu().numpy()
            predicted_top_10 = predicted_top_n[:, :10].detach().cpu().numpy()

            ground_truth = label.detach().cpu().numpy()
            n_batch = ground_truth.shape[0]
            ground_truth = ground_truth.reshape(n_batch, 1)

            correct_top_05 += np.count_nonzero((predicted_top_05-ground_truth)==0)
            correct_top_10 += np.count_nonzero((predicted_top_10-ground_truth)==0)


    val_top_01 = 100 * correct_top_01 / total
    val_top_05 = 100 * correct_top_05 / total
    val_top_10 = 100 * correct_top_10 / total

    print("\nTop-01 Validation set accuracy = %.2f %%" % val_top_01)
    print("Top-05 Validation set accuracy = %.2f %%" % val_top_05)
    print("Top-10 Validation set accuracy = %.2f %%" % val_top_10)

    return val_top_01


def test(model, test_iterator):

    model.eval()
    val_acc_top_01 = 0
    val_acc_top_05 = 0
    val_acc_top_10 = 0

    total = 0
    correct_top_01 = 0
    correct_top_05 = 0
    correct_top_10 = 0

    with torch.no_grad():
        # for batch_idx, (audio_feat, audio_ast_feat, question, label) in enumerate(test_iterator):
        for batch_idx, (audio_feat, audio_feat, question, label) in enumerate(test_iterator):
        
            audio_feat = audio_feat.to(dtype=torch.float)
            # audio_ast_feat = audio_ast_feat.to(dtype=torch.float)
            question = question.to(dtype=torch.float)
            label = label.to('cuda', dtype=torch.long)
            question_len = torch.ones((question.size(0),), dtype=torch.int8).to('cuda')
        
            # logits_output = model(audio_feat, audio_ast_feat, question)
            logits_output = model(audio_feat, question)


            total += logits_output.size(0)

            # top-01 accuracy
            _, predicted_top_01 = torch.max(logits_output.data, 1)
            correct_top_01 += (predicted_top_01 == label).sum().item()

            # top-05 and top-20 accuracy
            _, predicted_top_n = torch.sort(logits_output.data, dim=1, descending=True)
            
            predicted_top_05 = predicted_top_n[:, :5].detach().cpu().numpy()
            predicted_top_10 = predicted_top_n[:, :10].detach().cpu().numpy()

            ground_truth = label.detach().cpu().numpy()
            n_batch = ground_truth.shape[0]
            ground_truth = ground_truth.reshape(n_batch, 1)

            correct_top_05 += np.count_nonzero((predicted_top_05-ground_truth)==0)
            correct_top_10 += np.count_nonzero((predicted_top_10-ground_truth)==0)


    val_top_01 = 100 * correct_top_01 / total
    val_top_05 = 100 * correct_top_05 / total
    val_top_10 = 100 * correct_top_10 / total

    print("\nTop-01 Validation set accuracy = %.2f %%" % val_top_01)
    print("Top-05 Validation set accuracy = %.2f %%" % val_top_05)
    print("Top-10 Validation set accuracy = %.2f %%" % val_top_10)
    print('\n***********************************************************\n')

    # return val_acc_top_01



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio Question Answering')
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        "--model_save_dir", type=str, default='./checkpoints/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='MWAFM_Net',help="save model name")
    parser.add_argument(
        '--seed', type=int, default=8888, metavar='S',help='random seed (default: 1)')
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    
    model = MWAFM_Net().to('cuda')
    model = nn.DataParallel(model)
    model = model.to('cuda')

    if args.mode == "train":
        print("\n-------------------- Multi-scale Window-size Attention Model Training --------------------")
        # create data iterator
        # train_dataset = DataGenerator(data_config['train_metadata_path'])
        # train_dataset = DataGenerator(data_config, model_config)
        train_dataset = DataGenerator(data_config['train_metadata_path'])
        train_iterator = DataLoader(dataset=train_dataset, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'], 
                                    pin_memory=True, shuffle=True)
        # val_dataset = DataGenerator(data_config['val_metadata_path'])
        val_dataset = DataGenerator(data_config['val_metadata_path'])
        val_iterator = DataLoader(dataset=val_dataset, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'], 
                                  pin_memory=True, shuffle=True)
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=model_config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        best_acc = 0
        best_epoch = 0
        for epoch in range(start_epoch, model_config['num_epochs']):
            train(model, train_iterator, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_iterator, optimizer, criterion, epoch)
            if F >= best_acc:
                best_acc = F
                best_epoch = epoch
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
            print("\nTop-01 training-val best: epoch {},  acc: {:.2f}%".format(best_epoch, best_acc))
            print('\n***********************************************************\n')
    else:
        print("\n-------------------- Multi-scale Window-size Attention Model Testing --------------------")
        test_dataset = DataGenerator(data_config, model_config, mode='test')
        test_iterator = DataLoader(dataset=test_dataset, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'], 
                                   pin_memory=True, shuffle=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        test(model, test_iterator)

