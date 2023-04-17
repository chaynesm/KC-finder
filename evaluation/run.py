# -*- coding: utf-8 -*-
# Adapted by Yang Shi from jarvis.zhang
"""
Usage:
    run.py (rnn|sakt) --hidden=<h> [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 10]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 128]
    --layers=<int>                      layers of rnn or transformer [default: 1]
    --heads=<int>                       head number of transformer [default: 8]
    --dropout=<float>                   dropout rate [default: 0.1]
    --model=<string>                    model type
"""

import os
import random
import logging
import torch

import torch.optim as optim
import numpy as np

from datetime import datetime
from docopt import docopt
from data.dataloader import getDataLoader
from evaluation import eval
import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = docopt(__doc__)
    length = int(args['--length'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])
    heads = int(args['--heads'])
    dropout = float(args['--dropout'])
    if args['rnn']:
        model_type = 'RNN'
    elif args['sakt']:
        model_type = 'SAKT'
    '''
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(list(args.items()))
    '''
    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(torch.cuda.current_device())

    
    
    if model_type == 'RNN':
        from model.DKT.RNNModel import RNNModel
        model = RNNModel(questions * 2, hidden, layers, questions, device)
    elif model_type == 'SAKT':
        from model.SAKT.model import SAKTModel
        model = SAKTModel(heads, length-1, hidden, questions, dropout, device)

    
    
    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []
    from model.DKT.c2vRNNModel import c2vRNNModel
    from model.SAKT.model import SAKTModel
    for fold in range(5):
        torch.cuda.empty_cache()
        if model:
            del model
        print(torch.cuda.memory_allocated())
        print("----",fold,"-th run----")
        trainLoader, testLoade = getDataLoader(bs, questions, length, fold)
        node_count, path_count = np.load("np_counts.npy")

        model = c2vRNNModel(questions * 2, hidden, layers, questions, node_count, path_count, questions, device) 

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_func = eval.lossFunc(questions, length, bs, device)
        for epoch in range(epochs):
            print('epoch: ' + str(epoch))
            train_loss, model, optimizer = eval.train_epoch(model, trainLoader, optimizer,
                                              loss_func, device)

            val_loss, (first_total_scores, first_scores, scores, performance) = eval.test_epoch(model, testLoade, loss_func, device, epoch, fold)
            
            print(train_loss, val_loss)
            if val_loss - train_loss > 0.03 and epoch > 20:
                break
        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
        print(performance)
        
        del val_loss, train_loss
    print(np.mean(first_total_scores_list,axis=0))
    print(np.mean(performance_list,axis=0))
    for k in range(50):
        print("Scores")
        print(np.mean([s[k] for s in scores_list],axis=0))
        print("First Scores")
        print(np.mean([s[k] for s in first_scores_list],axis=0))

if __name__ == '__main__':
    main()
