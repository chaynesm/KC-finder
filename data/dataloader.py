# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 16:21:21
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 11:47:28
import torch
import torch.utils.data as Data
from data.readdata import DataReader


def getDataLoader(batch_size, num_of_questions, max_step, fold):
#     handle = DataReader('dataset/assist2009/builder_train.csv',
#                         'dataset/assist2009/builder_test.csv', max_step,
#                         num_of_questions)
#     handle = DataReader('dataset/CodeWorkOut/train.csv',
#                         'dataset/CodeWorkOut/test.csv', max_step,
#                         num_of_questions)
    handle = DataReader('dataset/CodeWorkOut/100try_filtered/train_firstatt_'+str(fold)+'.csv',
                        'dataset/CodeWorkOut/100try_filtered/val_firstatt_'+str(fold)+'.csv', max_step,
                        num_of_questions)
#     handle = DataReader('dataset/CodeWorkOut/train_firstatt_1feature_or.csv',
#                         'dataset/CodeWorkOut/test_firstatt_1feature_or.csv',
#                         'dataset/CodeWorkOut/train_firstatt_1feature_wb.csv',
#                         'dataset/CodeWorkOut/test_firstatt_1feature_wb.csv', max_step,
#                         num_of_questions)

    dtrain = torch.tensor(handle.getTrainData().astype(float).tolist(),
                          dtype=torch.float32)
    test_data, test_list = handle.getTestData()
    dtest = torch.tensor(test_data.astype(float).tolist(),
                         dtype=torch.float32)

    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader
