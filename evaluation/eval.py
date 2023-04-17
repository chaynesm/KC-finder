# -*- coding: utf-8 -*-
# Adapted by Yang Shi from jarvis.zhang
import tqdm
import torch
import logging
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger('main.eval')

def performance_granular(batch, pred, ground_truth, prediction, epoch):
    
    preds = {k:[] for k in range(50)}
    gts = {k:[] for k in range(50)}
    first_preds = {k:[] for k in range(50)}
    first_gts = {k:[] for k in range(50)}
    scores = {}
    first_scores = {}

    
    for s in range(pred.shape[0]):
        
        # Current problem as ground truth   
        delta = (batch[s][:, 0:50] + batch[s][:, 50:100])
        temp = pred[s][:50].mm(delta.T)
        
        index = torch.tensor([[i for i in range(50)]],
                             dtype=torch.long)
        
        
        p = temp.gather(0, index)[0].detach().cpu().numpy()
        a = (((batch[s][:, 0:50] - batch[s][:, 50:100]).sum(1) + 1) // 2).detach().cpu().numpy()

        for i in range(len(p)):
            if p[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta.detach().cpu().numpy()[i:]
                break
        
        
        for i in range(len(p)):
            for j in range(50):
                if delta[i,j] == 1:
                    preds[j].append(p[i])
                    gts[j].append(a[i])
                    if i == 0 or delta[i-1,j] != 1:
                        first_preds[j].append(p[i])
                        first_gts[j].append(a[i])
    first_total_gts = []
    first_total_preds = []
    for j in range(50):
        f1 = metrics.f1_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        recall = metrics.recall_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        precision = metrics.precision_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        acc = metrics.accuracy_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        try:
            auc = metrics.roc_auc_score(gts[j], preds[j])
        except ValueError:
            auc = 0.5
        scores[j]=[auc,f1,recall,precision,acc]

        
        
        first_f1 = metrics.f1_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_recall = metrics.recall_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_precision = metrics.precision_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_acc = metrics.accuracy_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        try:
            first_auc = metrics.roc_auc_score(first_gts[j], first_preds[j])
        except ValueError:
            first_auc = 0.5
            
        first_total_gts.extend(first_gts[j])
        first_total_preds.extend(first_preds[j])
        
        first_scores[j]=[first_auc,first_f1,first_recall,first_precision,first_acc]
    
    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    acc = metrics.accuracy_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    auc = metrics.roc_auc_score(
        ground_truth.detach().numpy(),
        prediction.detach().numpy())
    logger.info('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision) + ' acc: ' +
                str(acc))

    
    
    
    first_total_f1 = metrics.f1_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_recall = metrics.recall_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_precision = metrics.precision_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_acc = metrics.accuracy_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    try:
        first_total_auc = metrics.roc_auc_score(first_total_gts, first_total_preds)
    except ValueError:
        first_total_auc = 0.5
    
    first_total_scores = [first_total_auc,first_total_f1,first_total_recall,first_total_precision,first_total_acc]
    
    return first_total_scores, first_scores, scores, [auc,f1,recall,precision,acc]

def plot_heatmap(batch, pred, fold, batch_n):
    
    problem_dict = {"000000010":"1",
                    "000000001":"3",
                    "000010000":"5",
                    "010000000":"13",
                    "001000000":"232",
                    "000100000":"233",
                    "100000000":"234",
                    "000001000":"235",
                    "000000100":"236"
                   }
    problems = []
    for s in range(pred.shape[0]):
        
        delta = (batch[s][1:, 0:10] + batch[s][1:, 10:20]).detach().cpu().numpy()
        
        a = (((batch[s][:, 0:10] - batch[s][:, 10:20]) + 1) // 2)[1:].detach().cpu().numpy()
        
        
        p = pred[s].detach().cpu().numpy()

        for i in range(len(delta)):
            if np.sum(delta, axis=1)[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta[i:]
                break
        
        problems = [problem_dict["".join([str(int(i)) for i in sk])] for sk in delta]
        
        plt.figure(figsize=(15, 6), dpi=80)
    
        ax = sns.heatmap(p.T, annot=a.T, linewidth=0.5, vmin=0, vmax=1, cmap="Blues")

        plt.xticks(np.arange(len(problems))+0.5, problems, rotation=45)
        plt.yticks(np.arange(10)+0.5, ['234', '13', '232', '233', '5', '235', '236', '1', '3'], rotation=45)
        plt.xlabel("Attempting Problem")
        plt.ylabel("Problem")

        
        plt.title("Heatmap for student "+str(s)+" fold "+str(fold))
        plt.tight_layout()
        plt.savefig("heatmaps/b"+str(batch_n)+"_s"+str(s)+"_f"+str(fold)+".png")
        
        
def performance(ground_truth, prediction, epoch):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().numpy(),
                                             prediction.detach().numpy())
    auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig("e-"+str(epoch)+".png")
    
    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.round(prediction).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.round(prediction).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.round(prediction).detach().numpy())
    acc = metrics.accuracy_score(
        ground_truth.detach().numpy(),
        torch.round(prediction).detach().numpy())
    auc = metrics.roc_auc_score(
        ground_truth.detach().numpy(),
        prediction.detach().numpy())
    logger.info('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision) + ' acc: ' +
                str(acc))
    print('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision) + ' acc: ' +
                str(acc))
    return [auc,f1,recall,precision,acc]


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, bs, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.MSE = nn.MSELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.batch_size = bs
        self.device = device
    
    
    
    def forward(self, pred, pred_skill, kc_selecting_mask, batch, model):
        
        first_error_rate = 0.7
        learning_rate = 0.6
        prediction_weight = 0.8
        
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        pred = pred.to('cpu')
        pred_skill = pred_skill
        for student in range(pred.shape[0]):

            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:100]  # shape: [length, questions]
            
            # Current problem as ground truth       
            temp = pred[student][:self.max_step].mm(delta.t())
            index = torch.tensor([[i for i in range(self.max_step)]],
                                 dtype=torch.long)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:100]).sum(1) + 1) //
                 2)
            
            for i in range(len(p)):
                if p[i] > 0:
                    p = p[i:]
                    a = a[i:]
                    break


            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        
        
        loss = loss/pred.shape[0]
        
        lc = torch.sum(pred_skill, dim=0)
        kc_mask = (torch.sum(kc_selecting_mask, dim=0)/pred.shape[0])
        kc_mask_norm = torch.norm(kc_mask, p=1)
        kc_mask = kc_mask.ge(0.5)

        lc_loss = 0
        lc_calc = {}
        for i in range(lc.shape[1]):
            lc_component = torch.masked_select(lc[:,i]/pred.shape[0], kc_mask[:,i])
            
            if lc_component.shape[0]:
                opportunities = lc_component.shape[0]

                fitting_lc = first_error_rate*np.arange(1,opportunities+1)**-learning_rate

                lc_loss += self.MSE(torch.reshape(lc_component, (-1,)), torch.tensor(fitting_lc, dtype=torch.float).to(self.device))
                lc_calc[i] = lc_component.detach().cpu().numpy()

            total_loss = prediction_weight*loss + (1-prediction_weight)*torch.sqrt(lc_loss) + 1e-3*torch.norm(model.kc_selecting_fc.weight, p=1)

        
        return total_loss, lc_loss, prediction, ground_truth, pred_skill, kc_selecting_mask
    
    

def train_epoch(model, trainLoader, optimizer, loss_func, device):
    model.to(device)
    lc_total_loss = 0
    total_loss = 0
    count = 0
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
#         batch_new = batch[:,:-1,:].to(device)
        batch_new = torch.cat((batch[:,:,50:100] + batch[:,:,:50],batch[:,:,100:]), dim=2).to(device)
        pred, pred_skill, kc_selecting_mask, attention_weights = model(batch_new)
        loss, lc_loss, prediction, ground_truth, lc, kc_mask = loss_func(pred, pred_skill, kc_selecting_mask, batch[:,:,:100], model)
        optimizer.zero_grad()
        
        lc_total_loss += lc_loss
        total_loss += loss
        count+=1
        
        loss.backward()
        optimizer.step()
    return total_loss/count, model, optimizer


def test_epoch(model, testLoader, loss_func, device, epoch, fold):
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    lcs = []
    kc_masks = []
    attention_weights = torch.tensor([])
    batch_n = 0
    lc_total_loss = 0
    total_loss = 0
    for batch in tqdm.tqdm(testLoader, desc='Validaing:     ', mininterval=2):

        batch_new = torch.cat((batch[:,:,50:100] + batch[:,:,:50],batch[:,:,100:]), dim=2).to(device)
        pred, pred_skill, kc_selecting_mask, attention_weight = model(batch_new)
    
        
        loss, lc_loss, p, a, lc, kc_mask = loss_func(pred, pred_skill, kc_selecting_mask, batch[:,:,:100], model)
        lc_total_loss += lc_loss
        total_loss += loss

        
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        full_data = torch.cat([full_data, batch])
        attention_weights = torch.cat([attention_weights, attention_weight.to('cpu')])
        preds = torch.cat([preds, pred.cpu()])
        lcs.append(lc.cpu().detach().numpy())
        kc_masks.append(kc_mask.cpu().detach().numpy())

        batch_n += 1
        

    print(loss)
    print("LC loss:", lc_total_loss)
    np.save("attention.npy", attention_weights.detach().numpy())
    np.save("lcs"+str(fold)+".npy", lcs)
    np.save("kcs"+str(fold)+".npy", kc_masks)

    torch.save(model.state_dict(), "model"+str(fold)+".pth")

    return total_loss, performance_granular(full_data, preds, ground_truth, prediction, epoch)

