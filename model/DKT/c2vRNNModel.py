# -*- coding: utf-8 -*-
# Adapted by Yang Shi from jarvis.zhang
import torch
import torch.nn as nn
MAX_CODE_LEN = 100

class c2vRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, questions, device):
        super(c2vRNNModel, self).__init__()

        self.skill_n = 30
        self.embed_nodes = nn.Embedding(node_count+2, 100) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, 100) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(300,300)
        self.attention_layer = nn.Linear(300,1)

        self.prediction_layer = nn.Linear(300,1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(2*input_dim+300,
                          self.skill_n,
                          layer_dim,
                          batch_first=True)
        self.rnn2 = nn.LSTM(self.skill_n,
                          self.output_dim,
                          layer_dim,
                          batch_first=True)

        self.fc_skill = nn.Linear(300, self.skill_n)
        self.fc_pred = nn.Linear(self.skill_n, self.output_dim)
        self.kc_selecting_fc = nn.Linear(questions, self.skill_n)
        self.kc_selecting_fc.weight.data = self.kc_selecting_fc.weight.data+0.15
        self.dropout = nn.Dropout(p=0.1)
        self.leakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, evaluating=False):  # shape of input: [batch_size, length, questions * 2+c2vnodes]
        
        rnn_first_part = x[:, :, :50] # (b,l,q)
        
        print(rnn_first_part.shape, x.shape)

        c2v_input = x[:, :, 50:].reshape(x.size(0), x.size(1), MAX_CODE_LEN, 3).long() # (b,l,c,3)

        starting_node_index = c2v_input[:,:,:,0] # (b,l,c,1)
        ending_node_index = c2v_input[:,:,:,2] # (b,l,c,1)
        path_index = c2v_input[:,:,:,1] # (b,l,c,1)

        starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,c,1) -> (b,l,c,ne)
        ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,c,1) -> (b,l,c,ne)
        path_embed = self.embed_paths(path_index) # (b,l,c,1) -> (b,l,c,pe)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed), dim=3) # (b,l,c,2ne+pe+q)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed) # (b,l,c,2ne+pe+2q)
        
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed)) # (b,l,c,2ne+pe+2q)
        context_weights = self.attention_layer(full_embed_transformed) # (b,l,c,1)
        attention_weights = self.attention_softmax(context_weights) # (b,l,c,1)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2) # (b,l,2ne+pe+2q)
        
        kc_selecting_mask = self.sig(self.kc_selecting_fc(rnn_first_part))

        out = self.fc_skill(code_vectors)

        res2 = out * (kc_selecting_mask.ge(0.5).float())

        out2 = self.fc_pred(res2)  # shape of res: [batch_size, length, question]
        res = self.sig(out2)
        return res, res2, kc_selecting_mask, attention_weights
