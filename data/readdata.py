# -*- coding: utf-8 -*-
# Adapted by Yang Shi from jarvis.zhang
import numpy as np
import itertools
import pandas as pd


def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(",")
        if components[0] in node_word_index:
            starting_node = node_word_index[components[0]]
        else:
            starting_node = node_word_index['UNK']
        if components[1] in path_word_index:
            path = path_word_index[components[1]]
        else:
            path = path_word_index['UNK']
        if components[2] in node_word_index:
            ending_node = node_word_index[components[2]]
        else:
            ending_node = node_word_index['UNK']
        
        sample_index.append([starting_node,path,ending_node])
    return sample_index

skill_dict = {0: [0,1,3,5],
 1: [3,5,8],
 2: [1,3,5,8],
 3: [0,4],
 4: [0,2,3,5],
 5: [3,5],
 6: [6],
 7: [5,8],
 8: [0,1,3,5,7,8]}

MAX_CODE_LEN = 100

class DataReader():
    def __init__(self, train_path_1, test_path_1, maxstep, numofques):
        self.train_path = train_path_1
        self.test_path = test_path_1
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self, file_path):
        data = []
        css_list = []
        code_df = pd.read_csv("labeled_paths.tsv",sep="\t")
        training_students = np.load("training_students.npy",allow_pickle=True)
        all_training_code = code_df[code_df['SubjectID'].isin(training_students)]['RawASTPath']
        separated_code = []
        for code in all_training_code:
            if type(code) == str:
                separated_code.append(code.split("\U0001f972"))
        
        node_hist = {}
        path_hist = {}
        for paths in separated_code:
            starting_nodes = [p.split(",")[0] for p in paths]

            path = [p.split(",")[1] for p in paths]
            ending_nodes = [p.split(",")[2] for p in paths]
            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1

        node_count = len(node_hist)
        path_count = len(path_hist)
        np.save("np_counts.npy", [node_count, path_count])

        # small frequency then abandon, for node and path
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)
        
        
        
        
        
        with open(file_path, 'r') as file:
            for lent, css, ques, ans in itertools.zip_longest(*[file] * 4):
                lent = int(lent.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                css = [cs for cs in css.strip().strip(',').split(',')]
                temp = np.zeros(shape=[self.maxstep, 2 * self.numofques+MAX_CODE_LEN*3]) # Skill DKT #1, original
                if lent >= self.maxstep:
                    steps = self.maxstep
                    extra = 0

                    ques = ques[-steps:]
                    ans = ans[-steps:]
                    css = css[-steps:]
                else:
                    steps = lent
                    extra = self.maxstep-steps



                css_list_student = []
                for j in range(steps):
                    if ans[j] == 1:
                        temp[j+extra][ques[j]] = 1
                    else:
                        temp[j+extra][ques[j] + self.numofques] = 1
                            
                    code = code_df[code_df['CodeStateID']==css[j]]['RawASTPath'].iloc[0]
                    
                    css_list_student.append(css[j])
                    if type(code) == str:
                        code_paths = code.split("\U0001f972")
                        raw_features = convert_to_idx(code_paths, node_word_index, path_word_index)
                        if len(raw_features) < MAX_CODE_LEN:
                            raw_features += [[0,0,0]]*(MAX_CODE_LEN - len(raw_features))
                        else:
                            raw_features = raw_features[:MAX_CODE_LEN]
                        

                        features = np.array(raw_features).reshape(-1, MAX_CODE_LEN*3)
                        

                        temp[j+extra][2*self.numofques:] = features
                         
                data.append(temp.tolist())
                css_list.append(css_list_student)
                
            print('done: ' + str(np.array(data).shape))
        return data, css_list

    def getTrainData(self):
        print('loading train data...')
        trainData, css_list = self.getData(self.train_path)

        return np.array(trainData)

    def getTestData(self):
        print('loading test data...')

        testData, css_list = self.getData(self.test_path)
        return np.array(testData), css_list
