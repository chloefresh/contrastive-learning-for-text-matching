# -*- encoding: utf-8 -*-

import random
import time
from typing import List
import transformers
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, AutoModel, AutoConfig
from sklearn.metrics import f1_score, recall_score, accuracy_score
transformers.logging.set_verbosity_error()

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-5
MAXLEN = 64
POOLING = 'cls'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu') 


ALBERT_SMALL = '/nfs/PretrainedLM/albert_chinese_small/'
MACBERT = '/nfs/PretrainedLM/hfl/chinese-macbert-base'
model_path = MACBERT


SAVE_PATH = './saved_matchmodel_simcse_ce_macbert_cls/simcse_ce_sup.pt'


SNIL_TRAIN = './datasets/match_train.txt'
STS_DEV = '/raw_data/validation.json'
STS_TEST = '/raw_data/test.json'


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集
    """

    def load_snli_data(path):        
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]
        
    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f] 
        
    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:            
            return [(line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]) for line in f]   
    
    def load_match_data(path):
        with jsonlines.open(path, 'r') as f:
            return [(line['origin_a'], line['origin_b'], line['label']) for line in f]
        
        
    assert name in ["snli", "lqcmc", "sts","match"]
    if name == 'snli':
        return load_snli_data(path)   
    if name == 'match':
        return load_match_data(path)
    return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path) 
    

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        simcse_tokens = tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
        class_pos_tokens = tokenizer(text[0], text[1], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
        class_neg_tokens = tokenizer(text[0], text[2], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
        return [simcse_tokens, class_pos_tokens, class_neg_tokens]
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    
    
class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(float(line[2]))

class TestDataset2(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text[0],text[1], max_length=MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    def label(self, text):
        return int(float(text[2]))
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id(line), self.label(line)
    
    
class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)
        self.config = BertConfig.from_pretrained(pretrained_model)
        self.pooling = pooling
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                input_ids_pos=None,attention_mask_pos=None,token_type_ids_pos=None,
                input_ids_neg=None,attention_mask_neg=None,token_type_ids_neg=None):
        
        if not self.training:
            out_pos = self.bert(input_ids_pos, attention_mask_pos, token_type_ids_pos, output_hidden_states=True)
            logits_pos = self.classifier(self.dropout(out_pos.pooler_output))

            return logits_pos
        
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
   
        out_pos = self.bert(input_ids_pos, attention_mask_pos, token_type_ids_pos, output_hidden_states=True)
        out_neg = self.bert(input_ids_neg, attention_mask_neg, token_type_ids_neg, output_hidden_states=True)


        logits_pos = self.classifier(self.dropout(out_pos.pooler_output))
        logits_neg = self.classifier(self.dropout(out_neg.pooler_output))

        if self.pooling == 'cls':
            simcse_out = out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            simcse_out = out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            simcse_out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            simcse_out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
        
        return simcse_out, logits_pos, logits_neg
                  
            

def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

def compute_f1_recall_acc_binary(y_pred, y_true):
    '''
    Compute F1, recall, and accuracy for binary classification problem.
    y_pred: predicted labels, shape (N, 1)
    y_true: true labels, shape (N, 1)
    '''
    y_pred = y_pred.cpu().numpy().flatten()
    y_true = y_true.flatten()
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, recall, accuracy
    

def eval(model, dataloader):
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    predicted_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            source_pred = model(None,None,None,source_input_ids, source_attention_mask, source_token_type_ids,None,None,None)

            _, predicted = torch.max(source_pred, dim=1)
             
            predicted_tensor = torch.cat((predicted_tensor, predicted), dim=0)
            label_array = np.append(label_array, np.array(label))
            
    f1, recall, accuracy = compute_f1_recall_acc_binary(predicted_tensor, label_array)

    # corrcoef       
    return accuracy
        

def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数 
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]

        simcse_source = source[0]
        class_pos_source = source[1]
        class_neg_source = source[2]

        real_batch_num = simcse_source.get('input_ids').shape[0]

        #二分类-正样本
        input_ids_pos = class_pos_source.get('input_ids').to(DEVICE).view(real_batch_num, -1).to(DEVICE)
        attention_mask_pos = class_pos_source.get('attention_mask').to(DEVICE).view(real_batch_num, -1).to(DEVICE)
        token_type_ids_pos = class_pos_source.get('token_type_ids').to(DEVICE).view(real_batch_num, -1).to(DEVICE)
        #二分类-负样本
        input_ids_neg = class_neg_source.get('input_ids').to(DEVICE).view(real_batch_num, -1).to(DEVICE)
        attention_mask_neg = class_neg_source.get('attention_mask').to(DEVICE).view(real_batch_num, -1).to(DEVICE)
        token_type_ids_neg = class_neg_source.get('token_type_ids').to(DEVICE).view(real_batch_num, -1).to(DEVICE)

        #对比学习样本
        input_ids = simcse_source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = simcse_source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = simcse_source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)

        # 训练
        simcse_out, logits_pos, logits_neg = model(input_ids, attention_mask, token_type_ids,input_ids_pos,attention_mask_pos,token_type_ids_pos,
                input_ids_neg,attention_mask_neg,token_type_ids_neg)

        pos_loss = F.cross_entropy(logits_pos.view(-1, 2), torch.ones(real_batch_num, dtype=torch.int64).to(DEVICE))
        neg_loss = F.cross_entropy(logits_neg.view(-1, 2), torch.zeros(real_batch_num, dtype=torch.int64).to(DEVICE))

        contrastive_loss = simcse_sup_loss(simcse_out)


        loss = 0.2*contrastive_loss + 0.4*pos_loss + 0.4*neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')

            accuracy = eval(model, dev_dl)
            model.train()
            if best < accuracy:
                early_stop_batch = 0
                best = accuracy
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher accuracy: {best:.4f} in batch: {batch_idx}, save model")
                
                continue
            early_stop_batch += 1
            if early_stop_batch == 10:
                logger.info(f"accuracy doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {(batch_idx - 10) * BATCH_SIZE}")
                return 
    
    
if __name__ == '__main__':
    
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data('snli', SNIL_TRAIN)
    random.shuffle(train_data)                        
    dev_data = load_data('match', STS_DEV)
    test_data = load_data('match', STS_TEST)    
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE,shuffle=True)
    dev_dataloader = DataLoader(TestDataset2(dev_data), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset2(test_data), batch_size=BATCH_SIZE)
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    best = 0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    model.load_state_dict(torch.load(SAVE_PATH))
    dev_accuracy = eval(model, dev_dataloader)
    test_accuracy = eval(model, test_dataloader)

    logger.info(f'dev_accuracy: {dev_accuracy:.4f}')
    logger.info(f'test_accuracy: {test_accuracy:.4f}')
    