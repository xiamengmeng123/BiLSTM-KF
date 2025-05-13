import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from parser_my import args
import math



class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx  # 特征数据
        self.y = yy  # 标签数据
        self.tranform = transform # 数据变换（如果有的话）,对数据进行缩放

    def __getitem__(self, index):
        x1 = self.x[index]  # 获取指定索引的特征数据        
        y1 = self.y[index]  # 获取指定索引的标签数据
        if self.tranform != None:
            return self.tranform(x1), y1 # 如果有数据变换，则应用变换
            # return x1, y1
        return x1, y1  # 否则，直接返回数据
 
    def __len__(self):
        return len(self.x) # 返回数据集的长度
        print(f"len(self.x) is : {len(self.x)}") 

def normalize_column(data, col, norm_params):
    """归一化指定列并返回归一化参数"""
    if col == 's1':
        median = data[col].median()
        iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
        norm_params[col] = {'type': 'robust', 'median': median, 'iqr': iqr}
        data.loc[:,col] = (data[col] - median) / iqr  #使用.loc进行赋值
    else:
        min_val = data[col].min()
        max_val = data[col].max()
        norm_params[col] = {'type': 'minmax', 'min': min_val, 'max': max_val}
        data.loc[:,col] = (data[col] - min_val) / (max_val - min_val)

##统一归一化
# def normalize_column(data, col, norm_params):
#     """归一化指定列并返回归一化参数"""
#     min_val = data[col].min()
#     max_val = data[col].max()
#     norm_params[col] = {'type': 'minmax', 'min': min_val, 'max': max_val}
#     data.loc[:,col] = (data[col] - min_val) / (max_val - min_val)

def getData(corpusFile, sequence_length, batchSize):
    # 读取数据并预处理
    stock_data = pd.read_csv(corpusFile)
    stock_data.drop('time', axis=1, inplace=True)
    stock_data = stock_data.loc[:, (stock_data != 0).any(axis=0)]

    # 按时间顺序分割训练集和测试集（假设数据是按时间排序的）
    split_ratio = 0.01  # 1%训练，99%测试
    split_idx = int(len(stock_data) * split_ratio)
    train_data = stock_data.iloc[:split_idx]
    test_data = stock_data.iloc[split_idx:]
    train_data.to_csv("data/ready_train.csv",index=False)
    test_data.to_csv("data/ready_test.csv",index=False)

    # 对特征进行归一化（只在训练集计算参数）
    # 对特征进行归一化
    norm_params = {}
    for col in train_data.columns:
        if col in args.s_columns:
            normalize_column(train_data, col, norm_params)
            normalize_column(test_data, col, norm_params)  # 使用训练集的参数
    
    train_data.to_csv("data/ready_guiyi_train.csv",index=False)
    test_data.to_csv("data/ready_guiyi_test.csv",index=False)

    # 生成时间窗口数据
    def create_dataset(df, seq_length):
        X, Y = [], []
        for i in range(len(df) - seq_length):
            X.append(df.iloc[i:i+seq_length].values.astype(np.float32))   #训练集X的长度为1995 测试集X的长度为198001
            Y.append(df.iloc[i+seq_length].values.astype(np.float32))
        return np.array(X), np.array(Y)

    # 分别在训练集和测试集上生成数据
    trainX, trainY = create_dataset(train_data, sequence_length)
    testX, testY = create_dataset(test_data, sequence_length)

    # 构建DataLoader
    train_loader = DataLoader(Mydataset(trainX, trainY), batch_size=batchSize, shuffle=False)
    test_loader = DataLoader(Mydataset(testX, testY), batch_size=batchSize, shuffle=False)

    return norm_params, train_loader, test_loader
