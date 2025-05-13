import torch
import torch.nn as nn
from kan_my import KAN

#LSTM模型输出后添加注意力层
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)  #将lstm的输出（hidden_dim维）映射到1维，即每个时间步生成一个分数=value和权重矩阵（128,1）相乘+bias

    def forward(self, lstm_output):
        # lstm_output.shape: (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attention(lstm_output).squeeze(2), dim=1)  #对每个时间步的分数进行归一化，获得某个时间步的最高权重
        context = torch.sum(attn_weights.unsqueeze(2) * lstm_output, dim=1)  #加权求和得到上下文向量，主要反映最高权重的时间步特征 
        # 形状：(64,5,1) * (64,5,128) → (64,5,128) # dim=1 沿时间步维度求和 # 最终形状 → (64, 128)
        return context
    
#举例理解说明注意力机制    
# 输入序列: ["I", "love", "natural", "language", "processing"]
#               ↓
# 嵌入层: (5, 300)  # seq_len=5, embedding_dim=300
#               ↓
# 双向LSTM:
#    out.shape = (1, 5, 128)  # batch=1, seq_len=5, hidden_dim=128
#               ↓
# 注意力层:
#    scores = W * out → (1,5,1)
#    attn_weights = softmax(scores) → [0.1, 0.6, 0.2, 0.05, 0.05]  #5个时间步的归一化
#               ↓
# 上下文向量: 0.1*h1 + 0.6*h2 + 0.2*h3 + ... → (1,128)
#               ↓
# KAN/FC层 → 分类结果

# #单向lstm+KAN
# class lstm(nn.Module):
#     def __init__(self, input_size=7, hidden_size=64, num_layers=1, output_size=7, 
#                  dropout=0.1, batch_first=True):
#         super(lstm, self).__init__()
#         # LSTM参数配置（显式设置bidirectional=False）
#         self.rnn = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=batch_first,
#             dropout=dropout if num_layers>1 else 0,
#             bidirectional=False  # 显式声明单向LSTM 
#         )
        
#         # 注意力机制（适配单向LSTM的hidden_dim）
#         self.attention = Attention(hidden_size)  # hidden_size而非2*hidden_size 
        
#         # # 正则化层
#         # self.dropout = nn.Dropout(dropout)
#         # self.bn = nn.BatchNorm1d(hidden_size)
        
#         # 分类层
#         self.kan = KAN([hidden_size, 64, output_size])

#     def forward(self, x):
#         # LSTM前向传播
#         lstm_out, (h_n, c_n) = self.rnn(x)  # out.shape = (batch, seq_len, hidden_size)
        
#         # 注意力计算
#         context = self.attention(lstm_out)  # (batch, hidden_size)
        
#         # # 特征正则化
#         # context = self.dropout(context)
#         # context = self.bn(context)  # BN层在dropout后 
        
#         # 最终分类
#         output = self.kan(context)
#         return output


#双向lstm+KAN
class lstm(nn.Module): # 定义一个名为lstm的类，继承自nn.Module
    # 初始化函数，定义模型的各层和参数
    def __init__(self, input_size=7, hidden_size=64, num_layers=1 , output_size=7 , dropout=0.1, batch_first=True,bidirectional=True):
        super(lstm, self).__init__()  # 调用父类的构造函数
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size # 设置LSTM的隐藏层大小
        self.input_size = input_size # 设置LSTM的输入特征维度
        self.num_layers = num_layers # 设置LSTM的层数
        self.output_size = output_size  # 设置输出的维度
        self.dropout = dropout if num_layers>1 else 0 # 设置Dropout概率
        self.batch_first = batch_first # 设置batch_first参数，决定输入输出张量的维度顺序
        self.attention = Attention(hidden_size * 2)  #双向LSTM需隐藏层翻倍
        # self.attention = MultiHeadAttention(hidden_size * (2 if bidirectional else 1),num_heads)  #双向LSTM需隐藏层维度翻倍
        # self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout ) # 定义LSTM层
        #self.kan = KAN([self.hidden_size,64,self.output_size])
        #双向LSTM
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        # 如果使用双向LSTM，隐藏层的输出维度会翻倍
        self.kan = KAN([self.hidden_size * (2 if bidirectional else 1), 64, self.output_size])
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(self.hidden_size * (2 if bidirectional else 1))        

    def forward(self, x):  # 前向传播函数
         # 通过LSTM层，得到输出out和隐藏状态hidden, cell
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # print(f"LSTM的输shape :***************************** {out.shape}")  #([64,5,128])
        # 如果是双向LSTM，取最后一个时间步的前向和后向隐藏状态
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)   #(batch_size, hidden_size * 2)
        else:
            hidden = hidden[-1]
        # Batch Normalization
        hidden = self.bn(hidden)
        
        # Dropout
        hidden = self.dropout_layer(hidden)    
        # # 通过KAN层
        # out = self.kan(hidden)

        context = self.attention(out)  # 使用注意力加权后的上下文向量    
        out = self.kan(context) 

        return out # 返回输出

##单向LSTM+全连接层
# class lstm(nn.Module):
#     def __init__(self, 
#                  input_size=7, 
#                  hidden_size=32, 
#                  num_layers=1,
#                  output_size=7,
#                  dropout=0.2,         # 更合理的默认dropout值
#                  fc_hidden_dim=64,    # 全连接层隐藏维度
#                  batch_first=True):
#         """
#         参数说明：
#         input_size: 输入特征维度（与原始LSTM保持一致）
#         hidden_size: LSTM隐藏层维度
#         num_layers: LSTM堆叠层数
#         output_size: 输出维度
#         dropout: 仅在多层LSTM时生效
#         fc_hidden_dim: 全连接层中间维度（原KAN的64维）
#         """
#         super(lstm, self).__init__()
        
#         # LSTM模块配置
#         self.rnn = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=batch_first,
#             dropout=dropout if num_layers>1 else 0  # 多层时启用dropout
#         )
        
#         # 全连接模块配置
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, fc_hidden_dim),
#             nn.ReLU(),                     # 添加非线性激活
#             nn.Dropout(p=0.3),             # 防止过拟合
#             nn.Linear(fc_hidden_dim, output_size)
#         )

#     def forward(self, x):
#         """
#         前向传播流程：
#         1. LSTM提取时序特征
#         2. 取最后一个时间步的隐藏状态
#         3. 通过全连接网络
#         """
#         # LSTM输出形状：(batch, seq_len, hidden_size)
#         rnn_out, (hidden, cell) = self.rnn(x)  
        
#         # 获取最后一个时间步的隐藏状态（适用于多层LSTM）
#         # hidden.shape = (num_layers, batch, hidden_size)
#         last_hidden = hidden[-1]  # 取最后一层的输出 (batch, hidden_size)
        
#         # 全连接网络处理
#         output = self.fc(last_hidden)
#         return output


# # 双向LSTM网络+全连接层
# class lstm(nn.Module):
#     def __init__(
#         self,
#         input_size=7,          # 输入特征维度
#         hidden_size=64,        # LSTM隐藏层维度
#         num_layers=1,          # LSTM层数
#         output_size=7,         # 输出维度
#         dropout=0.2,           # Dropout概率（仅对多层LSTM生效）
#         bidirectional=True,    # 是否使用双向LSTM
#         batch_first=True       # 输入输出格式是否为(batch, seq, feature)
#     ):
#         super(lstm, self).__init__()
#         self.bidirectional = bidirectional

#         # 双向LSTM层
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=batch_first,
#             dropout=dropout if num_layers > 1 else 0,
#             bidirectional=bidirectional  # 关键修改：启用双向
#         )

#         # 注意力层（需适配双向LSTM的输出维度）
#         lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
#         self.attention = Attention(lstm_output_dim)

#         # 全连接层（需适配双向LSTM的输出维度）
#         # 如果是双向LSTM，隐藏层维度翻倍
#         fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
#         self.fc = nn.Sequential(
#             nn.Linear(fc_input_dim, 128),  # 输入维度适配双向
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, output_size)
#         )

#     def forward(self, x):
#         # LSTM前向传播
#         out, (hidden, cell) = self.lstm(x)  # out.shape: (batch, seq_len, hidden_size * 2)

#         # 取双向LSTM的最后一个时间步的隐藏状态
#         # hidden.shape: (num_layers * 2, batch, hidden_size)（双向时）
#         if self.bidirectional:
#             # 双向时，取前向和后向最后一个时间步的隐藏状态并拼接
#             last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_size * 2)
#         else:
#             # 单向时，直接取最后一层的隐藏状态
#             last_hidden = hidden[-1]  # (batch, hidden_size)

#         # 全连接层处理
#         output = self.fc(last_hidden)  # (batch, output_size)
#         return output

#     #带注意力机制的前向传播
#     def forward(self, x):
#         # LSTM前向传播
#         lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out.shape: (batch, seq_len, hidden_size * 2)

#         # 注意力计算（关键新增部分）
#         context = self.attention(lstm_out)  # context.shape: (batch, hidden_size * 2)

#         # 全连接层处理
#         output = self.fc(context)  # (batch, output_size)
#         return output

## LSTM+MLP
# class lstm(nn.Module):
#     def __init__(self, 
#                  input_size=7, 
#                  hidden_size=64,
#                  num_layers=1,
#                  output_size=7,
#                  dropout=0.1,
#                  bidirectional=True,
#                  batch_first=True,
#                  mlp_hidden_dims=[256,128, 64]):
#         super(lstm, self).__init__()
        
#         # LSTM参数设置
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
#         self.hidden_size = hidden_size
        
#         # LSTM层
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=batch_first,
#             bidirectional=bidirectional,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         # 计算MLP输入维度
#         mlp_input_dim = hidden_size * self.num_directions
        
#         # 构建MLP网络
#         mlp_layers = []
#         prev_dim = mlp_input_dim
#         for dim in mlp_hidden_dims:
#             mlp_layers.extend([
#                 nn.Linear(prev_dim, dim),
#                 nn.BatchNorm1d(dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout)
#             ])
#             prev_dim = dim
        
#         # 最终输出层
#         mlp_layers.append(nn.Linear(prev_dim, output_size))
        
#         # 组合MLP模块
#         self.mlp = nn.Sequential(*mlp_layers)

#     def forward(self, x):
#         # LSTM前向传播
#         # x shape: (batch_size, seq_len, input_size)
#         lstm_out, (h_n, c_n) = self.lstm(x)
        
#         # 获取最终隐藏状态
#         if self.bidirectional:
#             # 拼接最后两个方向的隐藏状态
#             hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, hidden_size*2)
#         else:
#             hidden = h_n[-1]  # (batch_size, hidden_size)
        
#         # MLP前向传播
#         output = self.mlp(hidden)  # (batch_size, output_size)
        
#         return output