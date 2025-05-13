
from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import matplotlib.pyplot as plt


#设置目标损失阈值
target_val_loss = 0.01
#自定义损失函数，针对诸如数值分布存在异常高峰或低峰的预测，如s1
class WeightedLoss(nn.Module):
    def __init__(self, s1_index, weight=5.0):
        super().__init__()
        self.s1_idx = s1_index  # 假设s1是输出中的第0个特征
        self.weight = weight
    
    def forward(self, pred, target):
        loss = (pred - target)**2
        # 对s1的损失加权
        loss[:, self.s1_idx] *= self.weight  
        return loss.mean()

def validate(model, test_loader, criterion):
    """验证集评估"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.squeeze(1).to(args.device)
            label = label.to(args.device)
            pred = model(data)
            total_loss += criterion(pred, label).item()
    return total_loss / len(test_loader)
    
def train(optim_params=None):
    """支持参数动态注入的训练函数
    Args:
        optim_params (dict): 贝叶斯优化生成的参数字典，格式示例：
            {
                'hidden_size': 512, 
                'layers': 2,
                'dropout': 0.3,
                'lr': 0.001,
                'batch_size': 128,
                'sequence_length': 20
            }
    """
    if optim_params:
        print("\n[DEBUG] 正在应用优化参数:", optim_params)
        # 类型安全转换
        args.hidden_size = int(optim_params.get('hidden_size', args.hidden_size))
        args.layers = int(optim_params.get('layers', args.layers))
        args.dropout = float(optim_params.get('dropout', args.dropout))
        args.lr = float(optim_params.get('lr', args.lr))
        args.batch_size = int(optim_params.get('batch_size', args.batch_size))
        args.sequence_length = int(optim_params.get('sequence_length', args.sequence_length))

    # 动态数据加载（必须放在参数覆盖之后）
    norm_params, train_loader, test_loader = getData(
        args.corpusFile, 
        args.sequence_length,  # 使用当前参数
        args.batch_size
    )
    # 创建lstm模型实例，并将其移动到指定的设备（CPU或GPU）(使用动态参数)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=args.input_size, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    # criterion = nn.MSELoss()  # 定义损失函数
    # criterion = WeightedLoss(s1_index=0,weight=0.5)  # 根据s1的位置调整
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001
    
    # 记录指标
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 10  # 连续10次验证损失未改善则停止
    

    # 打开一个名为 'lstm_loss_log.txt' 的文本文件，用于记录每个epoch的损失值
    with open('txt/lstm_opt_loss_log.txt', 'a+') as f:
        for epoch in range(args.epochs):  # 循环遍历每个epoch
            #训练阶段
            model.train()
            train_loss = 0  # 初始化当前epoch的总损失
            for data, label in train_loader:
                # 维度安全检查
                if data.dim() == 3 and data.size(1) == 1:
                    data = data.squeeze(1)
                data = data.to(args.device)
                label = label.to(args.device)
                
                pred = model(data)
                loss = criterion(pred, label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            # 验证阶段
            val_loss = validate(model, test_loader, criterion)

            # 记录日志
            log_msg = f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}'
            print(log_msg)
            f.write(log_msg + '\n')
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # 保存最佳模型
                torch.save({'state_dict': model.state_dict()}, args.save_file)
                print(f'Epoch {epoch+1}: 发现最佳模型，验证损失 {val_loss:.4f}')
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'早停触发，连续{patience}次未改善')
                    break
            
            # 目标损失检查
            if val_loss < target_val_loss:
                print(f'达到目标验证损失 {target_val_loss}，提前终止')
                break
    # 最终保存
    torch.save({'state_dict': model.state_dict()}, args.save_file)
    return {'val_loss': best_val_loss}

if __name__ == "__main__":
    train()


