import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--lambda_param', default=15,type=int)  #第几个窗口
parser.add_argument('--corpusFile', default='data/bar/md15_new.csv')
# 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--s_columns', default="s1,s2,s5", type=str) # 列名
parser.add_argument('--epochs', default=500, type=int) # 训练轮数
parser.add_argument('--layers', default=1, type=int) # LSTM层数,1
parser.add_argument('--input_size', default=3, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=512, type=int) #隐藏层的维度,1024
parser.add_argument('--lr', default=0.00002, type=float) #learning rate 学习率,0.0001
parser.add_argument('--sequence_length', default=2, type=int) # sequence的长度，默认是用前五帧的数据来预测下一帧的特征,5
parser.add_argument('--batch_size', default=64, type=int) #64
parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.36, type=float) #0.2
parser.add_argument('--save_file', default='model/stock_kan.pth') # 模型保存位置
# parser.add_argument('--n_future_steps',default=0,type=int)   #未来预测步数

args = parser.parse_args()

# 解析列名
args.s_columns = args.s_columns.split(",")

##超算中使用
# # 设置CUDA_VISIBLE_DEVICES
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择使用的GPU
# # slurm下检查环境变量中是否有CUDA_VISIBLE_DEVICES
# gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', args.gpu)
# device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and args.useGPU else "cpu")


#自己系统使用
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")

print(device)
args.device = device
