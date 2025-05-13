
import re
import pandas as pd
from parser_my import args
import numpy as np


# 定义文件名
input_md_file = f'data/bar/md{args.lambda_param}.xvg'
input_train_file = f'data/ready_train.csv'  # 原始划分的训练集
input_test_file = f'data/ready_test.csv'  # 原始划分的测试集
output_file = f'data/prediction/md{args.lambda_param}_prediction.xvg'
replacement_file = 'predictions_kan_xvg.xvg' #预测得到的测试集


comment_count = 0  # 统计注释行行数
with open(input_md_file, 'r') as infile:
    for line in infile:
        if line.startswith('#') or line.startswith('@'):  # 统计注释行
            comment_count += 1  # 增加计数器
        else:
                # 非注释行，按空格分隔并统计列数
            lie = line.split()  # 默认按空格分隔
            cols = len(lie)  # 获取当前非注释行的列数
            print(f"原xvg文件一共有多少列：{cols}")
            break  # 只需要第一行的列数，找到后可以退出循环

# 初始化 num 列表
nums = []
# 循环提取数字
for col in args.s_columns:
    # 提取 's' 后面的数字并转换为整数
    num = int(col[1:])  # col[1:] 获取 's' 后面的部分
    nums.append(num)

# 读取训练集和测试集
train_df = pd.read_csv(input_train_file)
test_df = pd.read_csv(input_test_file)

train_length = len(train_df)

no_inplace_row = comment_count + train_length + args.sequence_length
# 只复制前 no_inplace_row 行到 output_file
with open(input_md_file, 'r') as infile, open(output_file, 'w') as outfile:
    for i in range(no_inplace_row):
        line = infile.readline()  # 逐行读取
        if not line:  # 如果到达文件末尾，则停止
            break
        outfile.write(line)  # 写入到输出文件

# 读取 replacement_file 数据
replacement_data = pd.read_csv(replacement_file, header=None)  
print(replacement_data.head())
replace_row = len(replacement_data)  # replacement_file 的行数197996
print(f"替换行数一共为：{replace_row}")

# 替换对应列的数据
# 创建一个新的 DataFrame，包含8列，初始值为0!!!!!!!要改动列数
df = pd.DataFrame(0.0, index=replacement_data.index, columns=range(cols))
# 生成 time 列的数据，间隔为 0.02
time_interval = 0.02
begin_time_pre = 0.02 * (train_length + args.sequence_length)
time_data = np.arange(begin_time_pre, len(df) * time_interval+begin_time_pre, time_interval)
# 将 time 列放在第一列
df.iloc[:, 0] = time_data
 # replacement_data 的行并替换
# df.iloc[:, (nums[0]+1)] = replacement_data.iloc[:,0]  # 替换 num1 列   
# df.iloc[:, (nums[1]+1)] = replacement_data.iloc[:,1]  # 替换 num2 列
# df.iloc[:, (nums[2]+1)] = replacement_data.iloc[:,2]  # 替换 num3 列

input_size = args.input_size  # 获取输入大小
if input_size >= 1:
    df.iloc[:, (nums[0] + 1)] = replacement_data.iloc[:, 0]  # 替换 num1 列
if input_size >= 2:
    df.iloc[:, (nums[1] + 1)] = replacement_data.iloc[:, 1]  # 替换 num2 列
if input_size >= 3:
    df.iloc[:, (nums[2] + 1)] = replacement_data.iloc[:, 2]  # 替换 num3 列
if input_size >= 4:
    df.iloc[:, (nums[3] + 1)] = replacement_data.iloc[:, 3]  # 替换 num4 列
if input_size >= 5:
    df.iloc[:, (nums[4] + 1)] = replacement_data.iloc[:, 4]  # 替换 num4 列
    # 可以继续添加更多的条件，直到你需要的最大输入大小


# 格式化数据
# 将处理好的数据格式化，保留小数点后六位
df = df.apply(lambda x: x.map(lambda y: f"{y:.6f}"))


# 将处理好的数据追加入到 output_file
with open(output_file, 'a') as outfile:  # 以追加模式打开 output_file
    df.to_csv(outfile, header=False, index=False, sep=' ')  # 将 DataFrame 写入文件，使用空格分隔

print(f"已更新 {output_file}，替换列数据来自 {replacement_file}")