#将原xx.xvg文件变成xx_new.csv文件
#python filtered_multiples_of_frames.py 15 200000
import pandas as pd
import re 
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='处理数据文件')
parser.add_argument('id', type=int, help='第几个lambda')
parser.add_argument('time', type=int, help='文件中模拟多少帧')


# 解析命令行参数
args = parser.parse_args()

# 根据窗口值定义输入和输出文件名
# input_file = f'data/md{args.id}.xvg'
# output_file = f'data/md{args.id}_new.xvg'
# csv_output_file = f'data/md{args.id}_new.csv'

input_file = f'data/bar/md{args.id}.xvg'
output_file = f'data/bar/md{args.id}_new.xvg'
csv_output_file = f'data/bar/md{args.id}_new.csv'

# 提取以@和#开头的行
header_lines = []
column_names = ['time']
s_label = []
with open(input_file, 'r') as infile:
    for line in infile:
        if line.startswith('@'):
            header_lines.append(line)
            if re.match(r'@ s\d+ legend',line):
            # 提取列名
                s_label = line.split()[1]  # 跳过第一个字符串
                column_names.append(s_label)
        elif line.startswith('#'):
            header_lines.append(line)
        else:
            # 找到第一行非注释行，停止读取
            break

# 将注释行写入输出文件
with open(output_file, 'w') as outfile:
    outfile.writelines(header_lines)


# 读取非注释行数据
data = pd.read_csv(input_file, sep='\s+', header=None, skiprows=lambda x: x < len(header_lines))
print(data)

# # 间隔提取帧数
# 生成需要的间隔值0.2 0.4 0.6 ...4000
desired_values = [round(i * 0.02, 6) for i in range(0, int(args.time + 1))]    #时间间隔值记得修改
#desired_values = [round(i * 0.02, 6) for i in range(int(0 / 0.02), int(args.time / 0.02) + 1)]   

# 过滤出所需的值
filtered_data = data[data[0].isin(desired_values)]
print(filtered_data.head())
# # 提取前xx帧的数据
# filtered_data = data.iloc[:]


# 将过滤后的数据写入 CSV 文件，并添加表头
filtered_data.to_csv(csv_output_file, header=column_names, index=False)

# 将过滤后的数据写入输出文件
with open(output_file, 'a') as outfile:  # 以追加模式打开文件
    filtered_data.to_csv(outfile, index=False, header=False, sep=' ')

print(f"提取完成，结果已保存到 {output_file} 和 {csv_output_file}")
