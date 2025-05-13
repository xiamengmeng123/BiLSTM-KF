import pandas as pd
import os
from io import StringIO
import argparse
"""
    用于将md$i.xvg文件中的对应参数自动读取并传入模型的变量

"""

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='处理 LAMBDA_PARAM 值')
parser.add_argument('lambda_param', type=int, help='LAMBDA_PARAM 的值')
args = parser.parse_args()

# 构建文件名
md_file = f"data/bar/md{args.lambda_param}.xvg"
corpus_file = f"data/bar/md{args.lambda_param}_new.csv"    
# 检查文件是否存在
if not os.path.isfile(md_file):
    print(f"文件 {md_file} 不存在")
else:
    # 读取文件，跳过注释行
    with open(md_file, 'r') as f:
        lines = [line for line in f if not line.startswith(('#', '@'))]

    # 将过滤后的行转换为 DataFrame
    df = pd.read_csv(StringIO(''.join(lines)), sep='\s+', header=None)

    # 获取非零列的索引
    non_zero_columns = df.columns[(df != 0).any()].tolist()

    # 生成 S_COLUMNS，排除第一列（索引为 0 的列）
    S_COLUMNS = [f"s{index-1}" for index in non_zero_columns if index != 0]
    
    # 输出结果
    print(f"LAMBDA_PARAM={args.lambda_param}")
    print(f"CORPUS_FILE={corpus_file}")
    print(f"S_COLUMNS={','.join(S_COLUMNS)}")
    print(f"INPUT_SIZE={len(S_COLUMNS)}")

    # 替换 sub.sh 文件中的内容
    sub_file = f'sub{args.lambda_param}.sh'
    with open(sub_file, 'r') as file:
        script_lines = file.readlines()

    # 更新对应的行
    for i, line in enumerate(script_lines):
        if line.startswith("LAMBDA_PARAM="):
            script_lines[i] = f"LAMBDA_PARAM={args.lambda_param}\n"
        elif line.startswith("CORPUS_FILE="):
            script_lines[i] = f"CORPUS_FILE={corpus_file}\n"
        elif line.startswith("S_COLUMNS="):
            script_lines[i] = f"S_COLUMNS={','.join(S_COLUMNS)}\n"
        elif line.startswith("INPUT_SIZE="):
            script_lines[i] = f"INPUT_SIZE={len(S_COLUMNS)}\n"
    # 将修改后的内容写回 sub.sh 文件
    with open(sub_file, 'w') as file:
        file.writelines(script_lines)

    print(f"已更新 sub.sh 文件，LAMBDA_PARAM={args.lambda_param} 的相关内容。")
