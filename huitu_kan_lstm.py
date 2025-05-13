from LSTMModel import lstm # 导入自定义的LSTM模型类
from dataset import getData  # 导入数据处理函数
from parser_my import args # 导入命令行参数解析模块
import matplotlib.pyplot as plt
import numpy as np

# 获取训练和测试数据
norm_params, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)

# 打印归一化参数，确保它们的值是正确的
print(f"norm_params 是: {norm_params}")

# 读取文件内容并去除方括号
def read_predictions_file(file_path):
    preds, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            # print(f"line 是:{line}")
            pred, label = line.strip().split('],[')
            # 去除方括号并转换为浮点数
            pred = list(map(float, pred.strip('[]').split(',')))  # 处理为列表
            label = list(map(float, label.strip('[]').split(',')))  # 处理为列表
            preds.append(pred)
            labels.append(label)
    return preds, labels

# 反标准化
# def reverse_standardize(values, energy_max, energy_min):
#     # return [value * (energy_max - energy_min) + energy_min for value in values]
#     return [
#         value * (energy_max[feature] - energy_min[feature]) + energy_min[feature]
#         # for feature, value in zip(['coul', 'vdw'], values)
#         for feature, value in zip(args.s_columns, values)
#     ]

def reverse_standardize(values, norm_params, feature_names):
    """
    反归一化函数
    :param values: 单条数据的归一化值（列表，如 [s1_val, s2_val, ...]）
    :param norm_params: 归一化参数字典
    :param feature_names: 特征名称列表（如 ['s1', 's2', ...]）
    :return: 反归一化后的值（列表）
    """
    reversed_values = []
    for val, feature in zip(values, feature_names):
        params = norm_params[feature]
        if params['type'] == 'robust':
            # RobustScaler 反归一化
            reversed = val * params['iqr'] + params['median']
        elif params['type'] == 'minmax':
            # Min-Max 反归一化
            reversed = val * (params['max'] - params['min']) + params['min']
        reversed_values.append(reversed)
    return reversed_values

#绘制粗糙曲线
# # 绘制预测值和真实值
# def plot_predictions(preds, labels,feature_names):
#     num_features = len(feature_names)
#     plt.figure(figsize=(14, 7))  # 设置图像大小
#     # plt.plot(preds, label='Predicted ')
#     # plt.plot(labels, label='Actual ')
#     # plt.xlabel('Time Step')  # 横轴标签
#     # plt.ylabel('Stock Price')  # 纵轴标签
#     # plt.legend()
#     # plt.title('Predicted vs Actual Stock ')  # 图像标题
#     # plt.savefig("kan_lstm_predict")
#     # plt.show()

#     # 只取最后100个数据点
#     preds = preds[-1000:]
#     labels = labels[-1000:]

#     for i in range(num_features):
#         plt.subplot(num_features, 1, i + 1)  # 创建子图
#         plt.plot([pred[i] for pred in preds], label='Predicted')
#         plt.plot([label[i] for label in labels], label='Actual')
#         plt.xlabel('Time Step')  # 横轴标签
#         plt.ylabel(feature_names[i])  # 纵轴标签
#         plt.legend()
#         plt.title(f'Predicted vs Actual {feature_names[i]}')  # 图像标题

#     plt.tight_layout()  # 自动调整子图参数
#     plt.savefig("figure/kan_lstm_predict.png")  # 保存图像
#     plt.show()


#绘制平滑曲线
def moving_average(data, window_size):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_predictions(preds, labels, feature_names, window_size=5):
    num_features = len(feature_names)
    plt.figure(figsize=(14, 7))  # 设置图像大小

    # 只取最后1000个数据点
    preds = preds[-1000:]
    labels = labels[-1000:]

    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)  # 创建子图
        
        # 对预测值和真实值进行平滑处理
        smoothed_preds = moving_average([pred[i] for pred in preds], window_size)
        smoothed_labels = moving_average([label[i] for label in labels], window_size)

        # 绘制平滑后的曲线
        plt.plot(smoothed_preds, label='Predicted')
        plt.plot(smoothed_labels, label='Actual')
        plt.xlabel('Time Step')  # 横轴标签
        plt.ylabel(feature_names[i])  # 纵轴标签
        plt.legend()
        plt.title(f'Predicted vs Actual {feature_names[i]}')  # 图像标题

    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(f"figure/lstm_md{args.lambda_param}.png")  # 保存图像
    plt.show()

# 主函数
def main():

    file_path = 'txt/predictions_kan_lstm.txt'  # 文件路径

    preds, labels = read_predictions_file(file_path)
    # 假设你的数据有2个特征，这里可以根据实际特征名称进行修改
    feature_names = args.s_columns   
    # preds = reverse_standardize(preds, energy_max, energy_min)
    # labels = reverse_standardize(labels, energy_max, energy_min)
    preds = [reverse_standardize(pred, norm_params,feature_names) for pred in preds]
    labels = [reverse_standardize(label, norm_params,feature_names) for label in labels]
    plot_predictions(preds, labels,feature_names)

    #将结果写入xvg文件中
    with open('predictions_kan_xvg.xvg', 'w') as file:
        for pred in preds:
            # 格式化每个浮点数为小数点后5位            
            pred_str = ', '.join(f"{float(value):.5f}" for value in pred)  # 将每个浮点数转换为字符串并格式化
            file.write(f"{pred_str}\n")

# 运行主函数
main()
