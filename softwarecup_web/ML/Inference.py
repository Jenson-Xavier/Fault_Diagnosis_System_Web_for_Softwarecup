from ML.project_model import *
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import json
import copy
import warnings
warnings.filterwarnings("ignore")


# 自定义得分函数，对模型效果的评价指标
def macro_f1_compute(y_true, y_pred):
    macro_P = precision_score(y_true, y_pred, average='macro')
    macro_R = recall_score(y_true, y_pred, average='macro')
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
    return macro_F1


# 将预测的分类结果的各类别数量作为列表返回
def res_analysis(inp):
    inp = np.array(inp)
    inp = list(inp)
    counts = []
    proportion = []
    sum = np.shape(inp)[0]
    for i in range(6):
        counts.append(inp.count(i))
        proportion.append(counts[i] / sum)
    proportion = [round(x, 3) for x in proportion]
    return counts, proportion


# 上传测试csv文件，采用已训练好的模型进行结果的预测，在本地生成json文件
def inference(path_name):
    data_raw = pd.read_csv(path_name)
    data_clean = copy.deepcopy(data_raw)
    # 删除无关列
    features = data_clean.drop(['sample_id'], axis=1)

    # KNN填充缺失值
    imputer = KNNImputer(n_neighbors=3)

    # 将存在缺失值的列提取出来
    missing_features = features.loc[:, features.isnull().sum() > 0]
    for f in missing_features:
        features[['feature22', f]] = imputer.fit_transform(
            features[['feature22', f]])

    # 删除方差为0列，通过训练集已经得知feature57, feature77, feature100是方差为0列
    features = features.drop(
        ['feature57', 'feature77', 'feature100'], axis=1)

    # 检测待预测的数据中是否存在label列，如果存在就删除
    labels = None
    if "label" in features:
        labels = pd.DataFrame(features["label"])
        features = features.drop(['label'], axis=1)
        print("检测到数据中存在标签列。")

    # 特征缩放，标准化处理
    for col in features.columns:
        features[col] = (features[col] - features[col].mean()
                         ) / features[col].std()
    print(features.shape)

    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN1()
    model.load_state_dict(torch.load(
        "/myweb/softwarecup_web/ML_models/best_cnn_model.pt",
        map_location=device))

    # 必须转换为验证阶段
    model.eval()
    model.to(device)
    inp = np.array(features)
    inp = torch.tensor(inp)
    inp = inp.to(device)
    output = model(inp)
    if output.device.type == 'cuda':
        output = output.to('cpu')
    output = output.argmax(axis=1)

    # 如果存在标签列，就分析预测结果的好坏，记录下每个类别的精确率、召回率和macro_f1得分
    recall_res = None
    precision_res = None
    macro_f1_res = None
    correct_counts = None
    macro_f1_score = None

    if labels is not None:
        labels = np.array(labels).reshape((-1,))
        labels = torch.tensor(labels)
        precision_res, recall_res, macro_f1_res, correct_counts = \
            precision_recall_fscore_support(
                labels, output, labels=[0, 1, 2, 3, 4, 5])
        precision_res = list(precision_res)
        recall_res = list(recall_res)
        macro_f1_res = list(macro_f1_res)
        macro_f1_score = macro_f1_compute(labels, output)

        precision_res = [round(x, 3) for x in precision_res]
        recall_res = [round(x, 3) for x in recall_res]
        macro_f1_res = [round(x, 3) for x in recall_res]
        macro_f1_score = round(macro_f1_score*100, 2)

    # 获取预测的每个类别的数量和比例
    pred_counts, pred_proportion = res_analysis(output)

    # 生成json文件
    json_text = {}
    for i in range(inp.shape[0]):
        json_text[str(i)] = int(output[i])
    json_data = json.dumps(json_text)

    f = open("/myweb/softwarecup_web/ML_pred_results/submit.json", 'w')
    f.write(json_data)
    f.close()
    return json_text, pred_counts, pred_proportion, labels, macro_f1_score, \
        precision_res, recall_res, macro_f1_res


json_text, pred_counts, pred_proportion, labels,\
    macro_f1_score, precision_res, recall_res, macro_f1_res = inference(
        '/myweb/softwarecup_web/ML/validate_1000.csv')
print('finish')
print(pred_counts)
print(pred_proportion)
if labels is not None:
    print(f"上传数据中存在标签列，macro_f1得分为{macro_f1_score} (这个分数信息渲染到页面上)")
    print(precision_res)
    print(recall_res)
    print(macro_f1_res)
else:
    print(f"上传数据中不存在标签列")
