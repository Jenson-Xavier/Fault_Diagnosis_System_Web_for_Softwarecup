from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from ML.project_model import *
from sklearn.impute import KNNImputer
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


class MyDataset(Dataset):
    def __init__(self, feature_tensor, label_tensor):
        self.feature_tensor = feature_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.feature_tensor.size(0)

    def __getitem__(self, index):
        return self.feature_tensor[index], self.label_tensor[index]


# 上传训练csv文件，在本地生成target_model.pt文件
def train(path_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_raw = pd.read_csv(path_name)
    data_clean = copy.deepcopy(data_raw)
    data_clean = data_clean.drop(['sample_id'], axis=1)
    features = data_clean.drop(['label'], axis=1)
    labels = pd.DataFrame(data_clean['label'])
    # KNN填充缺失值
    imputer = KNNImputer(n_neighbors=3)
    # 将存在缺失值的列提取出来
    missing_features = features.loc[:, features.isnull().sum() > 0]
    for f in missing_features:
        features[['feature22', f]] = imputer.fit_transform(
            features[['feature22', f]])
    # 删除方差为0列，通过训练集已经得知feature57, feature77, feature100是方差为0列
    features = features.drop(['feature57', 'feature77', 'feature100'], axis=1)
    # 特征缩放，标准化处理
    for col in features.columns:
        features[col] = (features[col] - features[col].mean()
                         ) / features[col].std()
    print(features.shape)
    labels = np.array(labels).reshape((-1,))
    labels = torch.tensor(labels)
    labels = one_hot(labels)
    features = np.array(features)
    features = torch.tensor(features)
    features = features.to(device)

    train_dataset = MyDataset(features, labels)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True)

    # 加载模型
    model = CNN1()
    # 用预训练的参数初始化模型
    model.load_state_dict(torch.load(
        "/myweb/softwarecup_web/ML_models/best_cnn_model.pt",
        map_location=device))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    learning_rate = 1e-3
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=1e-8, amsgrad=False)
    # 设置训练轮数
    epoch = 20

    best_model = None
    best_score = 0.0

    for i in range(epoch):
        print(f"-------------第{i + 1}轮训练开始--------------")
        model.train()
        for data in train_dataloader:
            x_true, y_true = data
            x_true = x_true.to(device)
            y_true = y_true.to(device)
            y_pred = model(x_true)
            loss = loss_fn(y_pred.to(torch.float32), y_true.to(torch.float32))
            # 优化器优化模型
            optim.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新参数
            optim.step()

        # 到此，完成一轮的训练
        # 评价训练的效果，获取训练集的评分
        train_pred = model(features)
        if train_pred.device.type == 'cuda':
            train_pred = train_pred.to('cpu')
        train_pred = train_pred.argmax(axis=1)
        train_true = labels.argmax(axis=1)
        train_score = macro_f1_compute(train_true, train_pred)
        if train_score > best_score:
            best_score = train_score
            best_model = copy.deepcopy(model)
        print(f'第{i + 1}轮训练过后，训练集得分:{round(train_score * 100, 2)}')

    torch.save(best_model.state_dict(),
               "/myweb/softwarecup_web/ML_models/target_model.pt")


# train('/myweb/softwarecup_web/ML/train_10000.csv')
