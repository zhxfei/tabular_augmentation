"""
   File Name   :   utils.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/7/20
   Description :
"""
# from sklearnex import patch_sklearn
#
# patch_sklearn()
import os
import warnings
import random

import torch
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, \
    recall_score, average_precision_score
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def sample_with_ratio(df, label_column, num_samples, seed=42, positive_ratio=None):
    """从df中按照比例采样n条数据,默认按照原label比例"""
    if positive_ratio is None:
        return train_test_split(df, train_size=num_samples, stratify=df[label_column], random_state=seed)[0]
    nn, pn = df[label_column].value_counts()
    spn = int(num_samples * positive_ratio)
    snn = num_samples - spn
    assert spn < pn and snn < nn, ValueError("采样数 %d,%d 应少于原数据集中的样本数 %d,%d" % (spn, snn, pn, nn))
    pdf_mask = df[label_column] == 1
    pdf = df[pdf_mask]
    ndf = df[~pdf_mask]
    _pdf, _ = train_test_split(pdf, train_size=spn, random_state=seed)
    _ndf, _ = train_test_split(ndf, train_size=snn, random_state=seed)
    new_df = _pdf.merge(_ndf, how='outer')
    return shuffle(new_df, random_state=seed)


def get_category_columns(df, values_type_limit=10):
    fea = [col for col in df.columns if len(df[col].value_counts()) < values_type_limit]
    return fea


def _tabular_model_test(train_x, train_y, test_x, y_test, model_name, model_params=None, sample_weight=None,
                        class_num=10):
    folds = 3
    seed = 42

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    train_matrix = xgb.DMatrix(train_x, label=train_y, weight=sample_weight)
    test_matrix = xgb.DMatrix(test_x, label=y_test)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'aucpr',
              'gamma': 1,
              'min_child_weight': 1.5,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.04,
              'tree_method': 'exact',
              'seed': 2020,
              'nthread': 36,
              }
    # model = xgb.XGBClassifier(**params)
    # model.fit(trn_x, trn_y, eval_metric=['auc'])

    # val_pred = model.predict(val_x, ntree_limit=model.best_ntree_limit)
    model = xgb.train(params, train_matrix,
                      num_boost_round=500, verbose_eval=200,
                      # early_stopping_rounds=20
                      )
    test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)
    test_pred_label = [1 if p >= 0.5 else 0 for p in test_pred]

    # 计算准确率和AUC
    test_accuracy = accuracy_score(y_test, test_pred_label)
    test_precision = precision_score(y_test, test_pred_label)
    test_f1 = f1_score(y_test, test_pred_label)
    test_auc = roc_auc_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred_label)
    test_ap = average_precision_score(y_test, test_pred_label)
    # 计算混淆矩阵
    test_confusion_matrix = confusion_matrix(y_test, test_pred_label)

    # print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test precision: {test_precision}')
    print(f'Train Recall: {test_recall}')
    print(f'Test F1: {test_f1}')
    print(f'Test AP: {test_ap}')
    print(f'Test AUC: {test_auc}')
    print('Train Confusion Matrix:')
    # print(train_confusion_matrix)
    print('Test Confusion Matrix:')
    print(test_confusion_matrix)

    return test_accuracy, test_precision, test_f1, test_auc, test_ap


def tabular_model_test(x_train, y_train, x_test, y_test,
                       model_name='logistic_regression',
                       sample_weight=None, model_params=None, random_seed=42):
    """适用于表格结构化数据二分类任务"""
    # 训练模型
    # sample_weight = sample_weight if sample_weight is not None else None
    if model_name == 'xgb':
        xgb_params = {
            'objective': 'binary:logistic',
            'seed': random_seed
        }
        if model_params is not None:
            xgb_params.update(model_params)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(x_train, y_train, eval_metric=['auc'], sample_weight=sample_weight)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(random_state=random_seed)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=random_seed)
        model.fit(x_train, y_train)
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_seed)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    elif model_name == 'adaboost':
        model = AdaBoostClassifier(random_state=random_seed)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    elif model_name == "knn":
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
    elif model_name == 'mlp':
        model = MLPClassifier(random_state=random_seed, early_stopping=True, hidden_layer_sizes=128)
        model.fit(x_train, y_train)
    elif model_name == 'svm':
        model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        model.fit(x_train, y_train)
    else:
        # default model
        model = LogisticRegression()
        model.fit(x_train, y_train, sample_weight=sample_weight)

    # 预测
    test_pred = model.predict(x_test)

    # 将概率转换为类别
    test_pred_label = [1 if p >= 0.5 else 0 for p in test_pred]

    # 计算准确率和AUC
    test_accuracy = accuracy_score(y_test, test_pred_label)
    test_precision = precision_score(y_test, test_pred_label)
    test_f1 = f1_score(y_test, test_pred_label)
    test_auc = roc_auc_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred_label)
    test_ap = average_precision_score(y_test, test_pred_label)
    # 计算混淆矩阵
    test_confusion_matrix = confusion_matrix(y_test, test_pred_label)

    # print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test precision: {test_precision}')
    print(f'Test F1: {test_f1}')
    print(f'Test AUC: {test_auc}')
    print(f'Test AP: {test_ap}')
    # print(train_confusion_matrix)
    print('Test Confusion Matrix:')
    print(test_confusion_matrix)
    return test_accuracy, test_precision, test_f1, test_auc, test_ap


def image_model_test(x_train, y_train, x_test, y_test, model_name, model_params=None, sample_weight=None, class_num=10):
    """适用于图片多分类任务"""
    if model_name == 'svm':
        # model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        # model.fit(x_train, y_train)
        model = svm.SVC()
        model.fit(x_train, y_train, sample_weight=sample_weight)
    elif model_name == 'xgb':
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': class_num
        }
        if model_params is not None:
            xgb_params.update(model_params)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(x_train, y_train, eval_metric=['auc'], sample_weight=sample_weight)
    else:
        # default model
        model = LogisticRegression()
        model.fit(x_train, y_train)

    # 预测
    test_pred_label = model.predict(x_test)
    # acc = np.sum((test_pred_label == y_test)) / y_test.shape[0]
    test_accuracy = accuracy_score(y_test, test_pred_label)
    test_f1 = f1_score(y_test, test_pred_label, average='macro')
    test_precision = precision_score(y_test, test_pred_label, average='macro')
    # print("Performance Report: \n %s \n" % (classification_report(y_test, test_pred_label)))

    # 计算混淆矩阵
    # test_confusion_matrix = confusion_matrix(y_test, test_pred_label)
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test precision: {test_precision}')
    print(f'Test F1: {test_f1}')
    # print(test_confusion_matrix)
    return test_accuracy, test_precision, test_f1


def train_model_test(x_train, y_train, x_test, y_test, data_type='tabular',
                     model_name='xgb',
                     sample_weight=None,
                     model_params=None):
    if data_type == 'tabular':
        return tabular_model_test(x_train, y_train, x_test, y_test, model_name=model_name, sample_weight=sample_weight,
                                  model_params=model_params)
    elif data_type == 'xgb_tabular':
        return _tabular_model_test(x_train, y_train, x_test, y_test, model_name=model_name,
                                   sample_weight=sample_weight,
                                   model_params=model_params)
    elif data_type == 'image':
        return image_model_test(x_train, y_train, x_test, y_test, model_name, sample_weight=sample_weight,
                                model_params=model_params)
