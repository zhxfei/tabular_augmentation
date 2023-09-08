"""
   File Name   :   smote.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/7/17
   Description :
"""
import sys

import numpy as np
from imblearn.over_sampling import (SMOTE, SVMSMOTE, ADASYN, SMOTENC)
from imblearn.combine import SMOTETomek, SMOTEENN

from .custom_smote import (NormalSMOTE, ScoreSMOTE, CleanSMOTE, ReinforceSMOTE)


def smote_augmentation(x_train, y_train, generator_type,
                       oversample_num=10000, seed=42, positive_ratio=None,
                       knn_neighbors=3, categorical_features=[]):
    """
    二分类smote增强
    """
    positive_ratio = np.sum(y_train) / y_train.size if positive_ratio is None else positive_ratio
    positive_num = int(oversample_num * positive_ratio)
    method = getattr(sys.modules[__name__], generator_type)
    kwargs = {
        'random_state': seed,
        'sampling_strategy': {
            1: positive_num,
            0: oversample_num - positive_num
        }
    }
    if "SMOTENC" in generator_type:
        kwargs['categorical_features'] = categorical_features
    if "SMOTETomek" == generator_type or "SMOTEENN" == generator_type:
        pass
    elif "SMOTE" in generator_type:
        kwargs['k_neighbors'] = knn_neighbors
    else:
        kwargs['n_neighbors'] = knn_neighbors
    sv = method(**kwargs)

    try:
        x_train_smote, y_train_smote = sv.fit_resample(x_train, y_train)
    except (RuntimeError, ValueError) as e:
        raise e

    return x_train_smote, y_train_smote


def multiclass_smote_augmentation(x_train, y_train, seed,
                                  oversample_num=10000,
                                  method_name="SMOTE",
                                  class_num=10,
                                  knn_neighbors=3):
    """多分类smote增强,每个类的样本数量应该为相等"""
    strategy = {cls: int(oversample_num / class_num) for cls in range(class_num)}
    method = getattr(sys.modules[__name__], method_name)
    if "SMOTE" in method_name:
        sv = method(random_state=seed, sampling_strategy=strategy, k_neighbors=knn_neighbors)
    else:
        sv = method(random_state=seed, sampling_strategy=strategy, n_neighbors=knn_neighbors)
    x_train_smote, y_train_smote = sv.fit_resample(x_train, y_train)
    return x_train_smote, y_train_smote


