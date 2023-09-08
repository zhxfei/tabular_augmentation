"""
   File Name   :   mixup.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/7/17
   Description :
"""
import torch
import numpy as np


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparsity_level=0.0, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    add_noise = 0.0
    mult_noise = 1.0
    x = torch.tensor(x)
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.FloatTensor(x.shape).normal_()
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (
                2 * torch.FloatTensor(x.shape).uniform_() - 1) + 1
    ret = mult_noise * x + add_noise
    return ret.numpy()


def mixup_data(x, y, alpha=1.0, beta=1.0, oversample_num=10000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = x.values, y.values
    sample_num = x.shape[0]
    oversample_cnt = 0
    ret_x, ret_y_a, ret_y_b = x.copy(), y.copy(), y.copy()
    ret_lam = []

    lam = 1
    while oversample_cnt < oversample_num:
        if alpha > 0:
            lam = np.random.beta(alpha, beta)
        index = torch.randperm(sample_num)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        ret_x = np.vstack((ret_x, mixed_x))
        ret_y_a = np.append(ret_y_a, y)
        ret_y_b = np.append(ret_y_b, y[index])
        ret_lam.extend([lam] * sample_num)
        oversample_cnt += sample_num
    ret_x = ret_x[sample_num:]
    ret_y_a = ret_y_a[sample_num:]
    ret_y_b = ret_y_b[sample_num:]
    return ret_x, ret_y_a, ret_y_b, ret_lam


def mixup_data_with_neighbor(x, y, alpha=1.0, beta=1.0, oversample_num=10000, seed=42, knn_neighbors=5):
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = x.values, y.values

    sample_num = x.shape[0]
    oversample_cnt = 0
    ret_x, ret_y_a, ret_y_b = x.copy(), y.copy(), y.copy()
    ret_lam = []
    lam = 1

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    encoders = StandardScaler()
    ret_x = encoders.fit_transform(ret_x)
    nn_k = NearestNeighbors(n_neighbors=knn_neighbors)
    nn_k.fit(ret_x)
    nns = nn_k.kneighbors(ret_x, return_distance=False)[:, 1:]

    while oversample_cnt < oversample_num:
        if alpha > 0:
            lam = np.random.beta(alpha, beta)

        # index = torch.randperm(sample_num)
        selected_indices = np.random.choice(nns.shape[1], size=sample_num, replace=True)
        index = nns[np.arange(sample_num), selected_indices]
        mixed_x = lam * x + (1 - lam) * x[index, :]
        ret_x = np.vstack((ret_x, mixed_x))
        ret_y_a = np.append(ret_y_a, y)
        ret_y_b = np.append(ret_y_b, y[index])
        ret_lam.extend([lam] * sample_num)
        oversample_cnt += sample_num
    ret_x = ret_x[sample_num:]
    ret_y_a = ret_y_a[sample_num:]
    ret_y_b = ret_y_b[sample_num:]
    return ret_x, ret_y_a, ret_y_b, ret_lam


def mixup_nc(x, y, alpha=1.0, beta=1.0, oversample_num=10000, seed=42, category_columns=[]):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cat_indices = [x.columns.get_loc(col) for col in category_columns]
    num_indices = list(set(range(len(x.columns))) - set(cat_indices))
    x, y = x.values, y.values
    sample_num = x.shape[0]
    sample_cnt = 0
    ret_x, ret_y_a, ret_y_b = x.copy(), y.copy(), y.copy()
    ret_lam = []

    while sample_cnt < oversample_num:
        lam = np.random.beta(alpha, beta) if alpha > 0 else 1
        index = torch.randperm(sample_num)
        if lam > 0.5:
            mixed_x = x.copy()
        else:
            mixed_x = x.copy()[index, :]
        mixed_x.dtype = np.float64
        mixed_x[..., num_indices] = lam * x[..., num_indices] + (1 - lam) * x[index, :][..., num_indices]
        # mixed_x[..., cat_indices] = lam * x[..., cat_indices] + (1 - lam) * x[index, :][..., cat_indices]
        # _mixed_x = lam * x + (1 - lam) * x[index, :]
        # diff = mixed_x - _mixed_x
        ret_x = np.vstack((ret_x, mixed_x))
        ret_y_a = np.append(ret_y_a, y)
        ret_y_b = np.append(ret_y_b, y[index])
        ret_lam.extend([lam] * sample_num)
        sample_cnt += sample_num
    ret_x = ret_x[sample_num:]
    ret_y_a = ret_y_a[sample_num:]
    ret_y_b = ret_y_b[sample_num:]
    return ret_x, ret_y_a, ret_y_b, ret_lam


def mixup_augmentation_with_weight(
        x_train, y_train, oversample_num=0, alpha=1.0, beta=1.0, mixup_type='vanilla', seed=42,
        rebalanced_ita=None, noisy_add_level=0.4, noisy_mult_level=0.2, category_columns=[]):
    sample_weight = None
    np.random.seed(seed)
    if oversample_num > x_train.shape[0]:
        print(f'do {mixup_type} mixup....')
        x_train, y_train_a, y_train_b, ret_lam = mixup_data(
            x_train, y_train, oversample_num=oversample_num, alpha=alpha, beta=beta, seed=seed,
            # knn_neighbors=40
        )
        if mixup_type == 'noisy':
            add_noise_level = noisy_add_level
            mult_noise_level = noisy_mult_level
            x_train = _noise(
                x_train, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, seed=seed
            )
        # 重新整理数据
        fea, label, sample_weight = [], [], []
        minority_class = 1
        ratio = (y_train.shape[0] - np.sum(y_train)) / np.sum(y_train)
        if mixup_type == 'vanilla' or mixup_type == 'noisy':
            for index, lam in enumerate(ret_lam):
                if y_train_a[index] == y_train_b[index]:
                    fea.append(x_train[index])
                    label.append(y_train_a[index])
                    sample_weight.append(1)
                else:
                    fea.append(x_train[index])
                    fea.append(x_train[index])
                    label.append(y_train_a[index])
                    label.append(y_train_b[index])
                    sample_weight.append(ret_lam[index])
                    sample_weight.append(1 - ret_lam[index])
        elif mixup_type == 'rebalanced':
            # 当不平衡比例超过2是才会使用，若不平衡比例超过3则尽最大力度平衡
            ita = 1 if ratio > 3 else 0.7
            if rebalanced_ita is not None:
                ita = rebalanced_ita
            k = 2
            for index, lam in enumerate(ret_lam):
                if y_train_a[index] == y_train_b[index]:
                    fea.append(x_train[index])
                    label.append(y_train_a[index])
                    sample_weight.append(1)
                elif 1 - ita < lam < ita and ratio >= k:
                    fea.append(x_train[index])
                    label.append(minority_class)
                    sample_weight.append(1)
                else:
                    fea.append(x_train[index])
                    fea.append(x_train[index])
                    label.append(y_train_a[index])
                    label.append(y_train_b[index])
                    sample_weight.append(ret_lam[index])
                    sample_weight.append(1 - ret_lam[index])
        else:
            raise ValueError('未知的mixup类型')
        x_train, y_train = np.array(fea), np.array(label)
    print(f"positive: {np.sum(y_train)} negative: {len(y_train) - np.sum(y_train)}")
    return x_train, y_train, sample_weight
