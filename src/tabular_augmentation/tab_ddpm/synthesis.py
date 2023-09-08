"""
   File Name   :   ddpm_train.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/8/21
   Description :
"""
import os
import time
import logging

import zero
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from . import lib
from . import GaussianMultinomialDiffusion
from .lib.data import get_category_sizes
from .scripts.train import Trainer
from .scripts.utils_train import get_model


def set_logger(log_name):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # logging.basicConfig(filename=f'{log_name}.log', level=logging.INFO, format=log_format)
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger()


class FinData(Dataset):
    def __init__(self, x, y, category_columns):
        df, label = x.copy(), y.copy()

        self.origin_columns = df.columns
        self.category_columns = category_columns
        self.cat_encoder = {col: LabelEncoder() for col in self.category_columns}
        # self.num_encoder = {col: Normalizer() for col in }
        for col in self.category_columns:
            df[col] = self.cat_encoder[col].fit_transform(df[col])
        self.numerical_encoder = StandardScaler()
        df.loc[:, ~df.columns.isin(self.category_columns)] = self.numerical_encoder.fit_transform(
            df.loc[:, ~df.columns.isin(self.category_columns)])

        self.x_category = torch.tensor(np.array(df[self.category_columns]))
        self.x_category_columns = df[self.category_columns].columns
        self.x_numerical = torch.tensor(np.array(df.loc[:, ~df.columns.isin(self.category_columns)]))
        self.x_numerical_columns = df.loc[:, ~df.columns.isin(self.category_columns)].columns
        self.x = torch.tensor(
            np.concatenate([df.loc[:, ~df.columns.isin(self.category_columns)],
                            np.array(df[self.category_columns])], axis=1),
            dtype=torch.float32,
            requires_grad=True)
        self.y = torch.tensor(np.array(label), dtype=torch.float32, requires_grad=True)

    def get_category_sizes(self):
        size = get_category_sizes(self.x_category)
        return size

    def inverse_transform(self, x_gen):
        ret = pd.DataFrame(x_gen.copy(), columns=list(self.x_numerical_columns) + list(self.x_category_columns))
        ret = ret[self.origin_columns]
        ret[self.x_category_columns] = ret[self.x_category_columns].astype(int)
        for col in self.category_columns:
            ret[col] = self.cat_encoder[col].inverse_transform(ret[col])
        ret.loc[:, ~ret.columns.isin(self.category_columns)] = self.numerical_encoder.inverse_transform(
            ret.loc[:, ~ret.columns.isin(self.category_columns)])
        return ret

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TabDDPM:
    def __init__(self, train_dataset,
                 num_timesteps=1000,
                 model_params=None, model_type='mlp', gaussian_loss_type='mse',
                 scheduler='cosine',
                 seed=0, model_save=True, model_name="tab_ddpm"):
        self.seed = seed
        self.model_save = model_save
        self.model_name = model_name
        self.dir_name = os.getcwd() + '/results'
        self.train_dataset = train_dataset

        self.model_params = model_params
        self.model_type = model_type
        self.diffusion_num_timesteps = num_timesteps
        self.diffusion_loss_type = gaussian_loss_type
        self.diffusion_scheduler = scheduler

        self.model = None
        self.diffusion = None
        self.trainer = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._init_model()

    def _init_model(self):
        if self.model_params is None:
            self.model_params = {
                'num_classes': 2,
                'is_y_cond': True,
                'rtdl_params': {
                    'd_layers': [
                        512,
                        # 1024,
                        # 1024,
                        # 1024,
                        # 1024,
                        512,
                    ],
                    'dropout': 0.0
                }
            }
        if 'd_in' not in self.model_params:
            self.num_classes = np.array(self.train_dataset.get_category_sizes())
            if len(self.num_classes) == 0:
                self.num_classes = np.array([0])
            self.num_numerical_features = self.train_dataset.x_numerical.shape[1] \
                if self.train_dataset.x_numerical is not None else 0
            d_in = np.sum(self.num_classes) + self.num_numerical_features
            self.model_params['d_in'] = d_in

        logging.info(self.model_params)
        self.model = get_model(
            self.model_type,
            self.model_params,
            self.num_numerical_features,
            category_sizes=self.train_dataset.get_category_sizes()
        )
        self.model.to(self.device)

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.diffusion_loss_type,
            num_timesteps=self.diffusion_num_timesteps,
            scheduler=self.diffusion_scheduler,
            device=self.device
        )
        self.diffusion.to(self.device)

    def train(
            self,
            train_dataset=None,
            # eval_dataset=None,
            steps=5000,
            total_steps=None,
            lr=0.002,
            weight_decay=1e-4,
            batch_size=1024,
            model_save=None
    ):
        zero.improve_reproducibility(self.seed)
        train_loader = lib.prepare_fast_dataloader(train_dataset if train_dataset is not None else self.train_dataset,
                                                   split='train', batch_size=batch_size)
        # eval_loader = lib.prepare_fast_dataloader(eval_dataset, split='train',
        #                                           batch_size=batch_size) if eval_dataset is not None else None
        self.trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=lr,
            weight_decay=weight_decay,
            steps=steps,
            total_steps=total_steps,
            device=self.device
        )

        self.train_()
        model_save = model_save if model_save is not None else self.model_save
        if model_save:
            self.save_model()

    def train_(self):
        assert self.trainer is not None
        self.model.train()
        self.diffusion.train()
        self.trainer.run_loop()

    def save_model(self):
        self.trainer.loss_history.to_csv(os.path.join(self.dir_name, f'{self.model_name}_loss.csv'), index=False)
        torch.save(self.diffusion._denoise_fn.state_dict(), os.path.join(self.dir_name, f'{self.model_name}.pt'))
        torch.save(self.trainer.ema_model.state_dict(), os.path.join(self.dir_name, f'{self.model_name}_ema.pt'))

    def load_model(self, model_path):
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.diffusion_loss_type,
            num_timesteps=self.diffusion_num_timesteps,
            scheduler=self.diffusion_scheduler,
            device=self.device
        )
        self.diffusion.to(self.device)

    def sample(self, num_samples,
               batch_size=200,
               positive_ratio=None,
               ):
        zero.improve_reproducibility(self.seed)

        self.model.eval()
        self.diffusion.eval()

        not_balance = None
        _, empirical_class_dist = torch.unique(self.train_dataset.y, return_counts=True)
        if not_balance == 'fix':
            empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
            x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

        elif not_balance == 'fill':
            ix_major = empirical_class_dist.argmax().item()
            val_major = empirical_class_dist[ix_major].item()
            x_gen, y_gen = [], []
            for i in range(empirical_class_dist.shape[0]):
                if i == ix_major:
                    continue
                distrib = torch.zeros_like(empirical_class_dist)
                distrib[i] = 1
                num_samples = val_major - empirical_class_dist[i].item()
                x_temp, y_temp = self.diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
                x_gen.append(x_temp)
                y_gen.append(y_temp)

            x_gen = torch.cat(x_gen, dim=0)
            y_gen = torch.cat(y_gen, dim=0)

        else:
            x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)
        #
        # num_numerical_features = self.dataset.x_numerical.shape[1] if self.dataset.x_numerical is not None else 0
        x_gen, y_gen = x_gen.numpy(), y_gen.numpy()
        x_gen = self.train_dataset.inverse_transform(x_gen)
        return x_gen, y_gen


# def ddpm_synthesis(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
#                    init_synthesizer=False):
#     ratio = positive_ratio if positive_ratio is not None else np.sum(y_train) / y_train.shape[0]
#
#     fraud_df = x_train[np.where(y_train > 0.5, True, False)]
#     non_fraud_df = x_train[~np.where(y_train > 0.5, True, False)]
#
#     # 使用ctgan合成，针对每一个类都进行一次训练
#     fraud_sample_num = int(oversample_num * ratio)
#     no_fraud_sample_num = oversample_num - fraud_sample_num
#
#     fraud_df = _ddpm_synthesis(
#         fraud_df, generator_type, 'positive', oversample_num=fraud_sample_num, seed=seed,
#         init_synthesizer=init_synthesizer)
#     non_fraud_df = _ddpm_synthesis(
#         non_fraud_df, generator_type, 'negative', oversample_num=no_fraud_sample_num, seed=seed,
#         init_synthesizer=init_synthesizer)
#
#     fraud_df['label'] = 1
#     non_fraud_df['label'] = 0
#
#     combined_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
#     combined_df = combined_df.drop_duplicates(keep='first')
#     shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
#     features = [f for f in shuffled_df.columns if f not in ['label']]
#     x_train, y_train = shuffled_df[features], shuffled_df['label']
#     return x_train, y_train


def ddpm_synthesis(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
                   train_steps=10000,
                   init_synthesizer=False):
    # ratio = positive_ratio if positive_ratio is not None else np.sum(y_train) / y_train.shape[0]
    # features = [f for f in x_train.columns if f not in ['label']]

    res = _ddpm_synthesis(x_train, generator_type, 'positive', oversample_num=oversample_num, seed=seed, y=y_train,
                          init_synthesizer=init_synthesizer, train_steps=train_steps)
    return res


def _ddpm_synthesis(df, generator_type, synthesis_type, oversample_num=0, seed=42, generator_params=None, y=None,
                    init_synthesizer=False, train_steps=10000):
    assert synthesis_type in ['positive', 'negative']
    assert generator_type == 'DDPM'
    if y is None:
        if synthesis_type == 'positive':
            y = np.array([1] * df.shape[0])
        else:
            y = np.array([0] * df.shape[0])

    train_dataset = FinData(df, y, category_columns=[])
    ddpm = TabDDPM(train_dataset, seed=seed, model_save=False, model_name="model_name")
    ddpm.train(steps=train_steps)
    ret = ddpm.sample(oversample_num)
    if y is None:
        return ret[0]
    return ret


def ddpm_train():
    from utils import get_category_columns_by_dataset, get_category_columns
    from utils import get_credit_card_data, get_auto_data, get_prp_auto_data, get_fraud_data, get_fin_data

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--positive_ratio', default=None)
    parser.add_argument('--dataset', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()

    model_name = f"{args.dataset}_{args.positive_ratio}_{args.num_samples}_{int(time.time())}"
    set_logger(model_name)

    # import sys
    # f = open(f'{model_name}_{int(time.time())}.output', 'a')
    # sys.stdout = f
    print(args)

    from sklearn.metrics import roc_auc_score
    seed = args.seed
    dateset_func_map = {
        'auto': get_auto_data,
        'prp': get_prp_auto_data,
        'credit_card': get_credit_card_data,
        'fraud': get_fraud_data,
        'finance': get_fin_data
    }
    data = dateset_func_map[args.dataset](
        get_dummies=True, positive_ratio=None, seed=seed, num_samples=args.num_samples
    )
    x, y, x_test, y_test = data
    import xgboost as xgb
    model = xgb.XGBClassifier()

    model.fit(x, y)
    y_hat = model.predict(x_test)
    test_pred = y_hat
    test_pred_label = [1 if p >= 0.5 else 0 for p in y_hat]
    f1, auc = print_eval_result(y_test, test_pred, test_pred_label)

    auc = roc_auc_score(y_test, y_hat)
    logging.info(auc)
    logging.info(model_name)

    train_dataset = FinData(x, y, category_columns=[])
    eval_dataset = FinData(x_test, y_test, category_columns=[])
    tab_ddpm = TabDDPM(train_dataset, seed=seed, model_save=False,
                       model_name=model_name)

    eval_per_steps = args.step / 20
    res_map = {
        'ddpm_f1': [f1, ],
        'ddpm_auc': [auc, ]
    }
    for i in range(20):
        if i == 0:
            tab_ddpm.train(steps=eval_per_steps, model_save=False, total_steps=args.step)
        else:
            tab_ddpm.train_()
        # tab_ddpm.load_model(tab_ddpm.dir_name + '/auto_None_50_1692843751.pt')
        for num in [50, ]:
            with torch.no_grad():
                x_gen, y_gen = tab_ddpm.sample(num_samples=num, batch_size=1000)
            model.fit(x_gen, y_gen)
            y_hat = model.predict(x_test)
            auc = roc_auc_score(y_test, y_hat)
            logging.info(f'after: {auc}')
            test_pred = y_hat
            test_pred_label = [1 if p >= 0.5 else 0 for p in y_hat]
            f1, auc = print_eval_result(y_test, test_pred, test_pred_label)
            res_map['ddpm_f1'].append(f1)
            res_map['ddpm_auc'].append(auc)
    print(res_map)
    tab_ddpm.save_model()
    # from utils import generate_line_chart
    # generate_line_chart(res_map)


def print_eval_result(y_test, test_pred, test_pred_label):
    # 计算准确率和AUC
    from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, \
        average_precision_score, confusion_matrix
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
    return test_f1, test_auc


if __name__ == '__main__':
    ddpm_train()
