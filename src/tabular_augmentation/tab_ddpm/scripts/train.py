from copy import deepcopy
import torch
import os
import numpy as np
import zero
import sys

import pandas as pd

from .. import GaussianMultinomialDiffusion
from .. import lib
from ..scripts.utils_train import get_model, make_dataset, update_ema

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, total_steps=None, device=torch.device('cpu'),
                 # eval_loader=None,
                 # eval_every=5000
                 ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.loop_steps = steps
        self.cur_step = 0
        self.total_steps = total_steps if total_steps is not None else steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.curr_loss_multi = 0.0
        self.curr_loss_gauss = 0.0
        self.curr_count = 0

        # self.train_and_eval = False if eval_loader is None else True
        # self.eval_loader = eval_loader
        # self.eval_every = 5000

    def _anneal_lr(self, step):
        frac_done = step / self.total_steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step_cnt = 0
        while self.cur_step < self.total_steps:
            if step_cnt > self.loop_steps:
                break

            self.diffusion.train()
            x, out_dict = next(self.train_iter)

            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(self.cur_step)

            self.curr_count += len(x)
            self.curr_loss_multi += batch_loss_multi.item() * len(x)
            self.curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (self.cur_step + 1) % self.log_every == 0:
                mloss = np.around(self.curr_loss_multi / self.curr_count, 4)
                gloss = np.around(self.curr_loss_gauss / self.curr_count, 4)
                if (self.cur_step + 1) % self.print_every == 0:
                    print(
                        f'Step {(self.cur_step + 1)}/{self.total_steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}',
                        flush=True)
                self.loss_history.loc[len(self.loss_history)] = [self.cur_step + 1, mloss, gloss, mloss + gloss]
                self.curr_count = 0
                self.curr_loss_gauss = 0.0
                self.curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            self.cur_step += 1
            step_cnt += 1


def train(
        parent_dir,
        real_data_path='data/higgs-small',
        steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=1024,
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        device=torch.device('cpu'),
        seed=0,
        change_val=False
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)

    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
