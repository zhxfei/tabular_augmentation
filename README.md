# Description

`tabular_augmentation` contains some classical and novel methods used for data augmentation, making tabular data
augmentation easier, especially for few-shot learning case.

# Usage

SMOTE-based methods

```python
from tabular_augmentation import smote_augmentation
method = 'SVMSMOTE'
x_synthesis, y_synthesis = smote_augmentation(x_few_train, y_few_train, method, seed=seed,
                                              oversample_num=100, positive_ratio=None,
                                              knn_neighbors=3)
tabular_model_test(x_synthesis, y_synthesis, x_test, y_test, model_name='xgb')
```

Mixup-base methods
```python
from tabular_augmentation import mixup_augmentation_with_weight
method = 'vanilla'
x_synthesis, y_synthesis, sample_weight = mixup_augmentation_with_weight(
            x_few_train, y_few_train, oversample_num=200, alpha=1, beta=1, mixup_type=method, seed=seed, rebalanced_ita=1)
tabular_model_test(x_synthesis, y_synthesis, x_test, y_test, model_name='xgb', sample_weight=sample_weight)
```

CTGAN/TVAE-based methods

Methods(CTGAN/TVAE/DeltaTVAE/DiffTVAE) use `sdv_synthesis` function to generate synthetic data, and ConditionalTVAE use `sdv_synthesis_cvae` function
```python
from tabular_augmentation import sdv_synthesis, sdv_synthesis_cvae
method = 'CTGAN'

x_synthesis, y_synthesis = sdv_synthesis(
            x_few_train, y_few_train, method, oversample_num=5000,
            seed=seed, init_synthesizer=True, positive_ratio=0.5,
        )
tabular_model_test(x_synthesis, y_synthesis, x_test, y_test, model_name='xgb')

```

TabDDPM-based methods
```python
from tabular_augmentation import ddpm_synthesis

method = "DDPM"

x_synthesis, y_synthesis = ddpm_synthesis(
            x_few_train, y_few_train, method, oversample_num=5000, seed=seed, init_synthesizer=True, positive_ratio=None, train_steps=10000)
tabular_model_test(x_synthesis, y_synthesis, x_test, y_test, model_name='xgb')

```
# Example
For details, please refer to  [example.ipynb](https://github.com/zhxfei/tabular_augmentation/blob/master/example.ipynb)

# Cite
#### SMOTE
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)

#### MIXUP
[ICLR' 18]mixup: BEYOND EMPIRICAL RISK MINIMIZATION [Mixup](https://github.com/facebookresearch/mixup-cifar10)

[ICLR' 22]Noisy Feature Mixup
[NoisyMixup](https://github.com/erichson/NFM)

[ECCV' 20]Remix: Rebalanced Mixup

#### CTGAN/TVAE
[NIPS' 19]Modeling Tabular data using Conditional GAN
[CTGAN](https://github.com/sdv-dev/CTGAN)

#### TabDDPM
[ICML' 23] TabDDPM: Modelling Tabular Data with Diffusion Models
[TabDDPM](https://github.com/yandex-research/tab-ddpm)
