"""
   File Name   :   custom_smote.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/8/16
   Description :
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import _safe_indexing
from scipy import sparse
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.base import ArraysTransformer, check_classification_targets, check_sampling_strategy, label_binarize
from sklearn.metrics import roc_auc_score


def data_cleaning(x, y, random_seed=42, tol=0.2):
    x, y = x.copy(), y.copy()
    columns = list(x.columns)
    keep = columns + ['label']

    import xgboost as xgb
    xgb_params = {
        'objective': 'binary:logistic',
        'seed': random_seed
    }
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(x, y, eval_metric='auc')

    x['pred'] = clf.predict_proba(x)[:, 1]
    x['label'] = y
    x['pred_label_diff'] = np.abs(x['pred'] - x['label'])

    operate_sample = x[x.pred_label_diff < tol][keep]
    not_operate_sample = x[x.pred_label_diff >= tol][keep]

    return operate_sample[columns], operate_sample['label'], not_operate_sample[columns], not_operate_sample['label']


class NormalSMOTE(SMOTE):
    def __init__(self, *args, **kwargs):
        super(NormalSMOTE, self).__init__(*args, **kwargs)

    def fit_resample(self, X, y):
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        output = self._fit_resample(X, y)
        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        # X_, y_ = output[0], y_
        # X__, y__ = arrays_transformer.transform(output[0], y_)

        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            # 获取相同类的X
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            # 在相同类上训练KNN
            from sklearn.preprocessing import StandardScaler
            mm_train = StandardScaler()
            X_class_normalized = mm_train.fit_transform(X_class)

            self.nn_k_.fit(X_class_normalized)
            # 对每个元素获取相同类的下标
            nns = self.nn_k_.kneighbors(X_class_normalized, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled


class CleanSMOTE(SMOTE):
    def __init__(self, *args, **kwargs):
        super(CleanSMOTE, self).__init__(*args, **kwargs)
        self.tol = 0.2

    def fit_resample(self, X, y, seed=42):
        opx, opy, not_opx, not_opy = data_cleaning(X, y, random_seed=seed, tol=self.tol)
        X, y = opx, opy
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        output = self._fit_resample(X, y)
        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)

        X_ = pd.concat([X_, not_opx])
        y_ = np.append(y_, not_opy)

        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            # 获取相同类的X
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            # 在相同类上训练KNN
            from sklearn.preprocessing import StandardScaler
            mm_train = StandardScaler()
            X_class_normalized = mm_train.fit_transform(X_class)

            self.nn_k_.fit(X_class_normalized)
            # 对每个元素获取相同类的下标
            nns = self.nn_k_.kneighbors(X_class_normalized, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled


class ScoreSMOTE(SMOTE):
    def __init__(self, *args, **kwargs):
        super(ScoreSMOTE, self).__init__(*args, **kwargs)
        self.clf = xgb.XGBClassifier()

    def fit_resample(self, X, y):
        check_classification_targets(y)

        self.clf.fit(X, y)
        feature_importance = self.clf.feature_importances_
        indices = np.argsort(feature_importance)[:: -1][:10]

        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        output = self._fit_resample(X, y, indices)
        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _fit_resample(self, X, y, indices):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            # 获取相同类的X
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            # 在相同类上训练KNN
            from sklearn.preprocessing import StandardScaler
            mm_train = StandardScaler()
            X_class_normalized = mm_train.fit_transform(X_class)
            X_class_partial = X_class_normalized[..., indices]
            from sklearn.preprocessing import StandardScaler
            mm_train = StandardScaler()
            X_class_partial = mm_train.fit_transform(X_class_partial)

            self.nn_k_.fit(X_class_partial)
            # 对每个元素获取相同类的下标
            nns = self.nn_k_.kneighbors(X_class_partial, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled


class ReinforceSMOTE(SMOTE):
    def __init__(self, *args, **kwargs):
        super(ReinforceSMOTE, self).__init__(*args, **kwargs)
        self.clf = xgb.XGBClassifier()

    def fit_resample(self, X, y):
        self.clf.fit(X, y)
        best_score = 0
        rx, ry = None, None
        for i in range(10):
            self.random_state = i
            oversample_x, oversample_y = self.fit_resample_(X, y)
            pred = self.clf.predict(oversample_x)
            score = roc_auc_score(oversample_y, pred)
            if score > best_score:
                rx = oversample_x
                ry = oversample_y
                best_score = score
        return rx, ry

    def fit_resample_(self, X, y):
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        output = self._fit_resample(X, y)
        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])


# class MixSMOTENC(SMOTENC):
#     def __init__(self, *args, **kwargs):
#         super(MixSMOTENC, self).__init__(*args, **kwargs)
#
#     def _fit_resample(self, X, y):
#         # FIXME: to be removed in 0.12
#         if self.n_jobs is not None:
#             warnings.warn(
#                 "The parameter `n_jobs` has been deprecated in 0.10 and will be "
#                 "removed in 0.12. You can pass an nearest neighbors estimator where "
#                 "`n_jobs` is already set instead.",
#                 FutureWarning,
#             )
#
#         self.n_features_ = _num_features(X)
#         self._validate_column_types(X)
#         self._validate_estimator()
#
#         # compute the median of the standard deviation of the minority class
#         target_stats = Counter(y)
#         class_minority = min(target_stats, key=target_stats.get)
#
#         X_continuous = _safe_indexing(X, self.continuous_features_, axis=1)
#         X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
#         X_minority = _safe_indexing(X_continuous, np.flatnonzero(y == class_minority))
#
#         if sparse.issparse(X):
#             if X.format == "csr":
#                 _, var = csr_mean_variance_axis0(X_minority)
#             else:
#                 _, var = csc_mean_variance_axis0(X_minority)
#         else:
#             var = X_minority.var(axis=0)
#         self.median_std_ = np.median(np.sqrt(var))
#
#         X_categorical = _safe_indexing(X, self.categorical_features_, axis=1)
#         if X_continuous.dtype.name != "object":
#             dtype_ohe = X_continuous.dtype
#         else:
#             dtype_ohe = np.float64
#
#         if self.categorical_encoder is None:
#             self.categorical_encoder_ = OneHotEncoder(
#                 handle_unknown="ignore", dtype=dtype_ohe
#             )
#         else:
#             self.categorical_encoder_ = clone(self.categorical_encoder)
#
#         # the input of the OneHotEncoder needs to be dense
#         X_ohe = self.categorical_encoder_.fit_transform(
#             X_categorical.toarray() if sparse.issparse(X_categorical) else X_categorical
#         )
#         if not sparse.issparse(X_ohe):
#             X_ohe = sparse.csr_matrix(X_ohe, dtype=dtype_ohe)
#
#         # we can replace the 1 entries of the categorical features with the
#         # median of the standard deviation. It will ensure that whenever
#         # distance is computed between 2 samples, the difference will be equal
#         # to the median of the standard deviation as in the original paper.
#
#         # In the edge case where the median of the std is equal to 0, the 1s
#         # entries will be also nullified. In this case, we store the original
#         # categorical encoding which will be later used for inversing the OHE
#         if math.isclose(self.median_std_, 0):
#             self._X_categorical_minority_encoded = _safe_indexing(
#                 X_ohe.toarray(), np.flatnonzero(y == class_minority)
#             )
#
#         X_ohe.data = np.ones_like(X_ohe.data, dtype=X_ohe.dtype) * self.median_std_ / 2
#         X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr")
#
#         X_resampled, y_resampled = super()._fit_resample(X_encoded, y)
#
#         # reverse the encoding of the categorical features
#         X_res_cat = X_resampled[:, self.continuous_features_.size:]
#         X_res_cat.data = np.ones_like(X_res_cat.data)
#         X_res_cat_dec = self.categorical_encoder_.inverse_transform(X_res_cat)
#
#         if sparse.issparse(X):
#             X_resampled = sparse.hstack(
#                 (
#                     X_resampled[:, : self.continuous_features_.size],
#                     X_res_cat_dec,
#                 ),
#                 format="csr",
#             )
#         else:
#             X_resampled = np.hstack(
#                 (
#                     X_resampled[:, : self.continuous_features_.size].toarray(),
#                     X_res_cat_dec,
#                 )
#             )
#
#         indices_reordered = np.argsort(
#             np.hstack((self.continuous_features_, self.categorical_features_))
#         )
#         if sparse.issparse(X_resampled):
#             # the matrix is supposed to be in the CSR format after the stacking
#             col_indices = X_resampled.indices.copy()
#             for idx, col_idx in enumerate(indices_reordered):
#                 mask = X_resampled.indices == col_idx
#                 col_indices[mask] = idx
#             X_resampled.indices = col_indices
#         else:
#             X_resampled = X_resampled[:, indices_reordered]
#
#         return X_resampled, y_resampled

