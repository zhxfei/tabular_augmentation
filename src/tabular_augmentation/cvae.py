"""
   File Name   :   cvae.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/9/5
   Description :
"""
import numpy as np
import torch
from torch.nn import Linear
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.tvae import Encoder, Decoder


class CustomTensorDataset(TensorDataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __getitem__(self, index):
        return self.tensors[0][index], self.tensors[1][index]

    def __len__(self):
        return self.tensors[0].size(0)


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class ConditionalTVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
            self,
            embedding_dim=128,
            compress_dims=(128, 128),
            decompress_dims=(128, 128),
            l2scale=1e-5,
            num_classes=2,
            batch_size=500,
            epochs=300,
            loss_factor=2,
            cuda=True,
            verbose=False
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.num_classes = num_classes

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._verbose = verbose

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        y = train_data['label']
        train_data.drop(['label'], axis=1, inplace=True)

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        self.dataset = CustomTensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device),
                                           torch.from_numpy(y.to_numpy().astype('float32')).to(self._device))
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        embed_dim = int(data_dim / 2)
        self.embed_class = Linear(self.num_classes, embed_dim)
        self.encoder = Encoder(data_dim + embed_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim + embed_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.embed_class.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(self.loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)

                y = data[1].to(self._device)
                y = self.convert_one_hot(y)
                y = y.view(-1, self.num_classes)
                emb_y = self.embed_class(y)

                emb_m = torch.cat([real, emb_y], dim=1)
                mu, std, logvar = self.encoder(emb_m)

                eps = torch.randn_like(std)
                emb = eps * std + mu
                z = torch.cat([emb, emb_y], dim=1)
                rec, sigmas = self.decoder(z)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if self._verbose:
                    print(f'Epoch {i + 1}, Loss 1: {loss_1.detach().cpu(): .4f},'  # noqa: T001
                          f'Loss 2: {loss_2.detach().cpu(): .4f}',
                          flush=True)

    def convert_one_hot(self, input_tensor):
        # 确定目标张量的形状（50x2）
        batch_size = input_tensor.shape[0]
        target_shape = (batch_size, self.num_classes)

        # 创建全零的目标张量
        one_hot_embedding = torch.zeros(target_shape)

        # 使用scatter_函数将对应位置设置为1
        one_hot_embedding.scatter_(1, input_tensor.view(-1, 1).long(), 1)

        # print(one_hot_embedding)
        return one_hot_embedding

    def fix_sample(self, samples):
        self.encoder.eval()
        self.decoder.eval()
        # source_x = self.transformer.transform(source_x)

        steps = samples // self.batch_size + 1
        data = []
        all_index = list(range(0, len(self.dataset)))
        for _ in range(steps):
            indexes = np.random.choice(all_index, self.batch_size)
            data_ = self.dataset[indexes]

            real = data_[0].to(self._device)
            # real = data_[indexes]
            mean, std, _ = self.encoder(real)
            # std = torch.zeros_like(std)

            # print(mean, std)
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    @random_state
    def sample(self, samples, **kwargs):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            y = torch.Tensor([kwargs['label'], ] * self.batch_size)
            y = self.convert_one_hot(y)
            y = y.view(-1, self.num_classes)
            emb_y = self.embed_class(y)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            z = torch.cat([noise, emb_y], dim=1)

            fake, sigmas = self.decoder(z)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)


class FixedTVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
            self,
            embedding_dim=128,
            compress_dims=(128, 128),
            decompress_dims=(128, 128),
            l2scale=1e-5,
            batch_size=500,
            epochs=300,
            loss_factor=2,
            cuda=True,
            verbose=False
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._verbose = verbose

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        self.dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(self.loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if self._verbose:
                    print(f'Epoch {i + 1}, Loss 1: {loss_1.detach().cpu(): .4f},'  # noqa: T001
                          f'Loss 2: {loss_2.detach().cpu(): .4f}',
                          flush=True)

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)


class DeltaTVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
            self,
            embedding_dim=128,
            compress_dims=(128, 128),
            decompress_dims=(128, 128),
            l2scale=1e-5,
            num_classes=2,
            batch_size=500,
            epochs=300,
            loss_factor=2,
            cuda=True,
            verbose=False
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.num_classes = num_classes

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._verbose = verbose

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        self.dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions

        self.encoder = Encoder(data_dim * 2, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim + data_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(self.loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                index = torch.randperm(real.shape[0])
                real_b = real[index, :]

                mu, std, logvar = self.encoder(torch.cat([real, real_b], dim=1))

                eps = torch.randn_like(std)
                emb = eps * std + mu

                rec, sigmas = self.decoder(torch.cat([emb, real_b], dim=1))
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if self._verbose:
                    print(f'Epoch {i + 1}, Loss 1: {loss_1.detach().cpu(): .4f},'  # noqa: T001
                          f'Loss 2: {loss_2.detach().cpu(): .4f}',
                          flush=True)

    @random_state
    def sample(self, samples):
        self.encoder.eval()
        self.decoder.eval()
        # source_x = self.transformer.transform(source_x)

        steps = samples // self.batch_size + 1
        data = []
        all_index = list(range(0, len(self.dataset)))
        for _ in range(steps):
            indexes = np.random.choice(all_index, self.batch_size)
            data_ = self.dataset[indexes]

            real = data_[0].to(self._device)
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)

            z = torch.cat([noise, real], dim=1)
            # real = data_[indexes]
            fake, sigmas = self.decoder(z)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)


class DiffTVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
            self,
            embedding_dim=128,
            compress_dims=(128, 128),
            decompress_dims=(128, 128),
            l2scale=1e-5,
            num_classes=2,
            batch_size=500,
            epochs=300,
            loss_factor=2,
            cuda=True,
            verbose=False
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.num_classes = num_classes

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._verbose = verbose

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        self.dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        self.forward_linear = Linear(data_dim * 2, data_dim)
        self.reverse_linear = Linear(data_dim * 2, data_dim)
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.forward_linear.parameters()) + list(self.reverse_linear.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(self.loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                index = torch.randperm(real.shape[0])
                real_b = real[index, :]

                real_diff = self.forward_linear(torch.cat([real, real_b], dim=1))
                mu, std, logvar = self.encoder(real_diff)

                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)

                rec = self.reverse_linear(torch.cat([rec, real_b], dim=1))
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if self._verbose:
                    print(f'Epoch {i + 1}, Loss 1: {loss_1.detach().cpu(): .4f},'  # noqa: T001
                          f'Loss 2: {loss_2.detach().cpu(): .4f}',
                          flush=True)

    @random_state
    def sample(self, samples):
        self.encoder.eval()
        self.decoder.eval()
        # source_x = self.transformer.transform(source_x)

        steps = samples // self.batch_size + 1
        data = []
        all_index = list(range(0, len(self.dataset)))
        for _ in range(steps):
            indexes = np.random.choice(all_index, self.batch_size)
            data_ = self.dataset[indexes]

            real = data_[0].to(self._device)
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)

            # real = data_[indexes]
            fake, sigmas = self.decoder(noise)
            fake = self.reverse_linear(torch.cat([fake, real], dim=1))

            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
