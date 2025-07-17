import os
from collections import OrderedDict
from typing import Iterable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from utils import SoftClusterAssignment
from matplotlib import pyplot as plt


class galaxy10Dataset(Dataset):
    def __init__(self,yInput,yOutput):
        self.yInput = yInput
        self.yOutput = yOutput

    def __len__(self):
        return len(self.yInput)

    def __getitem__(self, idx):
        return self.yInput[idx], self.yOutput[idx], idx


def cluster_acc(y_true, y_pred):
    """
    This is code from original repository.
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)

    accuracy = 0
    for idx in range(len(ind[0]) - 1):
        i = ind[0][idx]
        j = ind[1][idx]
        accuracy += w[i, j]
    accuracy = accuracy * 1.0 / y_pred.size
    return accuracy


class SAE(pl.LightningModule):
    """
    Stacked AutoEncoder for pretraining the initial parameter and cluster centroid.

    :param dimensions: [input_dim, hidden_dim_1, ..., hidden_dim_N]
    :param activation: Non-linear activation function both in encoder and decoder
    :param final_activation: Non-linear activation function in final layer
    :param dropout: Dropout rate in each layer

    :param batch_size: Size of minibatch
    :param lr: Learning rate
    :param lr_decay: Learning rate decay ratio
    :param lr_decay_step: Learning rate decay frequency
    :param weight_decay: Weight decay ratio
    """

    def __init__(
        self,
        dimensions: Iterable[int],
        activation: Optional[nn.Module] = nn.ReLU(),
        final_activation: Optional[nn.Module] = None,
        dropout: Optional[float] = 0.0,
        batch_size: int = 64,
        lr: float = 0.1,
        lr_decay: float = 0.1,
        lr_decay_step: int = 20000,
        weight_decay: float = 0.0,
    ):
        super(SAE, self).__init__()
        self.criterion = nn.MSELoss()

        # stack encoder layers
        encoder_layers = self._add_linear_layer_stack(
            dimensions[:-1], activation, dropout
        )
        encoder_layers.extend(
            self._add_linear_layer_stack(
                [dimensions[-2], dimensions[-1]], final_activation, dropout=None,
            )
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # stack decoder layers
        decoder_layers = self._add_linear_layer_stack(
            list(reversed(dimensions[1:])), activation, dropout
        )
        decoder_layers.extend(
            self._add_linear_layer_stack(
                [dimensions[1], dimensions[0]], final_activation, dropout=None,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # initialize parameter
        self.encoder.apply(self._init_weight)
        self.decoder.apply(self._init_weight)

        self.save_hyperparameters()
        self.batch_losses = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)

    def prepare_data(self) -> None:
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        # train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        # test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
	
        # Load images and labels from the Galaxy 10 file 
        with h5py.File('/scratch1/10386/lsmith9003/share/tutorialData/galaxy10/Galaxy10_DECals.h5','r') as File:
            labels = np.array(File['ans'])
            indSub = np.where((labels==2) | (labels==8))
            yInput = np.array(File['images'])[indSub]

        # Downsample inputs (if necessary)
        yInput = yInput[:,96:160,96:160,:]
        yInput = yInput[0::]/255	

        # Define test/train split
        yInput_train, yInput_test = train_test_split(yInput,train_size=0.90,random_state=42)  
        yInput_train = torch.tensor(yInput_train,dtype=torch.float32)
        yInput_test = torch.tensor(yInput_test,dtype=torch.float32)
        yOutput_train = yInput_train
        yOutput_test = yInput_test

        train_dset = galaxy10Dataset(yInput_train,yOutput_train)
        test_dset = galaxy10Dataset(yInput_test,yOutput_test)

        # Hyperparameter Tuning by cross-validation on a validation set is not an option in unsupervised clustering.
        # So train and test set are combined.
        self.dset = ConcatDataset([train_dset, test_dset])

    def _add_linear_layer_stack(
        self,
        dims: Iterable[int],
        activation: Optional[nn.Module],
        dropout: Optional[float],
    ) -> List[nn.Module]:
        def single_unit(in_dim: int, out_dim: int) -> List[nn.Module]:
            unit = [nn.Linear(in_dim, out_dim)]
            if activation is not None:
                unit.append(activation)
            if dropout is not None:
                unit.append(nn.Dropout(0.2))
            return nn.Sequential(*unit)

        return [single_unit(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]

    def _init_weight(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # follow paper setting
            nn.init.constant_(layer.bias, 0)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        scheduler = StepLR(
            optimizer, self.hparams.lr_decay_step, gamma=self.hparams.lr_decay
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> dict:
        data, _, __ = batch
        flatten = data.reshape(self.hparams.batch_size, -1)
        reconstruction = self(flatten)
        loss = self.criterion(reconstruction, flatten)
        self.batch_losses.append(loss)
        return {"loss": loss}

    def on_train_epoch_end(self):
        loss = torch.stack(self.batch_losses).mean()
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_losses.clear()
        print("")


class DEC(pl.LightningModule):
    """
    Deep Embedded Clustering

    :param encoder: Finetuned Encoder of Stacked AutoEncoder
    :param num_cluster: Number of cluster
    :param hidden_dim: Dimension of final encoder vector
    :param alpha: Freedom value of t-distribution
    :param batch_size: Batch size
    :param lr_dec: Learning rate
    :param tol: Threshold to stop training
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_cluster: int = 2,
        hidden_dim: int = 10,
        alpha: float = 1.0,
        batch_size: int = 64,
        lr_dec: float = 0.01,
        tol: float = 1e-3,
    ):
        super(DEC, self).__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.assignment = SoftClusterAssignment(num_cluster, hidden_dim, alpha)
        self.criterion = F.kl_div
        self.kmeans = KMeans(self.hparams.num_cluster, n_init=20)
        self.init = True

        self.batch_losses = []

    def forward(self, batch):
        return self.assignment(self.encoder(batch))

    def prepare_data(self) -> None:
        #transform = transforms.Compose([transforms.ToTensor(),])

        # Hyperparameter Tuning by cross-validation on a validation set is not an option in unsupervised clustering.
        # So train and test set are combined.
        # train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        # test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)

        # Load images and labels from the Galaxy 10 file 
        with h5py.File('/scratch1/10386/lsmith9003/share/tutorialData/galaxy10/Galaxy10_DECals.h5','r') as File:
            labels = np.array(File['ans'])
            indSub = np.where((labels==2) | (labels==8))
            yInput = np.array(File['images'])[indSub]
            labels = labels[indSub]

        # Downsample inputs (if necessary)
        yInput = yInput[:,96:160,96:160,:]
        yInput = yInput[0::]/255     

        # Define test/train split
        yOutput = labels
        yOutput[yOutput==2] = 0
        yOutput[yOutput==8] = 1
        yInput_train, yInput_test, yOutput_train, yOutput_test = train_test_split(yInput,yOutput,train_size=0.90,random_state=42)  
        yInput_train = torch.tensor(yInput_train,dtype=torch.float32)
        yInput_test = torch.tensor(yInput_test,dtype=torch.float32)
        yOutput_train = torch.tensor(yOutput_train,dtype=torch.float32)
        yOutput_test = torch.tensor(yOutput_test,dtype=torch.float32)

        train_dset = galaxy10Dataset(yInput_train,yOutput_train)
        test_dset = galaxy10Dataset(yInput_test,yOutput_test)

        self.dset = ConcatDataset([train_dset, test_dset])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    #def val_dataloader(self) -> DataLoader:
    #    return DataLoader(
    #        self.dset, batch_size=self.hparams.batch_size, shuffle=False, drop_last=True
    #    )

    def configure_optimizers(self) -> torch.optim:
        return optim.SGD(self.parameters(), lr=self.hparams.lr_dec, momentum=0.9)

    def training_step(self, batch, batch_idx) -> dict:
        if self.init:
            init_info = self._initialize_centroid()
            self.assignment = SoftClusterAssignment(
                self.hparams.num_cluster,
                self.hparams.hidden_dim,
                self.hparams.alpha,
                init_info["centroid"],
            )
            # print(f"Initial acc: {init_info['accuracy']}")
            self.init = False

        data, target, _ = batch
        q = self(data.reshape(self.hparams.batch_size, -1))
        p = self._get_target_distribution(q).detach()

        loss = self.criterion(q.log(), p)
        self.batch_losses.append(loss)
        return {"loss": loss}

    def on_train_epoch_end(self):
        loss = torch.stack(self.batch_losses).mean()
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_losses.clear()
        print("")


    #def validation_step(self, batch, batch_idx) -> dict:
    #    data, target = batch
    #    embedded = self(data.reshape(self.hparams.batch_size, -1))
    #    pred = torch.cat([embedded]).max(1)[1]
    #    accuracy = cluster_acc(target.cpu().numpy(), pred.cpu().numpy())
    #    log = {"accuracy": accuracy}
    #    return {"accuracy": accuracy, "log": log}

    def on_train_end(self):
        train_data = self.train_dataloader()
        q_all = []
        x_all = []
        idx_all = []
        with torch.no_grad():
            for batch, idx in train_data:
                x, y = batch
                q = self.encoder(x.reshape(self.hparams.batch_size, -1))
                q_all.append(q)
                x_all.append(x)
                idx.append(idx)
        x_all = np.array(x_all)        
        q_all = np.array(q_all)
        x_shape = np.shape(x_all)
        q_shape = np.shape(q_all)
        q_all = np.reshape(q_all,shape=(q_shape[0]*q_shape[1],q_shape[2]))
        x_all = np.reshape(x_all,shape=(x_shape[0]*x_shape[1],x_shape[2],x_shape[3],x_shape[4]))

        savemat('latent_space_dec.mat',{'yLat': q_all})
        savemat('idx_dec.mat',{'idx': idx})
        


        #pca = PCA(n_components=2)
        #yLatPlot = pca.fit_transform(q_all)

        #kmeans_end = KMeans(self.hparams.num_cluster, n_init=20).fit(yLatPlot)
        #klabels = kmeans_end.labels_
        #kcen = kmeans_end.cluster_centers_

        # Extract the coordinates corresponding to each of the three clusters
        #kind0 = np.where(klabels==0)
        #kind1 = np.where(klabels==1)
        #kind0 = kind0[0]
        #kind1 = kind1[0]

        #kindPlot0 = np.argmin((yLatPlot[kind0,0] - kcen[0,0])**2 + (yLatPlot[kind0,1] - kcen[0,1])**2)
        #kindPlot1 = np.argmin((yLatPlot[kind1,0] - kcen[1,0])**2 + (yLatPlot[kind1,1] - kcen[1,1])**2)

        #fig8 = plt.figure(figsize=(4,3.5))
        #ax8 = fig8.add_subplot()
        #ax8.scatter(yLatPlot[kind0,0],yLatPlot[kind0,1],marker='o',color='blue',alpha=0.15)
        #ax8.scatter(yLatPlot[kind1,0],yLatPlot[kind1,1],marker='o',color='red',alpha=0.15)
        #ax8.set_xlabel('latent coord 1', fontsize=10)
        #ax8.set_ylabel('latent coord 2', fontsize=10)
        #plt.tight_layout()
        #plt.savefig("latentSpace.png")

        # Figure 9: plot a representative image from each cluster
        #fig9 = plt.figure()
        #plt.subplot(1,2,1)
        #yEval0 = x_all[kind0[kindPlot0]]
        #yEval1 = x_all[kind1[kindPlot1]]

        #plt.imshow(yEval0)
        #plt.title('Cluster 0')
        #plt.subplot(1,2,2)
        #plt.imshow(yEval1)
        #plt.title('Cluster 1')
        #plt.savefig("cluster_images.png")



        #print(np.shape(q_all))	


    def _initialize_centroid(self) -> dict:
        print("Set Initial Centroid...")
        dloader = DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )
        label, feature = [], []

        for batch in dloader:
            data, target, _ = batch
            data, target = data.to(self.device), target.to(self.device)
            label.append(target)
            feature.append(
                self.encoder(data.reshape(self.hparams.batch_size, -1)).detach().cpu()
            )
        label = torch.cat(label)
        pred = self.kmeans.fit_predict(torch.cat(feature).numpy())
        accuracy = cluster_acc(label.cpu().numpy(), pred)

        return {
            "accuracy": accuracy,
            "centroid": torch.tensor(
                self.kmeans.cluster_centers_, requires_grad=True
            ).cuda(),
        }

    def _get_target_distribution(self, q):
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 1)).t()
        return p
