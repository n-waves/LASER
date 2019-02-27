

import argparse
import copy
import shutil
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


################################################

def LoadData(bdir, dfn, lfn, dim=1024, bsize=32, shuffle=False, quiet=False):
    x = np.fromfile(bdir + dfn, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)

    lbl = np.loadtxt(bdir + lfn, dtype=np.int32)
    # lbl_str = np.loadtxt(bdir + lfn, dtype=np.int32)
    # labels, lbl = np.unique(lbl_str, return_inverse=True)
    # lbl.reshape(lbl.shape[0], 1)
    if not quiet:
        print(' - read {:d}x{:d} elements in {:s}'.format(x.shape[0], x.shape[1], dfn))
        print(' - read {:d} labels [{:d},{:d}] in {:s}'
              .format(lbl.shape[0], lbl.min(), lbl.max(), lfn))

    D = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(lbl))
    loader = data_utils.DataLoader(D, batch_size=bsize, shuffle=shuffle)
    return loader, lbl

################################################

class Net(nn.Module):
    def __init__(self, idim=1024, odim=2, nhid=None,
                 dropout=0.0, gpu=0, activation='TANH'):
        super(Net, self).__init__()
        self.gpu = gpu
        modules = []

        modules = []
        print(' - mlp {:d}'.format(idim), end='')
        if len(nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for nh in nhid:
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    if activation == 'TANH':
                        modules.append(nn.Tanh())
                        print('-{:d}t'.format(nh), end='')
                    elif activation == 'RELU':
                        modules.append(nn.ReLU())
                        print('-{:d}r'.format(nh), end='')
                    else:
                       raise Exception('Unrecognized activation {activation}')
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, odim))
            print('-{:d}, dropout={:.1f}'.format(odim, dropout))
        else:
            modules.append(nn.Linear(idim, odim))
            print(' - mlp %d-%d'.format(idim, odim))
        self.mlp = nn.Sequential(*modules)
        # Softmax is included CrossEntropyLoss !

        if self.gpu >= 0:
            self.mlp = self.mlp.cuda()

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dset, name=' Dev', nlbl=4):
        correct = 0
        total = 0
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)
        for data in dset:
            X, Y = data
            Y = Y.long()
            if self.gpu >= 0:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).int().sum()
            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()

        print(' | {:4s}: {:5.2f}%'
                         .format(name, 100.0 * correct.float() / total), end='')
        print(' | classes:', end='')
        for i in range(nlbl):
            print(' {:5.2f}'.format(100.0 * corr[i] / total), end='')

        return correct, total



"""
Accuracy matrix:
Train     en     de     es     fr     it     ru     zh
 en:    90.88  86.48  67.62  61.98  69.95  22.95  11.65
 de:    73.23  92.90  77.23  74.05  72.30  24.80   9.93
 es:    65.62  80.58  92.03  73.28  69.03  34.10  12.58
 fr:    78.35  85.45  78.20  89.68  69.85  33.88   9.68
 it:    73.93  84.58  79.23  76.73  84.03  34.48  11.83
 ru:    57.33  63.78  45.80  52.78  51.15  66.08  36.28
 zh:    26.15  28.13  21.88  29.33  30.58  34.38  75.62
"""

def eval_net(m, dset, nlbl=4):
    correct = 0
    total = 0
    m.mlp.train(mode=False)
    corr = np.zeros(nlbl, dtype=np.int32)
    results = []
    lbls = []
    for data in dset:
        X, Y = data
        Y = Y.long()
        if m.gpu >= 0:
            X = X.cuda()
            Y = Y.cuda()
        outputs = m.mlp(X)
        _, predicted = torch.max(outputs.data, 1)
        lbls.append(np.array(Y.cpu()))
        results.append(np.array(predicted.cpu()))

    preds = np.concatenate(results)
    true_lbsl = np.concatenate(lbls)

    print("\nAccuracy", (preds == true_lbsl).sum() / len(preds))

    return preds


def main(model_save, dir, suffix="", dim=1024, dataset="mldoc" ):
    dir = Path(dir)
    if dir.name.endswith("-1"):
        size = 1000
    elif dir.name.endswith("-10"):
        size = 10000
    elif dir.name.endswith("-books"):
        size = 1800
    else:
        print("Unsupported size accepts dir names with suffix -1 or -10 ", dir)

    base_dir = f"embed{size}/"

    laser_model_lang = model_save.split(".")[1].split("-")[0]
    dst = dir.parent / f"{dir.name}-laser-{laser_model_lang}{suffix}"
    l, *_ = dir.name.split("-")

    test = f"{dataset}.test.enc"
    test_labels = f"{dataset}.test.lbl"


    #test_labels = "mldoc.train1000.lbl"
    bsize=12

    m = torch.load(model_save,map_location=lambda storage, loc: storage)
    m = m.cuda()
    m.eval()
    data_loader, lbl = LoadData(base_dir, test + '.' + l,
                           test_labels + '.' + l,
                           dim=dim, bsize=bsize,
                           shuffle=False, quiet=True)

    m.TestCorpus(data_loader, "Test")


    def make_data_set(name, laser_name, fallback_dev="train"):
        print(f"\n Making {name} set")
        train = f"{dataset}.{laser_name}.enc"
        train_labels = f"{dataset}.{laser_name}.lbl"
        data_loader, lbl = LoadData(base_dir, train + '.' + l,
                               train_labels + '.' + l,
                               dim=dim, bsize=bsize,
                               shuffle=False, quiet=True)
        m.TestCorpus(data_loader, name)


        preds = eval_net(m, data_loader)

        fn = dir/f"{l}.{name}.csv"
        if fn.exists():
            df = pd.read_csv(fn, header=None)
            df = df.iloc[(len(df)-len(lbl)):] # account for the training files where first 10% elements were taken as validation
        else:
            df = pd.read_csv(dir/f"{l}.{fallback_dev}.csv", header=None)
            df = df.iloc[:len(lbl)] # if using training only get first n for validatation

        print ("Size", len (df[0]), len(lbl))
        assert (df[0] == lbl).sum() / len(lbl) == 1, "The same labels on both sets"
        df[0] = preds
        print(df.head())
        dst.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst/f"{l}.{name}.csv", index=None, header=None)

    make_data_set("train", f"train{size}")
    make_data_set("dev", "dev")
    shutil.copy(dir / f"{l}.test.csv", dst)
    shutil.copy(dir / f"{l}.unsup.csv", dst)
    # outputs = m.forward(X)
    # _, predicted = torch.max(outputs.data, 1)
    # total += Y.size(0)
    # correct += (predicted == Y).int().sum()
    # for i in range(nlbl):
    #     corr[i] += (predicted == i).int().sum()

if __name__ == "__main__":
    fire.Fire(main)