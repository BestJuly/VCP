"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class VCPN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len=3, modality='rgb', class_num=5):
        """
        Args:
            feature_size (int): 512
        """
        super(VCPN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = class_num
        self.fc = nn.Linear(512*tuple_len, self.class_num)
        if modality == 'rgb':
            print('Use normal RGB clips for training')
            self.res = False
        else:
            print('[Warning]: use residual clips')
            self.res = True
        
        self.relu = nn.ReLU(inplace=True)

    def diff(self, x):
        shift_x = torch.roll(x, 1, 2)
        return ((shift_x - x) + 1)/2

    def forward(self, tuple):
        if not self.res:
          f1 = self.base_network(tuple[:, 0, :, :, :, :])
          f2 = self.base_network(tuple[:, 1, :, :, :, :])
          f3 = self.base_network(tuple[:, 2, :, :, :, :])
        else:
          f1 = self.base_network(self.diff(tuple[:, 0, :, :, :, :]))
          f2 = self.base_network(self.diff(tuple[:, 1, :, :, :, :]))
          f3 = self.base_network(self.diff(tuple[:, 2, :, :, :, :]))

        h = torch.cat((f1, f2, f3), dim=1)
        h = self.fc(h)  # logits
        return h



class VCOPN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len, res=False):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.res = res
        if res:
            print('[Warning] Use residual frames.')
        else:
            print('[Warning] Use normal RGB frames.')

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            if self.res:
                shift_clip = torch.roll(clip, 1, 2)
                f.append(self.base_network(clip - shift_clip))
            else:
                f.append(self.base_network(clip))

        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits

        return h


class VCOPN_RNN(nn.Module):
    """Video clip order prediction with RNN."""
    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 1024
        """
        super(VCOPN_RNN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))

        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)

        h = self.fc(hn.squeeze(dim=0))  # logits

        return h



class MultiMode(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, datamode):
        """
        Args:
            feature_size (int): 512
        """
        super(MultiMode, self).__init__()
        self.base_network = base_network
        self.datamode = datamode
        if datamode == 'rgb':
            print('Using RGB clips for anchor/pos/neg')
        if datamode == 'mix':
            print('Using RGB clips for anchor/neg, Res clips for pos')

    def forward(self, tuple):
        anchor = self.base_network(tuple[:, 0, :, :, :, :])

        clip = tuple[:, 1, :, :, :, :]
        if self.datamode == 'mix':
            shift_clip = torch.roll(clip, 1, 2)
            pos =  self.base_network(clip - shift_clip)
        else:
            pos = self.base_network(clip)
        
        neg = self.base_network(tuple[:, 2, :, :, :, :])

        return anchor, pos, neg
