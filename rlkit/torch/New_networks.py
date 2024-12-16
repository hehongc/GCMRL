import torch
from torch import nn as nn
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

from torch.nn.utils import weight_norm
import torch.nn.init as init
import pywt
import numpy as np
import numpy

from rlkit.torch.networks import Mlp


class SelfAttnEncoder(PyTorchModule):
    def __init__(self,
                 input_dim,
                 num_output_mlp=0,
                 task_gt_dim=5):
        super(SelfAttnEncoder, self).__init__()

        self.input_dim = input_dim
        self.score_func = nn.Linear(input_dim, 1)

        self.num_output_mlp = num_output_mlp

        if num_output_mlp > 0:
            self.output_mlp = Mlp(
                input_size=input_dim,
                output_size=task_gt_dim,
                hidden_sizes=[200 for i in range(num_output_mlp - 1)]
            )

    def forward(self, input):
        if len(input.shape) == 3:
            b, N, dim = input.shape
            scores = self.score_func(input.reshape(-1, dim)).reshape(b, N, -1)
        elif len(input.shape) == 2:
            b, dim = input.shape
            scores = self.score_func(input.reshape(-1, dim)).reshape(b, -1)

        return scores

    def compute_softmax_result(self, input_1, score_1, input_2, score_2):
        scores = torch.cat([score_1, score_2], dim=-1).cuda()
        softmax_scores = F.softmax(scores, dim=-1).cuda()

        if len(input_1.shape) == 3:
            t = input_1.shape[0]
            b = input_1.shape[1]
            softmax_score_1 = softmax_scores[:, :, 0].reshape(t, b, 1)
            combine_input_1 = input_1 * softmax_score_1
            softmax_score_2 = softmax_scores[:, :, 1].reshape(t, b, 1)
            combine_input_2 = input_2 * softmax_score_2
            combine_output = combine_input_1 + combine_input_2
        elif len(input_1.shape) == 2:
            t = input_1.shape[0]
            softmax_score_1 = softmax_scores[:, 0].reshape(t, 1)
            combine_input_1 = input_1 * softmax_score_1
            softmax_score_2 = softmax_scores[:, 1].reshape(t, 1)
            combine_input_2 = input_2 * softmax_score_2
            combine_output = combine_input_1 + combine_input_2

        return combine_output, softmax_score_1, softmax_score_2



class CVAE(nn.Module):
    def __init__(self,
                 hidden_size=64,
                 num_hidden_layers=1,
                 z_dim=20,
                 action_dim=5,
                 state_dim=2,
                 reward_dim=1,
                 use_ib=False,
                 ):
        
        super(CVAE, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.z_dim = z_dim

        self.use_ib = use_ib

        if self.use_ib:
            self.encoder = Mlp(
                input_size=self.state_dim*2+self.action_dim+self.reward_dim,
                output_size=self.z_dim*2,
                hidden_sizes=[hidden_size for i in  range(num_hidden_layers)]
            )
        else:
            self.encoder = Mlp(
                input_size=self.state_dim * 2 + self.action_dim + self.reward_dim,
                output_size=self.z_dim,
                hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
            )

        self.decoder = Mlp(
            input_size=self.z_dim+self.state_dim+self.action_dim,
            output_size=self.state_dim+self.reward_dim,
            hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
        )


class Ratio(nn.Module):
    def __init__(self):
        super(Ratio, self).__init__()
        self.ratio = nn.ParameterList([nn.Parameter(torch.tensor([0., 0.]), requires_grad = True)]).cuda()

    def softmax_radio(self):
        softmax_radio = F.softmax(self.ratio[0], dim=-1)
        return softmax_radio

    def forward(self, x1, x2):
        softmax_radio = self.softmax_radio()
        output = x1 * softmax_radio[0] + x2 * softmax_radio[1]

        return output

class Three_Ratio(nn.Module):
    def __init__(self):
        super(Three_Ratio, self).__init__()
        self.ratio = nn.ParameterList([nn.Parameter(torch.tensor([0., 0., 0.]), requires_grad = True)]).cuda()

    def softmax_radio(self):
        softmax_radio = F.softmax(self.ratio[0], dim=-1)
        return softmax_radio

    def forward(self, x1, x2, x3):
        softmax_radio = self.softmax_radio()
        output = x1 * softmax_radio[0] + x2 * softmax_radio[1] + x3 * softmax_radio[2]

        return output

