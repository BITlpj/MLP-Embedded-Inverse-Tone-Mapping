import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
# from utils import delta_e_loss
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        # self.linear = nn.Linear(in_f, out_f)
        self.linear=nn.Conv2d(in_channels=in_f,out_channels=out_f,stride=1,kernel_size=1,padding=0)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
class INF(nn.Module):
    def __init__(self, num_layers, hidden_dim, add_layer, weight_decay=None):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''
        color_layers = [SirenLayer(3, hidden_dim, is_first=True)]
        spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]
        output_layers = []

        for _ in range(1, add_layer - 2):
            color_layers.append(SirenLayer(hidden_dim, hidden_dim))
            spatial_layers.append(SirenLayer(hidden_dim, hidden_dim))
        color_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))

        for _ in range(add_layer, num_layers - 1):
            output_layers.append(SirenLayer(hidden_dim, hidden_dim))
        output_layers.append(SirenLayer(hidden_dim, 3, is_last=True))

        self.color_net = nn.Sequential(*color_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)


        self.color_net = nn.Sequential(*color_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)


    def forward(self, rgb_xy ):
        return self.output_net(torch.cat((self.color_net(rgb_xy[:,0:3,:,:]), self.spatial_net(rgb_xy[:,3:5,:,:])), 1))


    def load(self,parameters):
        self_para=self.state_dict()
        for num,i in enumerate(self_para):
            self_para[i]=parameters[num]
        self.load_state_dict(self_para)

class INF_meta(nn.Module):
    def __init__(self,update_lr,meta_lr,finetue_step,update_step):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''
        self.net = INF(num_layers=3, hidden_dim=64, add_layer=1)
        # self.params=self.net.params
        self.update_step = update_step
        self.finetue_step=finetue_step
        self.update_lr=update_lr
        self.meta_lr=meta_lr
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):

        task_num = x_qry.shape[0]

        ori_para=self.net.parameters()

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i])

            loss = F.mse_loss(logits, y_spt[i])

            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            self.net.load(fast_weights)

            for k in range(1, self.update_step):

                logits = self.net(x_spt[i])

                loss = F.mse_loss(logits, y_spt[i])
                grad = torch.autograd.grad(loss,self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                self.net.load(fast_weights)

                # losses_q[k + 1] = loss_q
            for k in range(1,self.finetue_step):
                logits_q = self.net(x_qry[i])

                loss_q = F.mse_loss(logits_q, y_qry[i])
                grad = torch.autograd.grad(loss_q, self.net.parameters())

                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                self.net.load(fast_weights)

        logits_q = self.net(x_qry[i])

        loss_q = F.mse_loss(logits_q, y_qry[i])

        grad = torch.autograd.grad(loss_q, self.net.parameters())

        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, ori_para)))
        self.net.load(fast_weights)

        return