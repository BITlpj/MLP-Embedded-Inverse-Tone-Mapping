import torch
import os
import torch.nn as nn
import numpy as np
import math

root='E:/dataset/HDRTV4K_sub/test'
def EOTF_PQ_cuda(ERGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ERGB = torch.clamp(ERGB, min=1e-10, max=1)

    X1 = ERGB ** (1 / m2)
    X2 = X1 - c1
    X2[X2 < 0] = 0

    X3 = c2 - c3 * X1

    X4 = (X2 / X3) ** (1 / m1)
    return X4 * 10000
def EOTF_PQ_cuda_inverse(LRGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    RGB_l = LRGB / 10000
    RGB_l = torch.clamp(RGB_l, min=1e-10, max=1)

    X1 = c1 + c2 * RGB_l ** m1
    X2 = 1 + c3 * RGB_l ** m1
    X3 = (X1 / X2) ** m2
    return X3

def HDR_to_ICTCP(ERGB, dim=1):
    LRGB = EOTF_PQ_cuda(ERGB)  # hw3
    LR, LG, LB = torch.split(LRGB, 1, dim=dim)  # hw1

    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=dim)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=dim)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I, T, P], dim=dim)  # hw3
    return ITP

def gamma(r):
    r2 = r / 12.92
    index = r > 0.04045  # pow:0.0031308072830676845,/12.92:0.0031308049535603713
    r2[index] = torch.pow((r[index] + 0.055) / 1.055, 2.4)
    return r2

def anti_g(r):
    r2 = r * 12.92
    index = r > 0.0031308072830676845
    r2[index] = torch.pow(r[index], 1.0 / 2.4) * 1.055 - 0.055
    return r2

def F(X):  # X为任意形状的张量
    FX = 7.787 * X + 0.137931
    index = X > 0.008856
    FX[index] = torch.pow(X[index], 1.0 / 3.0)
    return FX


def anti_F(X):  # 逆操作。
    tFX = (X - 0.137931) / 7.787
    index = X > 0.206893
    tFX[index] = torch.pow(X[index], 3)
    return tFX

def myPSrgb2lab(img):  # RGB img:[b,3,h,w]->lab,L[0,100],AB[-127,127]
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    r = gamma(r)
    g = gamma(g)
    b = gamma(b)

    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470
    X = X / 0.964221
    Z = Z / 0.825211

    F_X = F(X)
    F_Y = F(Y)
    F_Z = F(Z)

    # L = 903.3*Y
    # index = Y > 0.008856
    # L[index] = 116 * F_Y[index] - 16 # [0,100]
    L = 116 * F_Y - 16.0
    a = 500 * (F_X - F_Y)  # [-127,127]
    b = 200 * (F_Y - F_Z)  # [-127,127]

    # L = L
    # a = (a+128.0)
    # b = (b+128.0)
    return torch.stack([L, a, b], dim=1)

class delta_e_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        lab_1=myPSrgb2lab(x)
        lab_2=myPSrgb2lab(y)
        return torch.mean(torch.pow((lab_1[:,0,:,:]-lab_2[:,0,:,:]),2)+torch.pow((lab_1[:,1,:,:]-lab_2[:,1,:,:]),2)+torch.pow((lab_1[:,1,:,:]-lab_2[:,1,:,:]),2))
        # ICtCp_1=HDR_to_ICTCP(x)
        # ICtCp_2=HDR_to_ICTCP(y)
        # return 720 *torch.mean(torch.sqrt(torch.pow((ICtCp_1[:,0,:,:]-ICtCp_2[:,0,:,:]),2)+0.25*torch.pow((ICtCp_1[:,1,:,:]-ICtCp_2[:,1,:,:]),2)+torch.pow((ICtCp_1[:,1,:,:]-ICtCp_2[:,1,:,:]),2)))


def mkdir(path, fileName):
    url =os.path.join(path,fileName)
    if os.path.exists(url):
        print("exist")
    else:
        os.mkdir(url)

def online_sample(loss,sample_input,gt,rate):
    # print(pre)
    # print(sample_input)
    # print(gt)
    # loss=torch.abs(gt-pre)
    loss=torch.sum(loss,dim=2)
    # print(loss[0].shape)
    len=int(gt.shape[1]*rate)
    index=torch.multinomial(loss[0],len)
    # loss_index=torch.argsort(loss,dim=1,descending=True)

    gt=gt[:,index,:]
    sdr=sample_input[:,index,:]

    # os.system('pause')

    return sdr,gt


def online_sample_light(loss,sample_input,gt,rate):
    # print(loss.shape)
    loss=torch.sum(loss,dim=1)
    total_rate=math.sqrt(rate)
    total_len=int(gt.shape[2]*total_rate)
    total_width=int(gt.shape[3]*total_rate)
    total=total_len*total_width
    ori_len=gt.shape[2]
    ori_width=gt.shape[3]

    rate=math.sqrt(rate*0.2)
    len=int(gt.shape[2]*rate)
    width=int(gt.shape[3]*rate)
    ori_sample=total-len*width

    loss=torch.reshape(loss,(1,-1))

    index=torch.multinomial(loss,len*width)
    index=index.reshape(-1)
    gt=torch.reshape(gt,(3,-1))
    # print(np.linspace(0,ori_len*ori_width,ori_sample,dtype=np.int32))
    # print(gt.shape)
    # print(gt[:,np.linspace(0,ori_len*ori_width,ori_sample,dtype=np.int32)].shape)
    gt=torch.cat([gt[:,index],gt[:,np.clip(np.linspace(0,ori_len*ori_width,ori_sample,dtype=np.int32),a_min=0,a_max=ori_len*ori_width-1)]],dim=1)
    sample_input=torch.reshape(sample_input,(5,-1))

    # print(sample_input.shape)
    sample_input=torch.cat([sample_input[:,index],sample_input[:,np.clip(np.linspace(0,ori_len*ori_width,ori_sample,dtype=np.int32),a_min=0,a_max=ori_len*ori_width-1)]],dim=1)

    sample_input=torch.reshape(sample_input,(1,5,total_len,total_width))

    gt = torch.reshape(gt, (1, 3,total_len, total_width))
    return sample_input,gt

def load_metric(path):
    psnr_path=os.path.join(path,'psnr.npy')
    delta_path=os.path.join(path,'delta.npy')
    psnr=np.load(psnr_path,allow_pickle=True).item()
    delta=np.load(delta_path,allow_pickle=True).item()
    return psnr,delta

