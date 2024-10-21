import math
import os
import random

import cv2
import torch
import  numpy as np
import json

def sample(target_img,input_img,rate):
    shape=target_img.shape
    spa=np.dstack(np.meshgrid(np.linspace(0,stop=1,num=shape[1]), np.linspace(0,stop=1,num=shape[0])))
    rate=int(math.sqrt(rate))
    if rate<1:
        rate=1
    input_xy=np.concatenate((input_img,spa),axis=2)[::rate,::rate,:]
    out_rgb=target_img[::rate,::rate,:]

    return input_xy,out_rgb,np.concatenate((input_img,spa),axis=2)


class Dataset_pretrain(object):

    def __init__(self, root,rate,name_path):
        self.root = root
        name_list=[]
        with open(name_path,'r') as f:
            name_list=json.load(f)
        self.rate=rate
        self.sdr_img= [os.path.join(self.root, 'sdr/' + x) for x in name_list]
        self.hdr_img= [os.path.join(self.root, 'hdr/' + x.split('.')[0]+'.png') for x in name_list]
        spa=torch.tensor(np.dstack(np.meshgrid(np.linspace(0,stop=1,num=3840), np.linspace(0,stop=1,num=2160))),dtype=torch.float32)
        self.spa=torch.permute(spa,(2,0,1))

    def __getitem__(self, idx):
        spt_input   =[]
        spt_out     =[]
        qry_input   =[]
        qry_out     =[]
        # spa = spa.reshape([-1, 2])
        for i in range(0,3):
            index=random.randint(0,len(self.hdr_img)-1)
            hdr=cv2.imread(self.hdr_img[index],cv2.IMREAD_UNCHANGED)
            hdr=np.array(cv2.cvtColor(hdr,cv2.COLOR_BGR2RGB),dtype=np.float32)/(2**16-1)
            hdr=torch.tensor(hdr,dtype=torch.float32)
            hdr=torch.permute(hdr,(2,0,1))
            # hdr=torch.reshape(hdr,(-1,3))



            sdr=cv2.imread(self.sdr_img[index],cv2.IMREAD_UNCHANGED)
            self.spa = torch.tensor(
                np.dstack(np.meshgrid(np.linspace(0, stop=1, num=sdr.shape[1]), np.linspace(0, stop=1, num=sdr.shape[0]))),
                dtype=torch.float32)
            self.spa = torch.permute(self.spa, (2, 0, 1))
            sdr=np.array(cv2.cvtColor(sdr,cv2.COLOR_BGR2RGB),dtype=np.float32)/(2**8-1)
            sdr=torch.tensor(sdr,dtype=torch.float32)
            sdr=torch.permute(sdr,(2,0,1))
            sdr=torch.concatenate((sdr,self.spa),dim=0)

            spt_out.append(hdr)
            spt_input.append(sdr)

        spt_out=torch.stack(spt_out,dim=0)
        spt_input=torch.stack(spt_input,dim=0)

        index = random.randint(0, len(self.hdr_img)-1)

        hdr = cv2.imread(self.hdr_img[index], cv2.IMREAD_UNCHANGED)
        hdr = np.array(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB), dtype=np.float32) / (2 ** 16 - 1)
        hdr = torch.tensor(hdr, dtype=torch.float32)
        hdr = torch.permute(hdr, (2, 0, 1))
        # hdr = torch.reshape(hdr, (-1, 3))

        sdr = cv2.imread(self.sdr_img[index], cv2.IMREAD_UNCHANGED)
        sdr = np.array(cv2.cvtColor(sdr, cv2.COLOR_BGR2RGB), dtype=np.float32) / (2 ** 8 - 1)
        sdr = torch.tensor(sdr, dtype=torch.float32)

        sdr = torch.permute(sdr, (2, 0, 1))
        sdr = torch.concatenate((sdr, self.spa), dim=0)

        qry_out.append(hdr)
        qry_input.append(sdr)

        qry_out=torch.stack(qry_out,dim=0)
        qry_input=torch.stack(qry_input,dim=0)

        return spt_input,spt_out,qry_input,qry_out

    def __len__(self):
        return int(len(self.hdr_img)/4-1)

class Dataset_test(object):
    def __init__(self, root,rate,name_path):
        self.root = root
        name_list=[]
        # with open(name_path,'r') as f:
        #     name_list=json.load(f)
        name_list=os.listdir('./datasets/sdr')


        self.rate=rate
        self.sdr_img= [os.path.join(self.root, 'sdr/' + x) for x in name_list]
        self.hdr_img= [os.path.join(self.root, 'hdr/' + x.split('.')[0]+'.png') for x in name_list]

    def __getitem__(self, idx):

        # print(self.hdr_img[idx])
        hdr=cv2.imread(self.hdr_img[idx],cv2.IMREAD_UNCHANGED)
        hdr=np.array(cv2.cvtColor(hdr,cv2.COLOR_BGR2RGB),dtype=np.float32)/(2**16-1)
        L_hdr=torch.tensor(hdr)

        sdr=cv2.imread(self.sdr_img[idx],cv2.IMREAD_UNCHANGED)
        sdr=np.array(cv2.cvtColor(sdr,cv2.COLOR_BGR2RGB),dtype=np.float32)/(2**8-1)
        L_sdr=torch.tensor(sdr)


        rgb_xy,out,final=sample(L_hdr,L_sdr,self.rate)

        rgb_xy=np.transpose(rgb_xy,(2,0,1))
        L_hdr = np.transpose(L_hdr, (2, 0, 1))
        L_sdr = np.transpose(L_sdr, (2, 0, 1))
        out = np.transpose(out, (2, 0, 1))
        final = np.transpose(final, (2, 0, 1))


        return L_hdr,L_sdr,rgb_xy,out,final,self.hdr_img[idx]

    def __len__(self):
        return len(self.hdr_img)
