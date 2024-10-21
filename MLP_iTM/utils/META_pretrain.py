import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

def meta_pretrain(dataset,model,model_save_path,local):

    torch.set_default_dtype(torch.float32)
    if local:
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0,shuffle=True)
    else:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # dataset = Dataset(root, 50,train_list)
        sampler= DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0,sampler=sampler)
    model.cuda()
    model.train()

    for _ in range(0, 10):
        # torch.save(model.net.state_dict(), model_save_path)
        for spt_input,spt_out,qry_input,qry_out in tqdm(dataloader):
            spt_input=spt_input.cuda()
            spt_out=spt_out.cuda()
            qry_input=qry_input.cuda()
            qry_out=qry_out.cuda()
            a=model(spt_input,spt_out,qry_input,qry_out)
        print(_)
        torch.save(model.net.state_dict(),model_save_path)

