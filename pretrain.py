
from utils.META_pretrain import meta_pretrain
from utils.META_model import INF_meta
from utils.META_dataset import Dataset_pretrain

local=True
if local:
    root='./datasets/train'
else:
    root ='/gdata/liupj/iTM/sub_train/train'
if local:
    train_list='./datasets/train_list.json'
else:
    train_list ='/gdata/liupj/iTM/train/train_list.json'

if local:
    save_path='./pre_model/meta_fintune_500.pth'
else:
    save_path='/ghome/liupj/projects/iTM/pre_model/meta_fintune_500.pth'



model = INF_meta(update_lr=0.0001,meta_lr=0.0001,finetue_step=500,update_step=5)

Dataset = Dataset_pretrain(root, 50,train_list)

meta_pretrain(dataset=Dataset,model=model,model_save_path=save_path,local=local)