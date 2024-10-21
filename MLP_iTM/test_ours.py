from loop import loop_online_train_conv

local_name='local'

if local_name=='local':
    local=True
elif local_name=='vsation':
    local=False
elif local_name=='cluster':
    local=True

if local_name=='local':
    home_path='./'
elif local_name == 'vsation':
    home_path='/home/liupj/projects/iTM/iTM'
elif local_name == 'cluster':
    home_path='/ghome/liupj/projects/iTM'

if local_name=='local':
    path='./results/meta_delta'
elif local_name == 'vsation':
    path='/data/liupj/results/iTM/results/meta_delta'
elif local_name == 'cluster':
    path = '/gdata1/liupj/iTM/results/meta_delta'

if local_name=='local':
    model_path='./pre_model/meta_fintune_500.pth'
elif local_name == 'vsation':
    model_path='/home/liupj/projects/iTM/iTM/pre_model/meta_fintune_100.pth'
elif local_name == 'cluster':
    model_path='/ghome/liupj/projects/iTM/pre_model/meta_delta.pth'

if local_name=='local':
    dataset='./datasets/test'
elif local_name == 'vsation':
    dataset='/data/liupj/HDRTV4K'
elif local_name == 'cluster':
    dataset='/gdata/liupj/iTM/HDRTV4K'


loop_online_train_conv(epoch=200,rate=50,save_path=path,model_path=model_path,dataset_root=dataset)
