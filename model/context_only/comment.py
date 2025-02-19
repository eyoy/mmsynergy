import os
import sys
import numpy as np
import random
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
from training_structures.unimodal import test, train  # noqa
from unimodals.common_models import GRU, MLP, Identity, Sequential, Transformer, Raw  # noqa

# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# traindata, validdata, testdata = get_dataloader('/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', robust_test=False)
save_path = '/mnt/g/mmsynergy/model/tune/context_only/comment/'

lr_box = [0.0001, 0.0005, 0.001, 0.005, 0.01]
bz_box = [32, 64, 128]
hid_dims = [40]
hidhid_dims = [512]
modality_num = 2
for lr in lr_box:
    for bz in bz_box:
        for hid in hid_dims:
            for hidhid in hidhid_dims:
                seed = 521
                print("This is seed" + str(seed))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                traindata, validdata, testdata = get_dataloader(
                    '/mnt/g/mmsynergy/model/data/comment_contextual_censored.pkl', robust_test=False, max_pad=True, data_type='mosi', 
                    max_seq_len=50,batch_size=bz, num_workers=0)
                # mosi/mosei
                #encoder = Identity().cuda().permute(0, 2, 1)
                encoder = Transformer(785, hid).cuda()
                #encoder = Raw(hid).cuda()
                head = MLP(hid, hidhid, 1).cuda()
                train(encoder, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, lr=lr, early_stop=True,
                    weight_decay=0.001, criterion=torch.nn.L1Loss(), 
                    save_encoder=save_path+'models/encoder_uni_a_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz,hid,hidhid), 
                    save_head=save_path+'models/head_uni_a_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz,hid,hidhid), modalnum=modality_num, 
                    trainsave = save_path+'checkpoints/train_valid/uni_a_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz,hid,hidhid))

                print("Testing:")
                encoder = torch.load(save_path+'models/encoder_uni_a_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz,hid,hidhid)).cuda()
                head = torch.load(save_path+'models/head_uni_a_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz,hid,hidhid))
                test(encoder, head, testdata, 'affect', criterion=torch.nn.L1Loss(),
                    task="regression", modalnum=modality_num, no_robust=True, 
                    save = save_path+'checkpoints/test/test_uni_a_lr{}_bs{}_hid{}_hidhid{}.pt.pt'.format(lr,bz,hid,hidhid))
        # Compare this snippet from MultiBench2/examples/rr/tune/digg_uni_v.py: