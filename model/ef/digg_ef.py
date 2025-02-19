import os
import sys
import numpy as np
import random
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.Supervised_Learning import test, train  # noqa
from unimodals.common_models import (LSTM, MLP, Identity, Sequential,  # noqa
                                     Transformer, GRU)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
save_path = '/mnt/g/mmsynergy/model/tune/ef/digg/'
lr_box = [0.001,0.005, 0.0005, 0.0001, 0.01, 0.00001]
bz_box = [16, 32, 64, 128]
hid_dims = [40]
hidhid_dims = [512]
for lr in lr_box:
    for bz in bz_box:
        for hid in hid_dims:
            for hidhid in hidhid_dims:
                seed = 521
                print("This is seed" + str(seed))
                print("This is batch size" + str(bz))
                print("This is lr" + str(lr))
                print("This is hid" + str(hid))
                print("This is hidhid" + str(hidhid))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                traindata, validdata, testdata = get_dataloader(
                '/mnt/g/mmsynergy/model/data/digg_noplay.pkl', robust_test=False, max_pad=True, batch_size=bz,max_seq_len=50)
                # # mosi/mosei
                # encoders = [Identity().cuda(), Identity().cuda()]
                # head = Sequential(LSTM(325, 60, dropoutp=0.36).cuda(), MLP(60, 100, 1)).cuda()
                # humor/sarcasm
                encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
                head = Sequential(Transformer(325, hid).cuda(),MLP(hid, hidhid, 1)).cuda()
                fusion = ConcatEarly().cuda()
                train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=False, 
                    early_stop=True,lr=lr, save=save_path+'models/lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid), 
                    weight_decay=0.01, objective=torch.nn.L1Loss(), 
                    trainsave=save_path+'checkpoints/train_valid/perf_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid))

                print("Testing:")
                model = torch.load(save_path+'models/lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid)).cuda()
                test(model, testdata, 'affect', is_packed=False,
                    criterion=torch.nn.L1Loss(), task="regression", no_robust=True, 
                    save = save_path+'checkpoints/test/perf_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid))