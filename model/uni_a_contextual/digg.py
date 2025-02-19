import os
import sys
import random
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import Concat  # noqa
from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.Supervised_Learning_test_bimodal import test, train  # noqa
from unimodals.common_models import LSTM, MLP, Transformer  # noqa
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
save_path = '/mnt/g/mmsynergy/model/tune/uni_a_context/digg/'

lr_box = [0.001,0.005, 0.0005, 0.0001]
bz_box = [16, 32, 64, 128]
hid_dims = [40]
hidhid_dims = [512]
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
                traindata, validdata, test_robust = \
                    get_dataloader( '/mnt/g/mmsynergy/model/data/digg_contextual_censored.pkl', robust_test=False, batch_size = bz, max_seq_len=50)

                # mosi/mosei
                # encoders = [LSTM(272, 60, dropoutp=0.02).cuda(),
                #             LSTM(53, 400,dropoutp=0.03).cuda()]
                encoders = [Transformer(53, hid).cuda(),
                            Transformer(785, hid).cuda()]
                head = MLP(hid+hid, hidhid, 1).cuda()

                # humor/sarcasm
                # encoders=[Transformer(371,400).cuda(), \
                #     Transformer(81,100).cuda(),\
                #     Transformer(300,600).cuda()]
                # head=MLP(1100,256,1).cuda()

                fusion = Concat().cuda()

                train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
                    early_stop=True, is_packed=False, lr=lr, save=save_path+'models/lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid), weight_decay=0.01, objective=torch.nn.L1Loss(), 
                    trainsave= save_path+'checkpoints/train_valid/perf_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid))


                print("Testing:")
                model = torch.load(save_path+'models/lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid)).cuda()

                test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
                    criterion=torch.nn.L1Loss(), task='regression', no_robust=True, 
                    save = save_path+'checkpoints/test/perf_lr{}_bs{}_hid{}_hidhid{}.pt'.format(lr,bz, hid, hidhid))