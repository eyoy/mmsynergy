import os
import sys

import torch
import random
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from datasets.affect.get_data_nocontext import get_dataloader, Affectdataset  # noqa
from fusions.common_fusions import Concat  # noqa
from fusions.mult.mult_modified_cross import MULTModel  # noqa
from training_structures.Supervised_Learning_test import test, train  # noqa
from unimodals.common_models import MLP, Identity  # noqa
from torch.utils.data import DataLoader, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_path = '/mnt/g/mmsynergy/model/tune/mult/mult_digg/'

seed = 521
lr_box = [1e-3,5e-4]
bz_box = [16, 32, 128]
for bz in bz_box:
        for lr in lr_box:
                print("This is seed" + str(seed))
                print("This is batch size" + str(bz))
                print("This is lr" + str(lr))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                traindata, validdata,  test_robust = get_dataloader('/mnt/g/mmsynergy/model/data/digg_noplay.pkl', robust_test=False, max_pad=True, batch_size=bz,max_seq_len=50, num_workers=0)
                print(len(traindata), len(validdata), len(test_robust))
                class HParams():
                        num_heads = 10
                        layers = 4
                        attn_dropout = 0.1
                        attn_dropout_modalities = [0, 0]
                        relu_dropout = 0.1
                        res_dropout = 0.1
                        out_dropout = 0.1
                        embed_dropout = 0.2
                        embed_dim = 40
                        attn_mask = True
                        output_dim = 1
                        all_steps = False
                        
                encoders = [Identity().cuda(), Identity().cuda()]
                # vision, audio, text
                #fusion = MULTModel(2, [20, 5, 300], hyp_params=HParams).cuda()
                fusion = MULTModel(2, [272, 53], hyp_params=HParams).cuda()
                # fusion = MULTModel(3, [371, 81, 300], hyp_params=HParams).cuda()
                head = Identity().cuda() 
                # train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, early_stop=True, is_packed=False, 
                # lr=lr, clip_val=1, save= save_path+'models/lr{}_bs{}.pt'.format(lr,bz), 
                # weight_decay=0.01, objective=torch.nn.L1Loss(), 
                # trainsave= save_path+'checkpoints/train_valid/perf_lr{}_bs{}.pt'.format(lr,bz))
                # print("Testing:")
                #model = torch.load(save_path+'models/lr{}_bs{}.pt'.format(lr,bz)).cuda()
                model = torch.load(save_path+'models/lr0.0005_bs16.pt').cuda()


                test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
                criterion=torch.nn.L1Loss(), task='regression', no_robust=True, save = save_path+'checkpoints/test/perf_lr{}_bs{}.pt'.format(lr,bz))