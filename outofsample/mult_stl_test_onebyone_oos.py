import os
import sys
import numpy as np
import torch
import pandas as pd
import json

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from datasets.affect.get_data_test import get_dataloader, Affectdataset, drop_entry, _process_1, _process_2  # noqa
from fusions.common_fusions import Concat  # noqa
from fusions.mult.mult_modified_cross import MULTModel  # noqa
from training_structures.Supervised_Learning_test import test, train, test_single_batch  # noqa
from unimodals.common_models import MLP, Identity  # noqa
from torch.utils.data import DataLoader, Dataset
torch.multiprocessing.set_sharing_strategy('file_system')
import ast
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
traindata, vids_train, validdata, vids_valid, test_robust, vids = get_dataloader('/mnt/g/mmsynergy/model/data/comment_noplay.pkl', 
                                                                                 train_shuffle=False, robust_test=False, max_pad=True, batch_size=1,max_seq_len=50)
vids_train = [str(i) for i in vids_train]
vids_valid = [str(i) for i in vids_valid]
vids = [str(i) for i in vids]
print("The length of vid" +str(len(vids)))
print(vids_train[0])
# with open ('/home/ey/train_id.json', 'w') as f:
#     json.dump(list(vids_train), f)
# with open ('/home/ey/train_id.json', 'w') as f:
#     json.dump(list(vids_valid), f)
# with open ('/home/ey/train_id.json', 'w') as f:
#     json.dump(list(vids), f)
# print("Done")
data_box = [traindata, validdata, test_robust]
id_box = [vids_train, vids_valid, vids]

# model = torch.load('/home/ey/MultiBench2/clean_models/virality/virality_best_modified_attn2_seed31.pt').cuda()
# model =torch.load('/home/ey/MultiBench2/clean_models/digg/digg_best_seed521_newparam_norm.pt')
# model = torch.load('/home/ey/MultiBench2/clean_models/digg/temp_cos.pt')
model = torch.load("/mnt/g/mmsynergy/.pt")
total = [zip(vids_train, traindata),zip(vids_valid, validdata),zip(vids, test_robust)] 
final_data = []
for i in total:
     out_box = []
     cos_box = []
     true_box = []
     for id, data in i:
          print(ast.literal_eval(id))
          id = ast.literal_eval(id)[0]
          _, _, cos, true, out = test_single_batch(model=model, test_batch=data, input_to_float=True, return_sim=True)
          np.savez('/mnt/e/erya/mult_0114/outofsample/sync_matrix/comment/{}.npz'.format(id), cos.detach().cpu())
          print("Successfully saved to /mnt/e/erya/mult_0114/outofsample/sync_matrix/comment/{}.npz".format(id))
          out_box.append(out)
          #cos_box.append(cos)
          true_box.append(true)
     print(len(out_box))
     print(len(true_box))
     df = pd.DataFrame(np.column_stack([id_box[total.index(i)], out_box, true_box]), columns = ['id','comment_pred_stl','comment_true'])
     final_data.append(df)
df = pd.concat(final_data, ignore_index=True)
df.to_csv('/mnt/e/erya/mult_0114/outofsample/predicted/comment_cross_stl.csv')
print("Saved!" + '/mnt/e/erya/mult_0114/outofsample/predicted/comment_cross_stl.csv')

print("Done")



# final_data = []
# for i in range(len(data_box)):
#      print(len(data_box[i]))
#      out1_box = []
#      out2_box = []
#      cos_box = []
#      true1_box = []
#      true2_box = []
#      for batch in data_box[i]:
#           #print("This is batch" + str(batch))
#           ii = data_box[i].index(batch)
#           print(ii)
#           cos, out1, out2, true1, true2 = test_single_batch(model=model, test_batch=batch, input_to_float=True, return_sim=True)
#           np.savez('/mnt/e/erya/sync_matrix/{}.npz'.format(id_box[0][data_box[i].index(batch)]), cos.detach().cpu())
#           print("Successfully saved to /mnt/e/erya/sync_matrix/{}.npz".format(id_box[0][data_box[i].index(batch)]))
#           out1_box.append(out1)
#           out2_box.append(out2)
#           cos_box.append(cos)
#           true1_box.append(true1)
#           true2_box.append(true2)
#      print(len(out1_box))
#      print(len(out1_box))
#      print(len(cos_box))
#      print(len(true1_box))
#      print(len(id_box[i]))
#      print(len(true1_box))
#      df = pd.DataFrame(np.column_stack([id_box[i], cos_box, out1_box, out2_box, true1_box, true2_box ]), columns = ['id','cos','digg_pred','comment_pred','digg_true', 'comment_true'])
#      final_data.append(df)
# df = pd.concat(final_data, ignore_index=True)
# df.to_csv('/mnt/e/erya/digg.csv')
# print("Saved!" + '/mnt/e/erya/digg.csv')

# print("Done")


