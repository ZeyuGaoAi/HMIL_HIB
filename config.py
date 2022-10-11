from yacs.config import CfgNode as CN

_C = CN()

_C.model_name = "HMIL-HIB"

_C.n_class = [3, 7] # [3, 7, 11]
_C.max_length = 50

_C.pretrained = True
_C.freeze = False
_C.pretrained_path = ""

_C.instance_beta = 0

_C.IB_use = False
_C.IB_beta1 = 1e-2

_C.ClUB_beta2 = 1e-4
_C.ClUB_sample_dim = 128
_C.ClUB_hidden_size = 10

_C.T21 = [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
_C.T32 = [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

# dataset
_C.data_inst = "dataset"
_C.train_data_dir = "/home1/gzy/cls_scmc/%s/train"
_C.val_data_dir = "/home1/gzy/cls_scmc/%s/test"

# optimization
_C.lr = 1e-3
_C.CLUB_lr = 1e-4
_C.CLUB_iter_per_epoch = 5

_C.num_epochs = 100
_C.batch_size = 1
_C.num_workers = 4
