import os
import math
import argparse
import warnings
import torch
import random
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import rm_n_mkdir
from config import _C as cfg
from loss import print_metrics
from models_hierarchy import ResNetMTL_InfoMin_CLUB
from dataset import TrainDataset_hierarchy, ValDataset_hierarchy
from CLUB.mi_estimators import CLUB

warnings.filterwarnings('ignore')

def train_phase(model, optimizer, mi_estimator, mi_optimizer, dataloader, epoch):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    
    correct_p_3 = 0
    total_p_3 = 0
    correct_p_7 = 0
    total_p_7 = 0
    correct_p_11 = 0
    total_p_11 = 0
    loss = 0
    estimator_copy = CLUB(cfg.ClUB_sample_dim, cfg.ClUB_sample_dim, cfg.ClUB_hidden_size).cuda()
    for cell_imgs, patient_label, cell_rates in tqdm(dataloader):

        cell_imgs = torch.cat(cell_imgs, dim = 0).cuda()
        
        patient_label_3 = patient_label[0].cuda()
        patient_one_hot_3 = torch.nn.functional.one_hot(patient_label_3, cfg.n_class[0]).cuda()
        
        patient_label_7 = patient_label[1].cuda()
        patient_one_hot_7 = torch.nn.functional.one_hot(patient_label_7, cfg.n_class[1]).cuda()
        
        if len(cfg.n_class) > 2:
            patient_label_11 = patient_label[2].cuda()
            patient_one_hot_11 = torch.nn.functional.one_hot(patient_label_11, cfg.n_class[2]).cuda()

        cell_rates = torch.stack((1-cell_rates, cell_rates)).float().cuda()
        cell_rates = cell_rates.transpose(1,0)
        
        T21_matrix = T21.float().cuda()
        T32_matrix = T32.float().cuda()
        
        if len(cfg.n_class) > 2:
            T_matrix = [T21_matrix, T32_matrix]
        else:
            T_matrix = [T21_matrix]

        out_instance, A_raw, logits = model(cell_imgs, T_matrix)

        z1 = A_raw[5].detach()
        z2 = A_raw[6].detach()
        dis1 = A_raw[7]
        dis2 = A_raw[8]
        
        logits_3 = logits[0].permute(1,0)
        logits_3 = logits_3.unsqueeze(0)
        
        logits_7 = logits[1].permute(1,0)
        logits_7 = logits_7.unsqueeze(0)
        if len(cfg.n_class) > 2:
            logits_11 = logits[2].permute(1,0)
            logits_11 = logits_11.unsqueeze(0)
        
        out_instance_s = torch.nn.functional.gumbel_softmax(out_instance, hard=True)
        
        predicts_rate = []
        p_num = patient_label_3.shape[0]
        for i in range(p_num):
            out_instance_sub = out_instance_s[i*cfg.max_length:(i+1)*cfg.max_length]
            predicts_rate_sub = torch.sum(out_instance_sub, dim=0)/out_instance.shape[0] * p_num
            predicts_rate.append(predicts_rate_sub)
        predicts_rate = torch.stack(predicts_rate)
        loss_instance = torch.nn.MSELoss()(predicts_rate, cell_rates)
        
        loss_patient = torch.nn.CrossEntropyLoss()(logits_3, patient_one_hot_3)
        loss_patient += torch.nn.CrossEntropyLoss()(logits_7, patient_one_hot_7)
        
        if len(cfg.n_class) > 2:
            loss_patient += torch.nn.CrossEntropyLoss()(logits_11, patient_one_hot_11)

        loss = cfg.instance_beta*loss_instance + loss_patient
        
        Mu, Sigma = A_raw[0], A_raw[1]
        loss_kl = -0.5*(1+2*Sigma.log()-Mu.pow(2)-Sigma.pow(2)).sum(1).mean().div(math.log(2))
        loss += cfg.IB_beta1*loss_kl
        
        estimator_copy.load_state_dict(mi_estimator.state_dict())
        loss_mi = estimator_copy(z1,z2)
        loss += cfg.ClUB_beta2*loss_mi

        predicted_cells = torch.argmax(out_instance, 1)

        logits_T_3 = logits_3.transpose(1,0)
        logits_out_3 = logits_T_3[1]
        correct_p_3 += torch.argmax(logits_out_3, dim =1).eq(patient_label_3).sum().item()
        total_p_3 += logits_out_3.shape[0]
        
        logits_T_7 = logits_7.transpose(1,0)
        logits_out_7 = logits_T_7[1]
        correct_p_7 += torch.argmax(logits_out_7, dim =1).eq(patient_label_7).sum().item()
        total_p_7 += logits_out_7.shape[0]
        
        if len(cfg.n_class) > 2:
            logits_T_11 = logits_11.transpose(1,0)
            logits_out_11 = logits_T_11[1]
            correct_p_11 += torch.argmax(logits_out_11, dim =1).eq(patient_label_11).sum().item()
            total_p_11 += logits_out_11.shape[0]

        metrics['loss_instance'] += loss_instance.data.cpu().numpy()
        metrics['loss_patient'] += loss_patient.data.cpu().numpy()
        metrics['loss_IB'] += loss_kl.data.cpu().numpy()
        metrics['MI'] += loss_mi.data.cpu().numpy()
        metrics['loss'] += loss.data.cpu().numpy()

        epoch_samples += 1
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True) #retain_graph=True
        optimizer.step()
        
        mi_estimator.train()
        for j in range(cfg.CLUB_iter_per_epoch):
            x_samples = dis1.rsample().detach()
            y_samples = dis2.rsample().detach()
            mi_loss = mi_estimator.learning_loss(x_samples, y_samples)
            metrics['mi_loss'] += mi_loss.data.cpu().numpy()
            mi_optimizer.zero_grad()
            mi_loss.backward() #retain_graph=True
            mi_optimizer.step()
        mi_estimator.eval()
        

    epoch_loss = metrics['loss'] / epoch_samples
    metrics['patient_acc_3'] = (correct_p_3/total_p_3) * epoch_samples
    metrics['patient_acc_7'] = (correct_p_7/total_p_7) * epoch_samples
    if len(cfg.n_class) > 2:
        metrics['patient_acc_11'] = (correct_p_11/total_p_11) * epoch_samples

    print_metrics(metrics, epoch_samples, 'train')
    
    for k in metrics.keys():
        writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

    scheduler.step(epoch_loss)
            
def test_phase(model, dataloader, best_acc, epoch):

    metrics = defaultdict(float)
    epoch_samples = 0
    correct_c = 0
    total_c = 0
    correct_p_3 = 0
    total_p_3 = 0
    correct_p_7 = 0
    total_p_7 = 0
    correct_p_11 = 0
    total_p_11 = 0

    for cell_imgs, patient_label, cell_rates in tqdm(dataloader):
            
        cell_imgs = torch.cat(cell_imgs, dim = 0).cuda()

        patient_label_3 = patient_label[0].cuda()
        patient_label_7 = patient_label[1].cuda()
        patient_one_hot_3 = torch.nn.functional.one_hot(patient_label_3, cfg.n_class[0]).cuda()
        patient_one_hot_7 = torch.nn.functional.one_hot(patient_label_7, cfg.n_class[1]).cuda()
        
        if len(cfg.n_class) > 2:
            patient_label_11 = patient_label[2].cuda()
            patient_one_hot_11 = torch.nn.functional.one_hot(patient_label_11, cfg.n_class[2]).cuda()

        cell_rates = torch.stack((1-cell_rates, cell_rates)).float().cuda()
        cell_rates = cell_rates.transpose(1,0)

        T21_matrix = T21.float().cuda()
        T32_matrix = T32.float().cuda()
        
        if len(cfg.n_class) > 2:
            T_matrix = [T21_matrix, T32_matrix]
        else:
            T_matrix = [T21_matrix]

        with torch.no_grad():
            out_instance, A_raw, logits = model.test(cell_imgs, T_matrix)
        
        logits_3 = logits[0].permute(1,0)
        logits_7 = logits[1].permute(1,0)
        logits_3 = logits_3.unsqueeze(0)
        logits_7 = logits_7.unsqueeze(0)
        
        if len(cfg.n_class) > 2:
            logits_11 = logits[2].permute(1,0)
            logits_11 = logits_11.unsqueeze(0)
        
        out_instance_s = torch.nn.functional.gumbel_softmax(out_instance, hard=True)
                                    
        predicts_rate = []
        p_num = patient_label_3.shape[0]
        for i in range(p_num):
            out_instance_sub = out_instance_s[i*cfg.max_length:(i+1)*cfg.max_length]
            predicts_rate_sub = torch.sum(out_instance_sub, dim=0)/out_instance.shape[0] * p_num
            predicts_rate.append(predicts_rate_sub)            
        predicts_rate = torch.stack(predicts_rate)
        
        loss_instance = torch.nn.MSELoss()(predicts_rate, cell_rates)
        loss_patient = torch.nn.CrossEntropyLoss()(logits_3, patient_one_hot_3)
        loss_patient += torch.nn.CrossEntropyLoss()(logits_7, patient_one_hot_7)
        if len(cfg.n_class) > 2:
            loss_patient += torch.nn.CrossEntropyLoss()(logits_11, patient_one_hot_11)
        
        loss = cfg.instance_beta*loss_instance + loss_patient

        predicted_cells = torch.argmax(out_instance, 1)

        logits_T_3 = logits_3.transpose(1,0)
        logits_out_3 = logits_T_3[1]
        correct_p_3 += torch.argmax(logits_out_3, dim =1).eq(patient_label_3).sum().item()
        total_p_3 += logits_out_3.shape[0]
        
        logits_T_7 = logits_7.transpose(1,0)
        logits_out_7 = logits_T_7[1]
        correct_p_7 += torch.argmax(logits_out_7, dim =1).eq(patient_label_7).sum().item()
        total_p_7 += logits_out_7.shape[0]
        
        if len(cfg.n_class) > 2:
            logits_T_11 = logits_11.transpose(1,0)
            logits_out_11 = logits_T_12[1]
            correct_p_11 += torch.argmax(logits_out_11, dim =1).eq(patient_label_11).sum().item()
            total_p_11 += logits_out_11.shape[0]
        
        metrics['loss_instance'] += loss_instance.data.cpu().numpy()
        metrics['loss_patient'] += loss_patient.data.cpu().numpy()
        
        metrics['loss'] += loss.data.cpu().numpy()

        epoch_samples += 1

    epoch_loss = metrics['loss'] / epoch_samples
    metrics['patient_acc_3'] = (correct_p_3/total_p_3) * epoch_samples
    metrics['patient_acc_7'] = (correct_p_7/total_p_7) * epoch_samples
    if len(cfg.n_class) > 2:
        metrics['patient_acc_11'] = (correct_p_11/total_p_11) * epoch_samples

    print_metrics(metrics, epoch_samples, 'val')
    
    for k in metrics.keys():
        writer.add_scalar('val/%s' % k, metrics[k] / epoch_samples, epoch)

    acc_all = metrics['patient_acc_3'] + metrics['patient_acc_7']
    
    if len(cfg.n_class) > 2:
        acc_all += metrics['patient_acc_11']
        
    if acc_all > best_acc:
        print(f"saving best model to {checkpoint_path.replace('.pth','.best')}")
        best_acc = acc_all
        torch.save(model.state_dict(), checkpoint_path.replace('.pth','.best'))
        
    return best_acc

def build_HAMapping():
    T21 = np.array(cfg.T21)

    T21 = (T21.T / T21.sum(axis=1)).T
    T21 = torch.from_numpy(T21)

    T32 = np.array(cfg.T32)

    T32 = (T32.T / T32.sum(axis=1)).T
    T32 = torch.from_numpy(T32)
    return T21, T32


def build_model():
    model = ResNetMTL_InfoMin_CLUB(cfg.n_class, freeze=cfg.freeze, pretrained=cfg.pretrained).cuda()
        
    for name,parameters in model.named_parameters():
        print(name,':',parameters.shape)
    
    return model

def build_dataset():      
    dataset_train = TrainDataset_hierarchy(cfg.train_data_dir % cfg.data_inst, cfg)
    print(cfg.train_data_dir % cfg.data_inst)
    dataset_val = ValDataset_hierarchy(cfg.val_data_dir % cfg.data_inst, cfg)
    print(cfg.val_data_dir % cfg.data_inst)

    return dataset_train, dataset_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--gpus",
        default="4",
        help="gpus to use, 0,1,2,3"
    )
    parser.add_argument(
        "--data",
        default="fold1",
        help="data to use, string"
    )
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    ## check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    
    ## Parameters
    cfg.data_inst = args.data
        
    checkpoint_path = "./checkpoints/%s_%s.pth" % (cfg.model_name, cfg.data_inst)
        
    print("model will be save to %s" % checkpoint_path)
    
    
    rm_n_mkdir('./logs/%s_%s/' % (cfg.model_name, cfg.data_inst))
    writer = SummaryWriter('./logs/%s_%s/' % (cfg.model_name, cfg.data_inst))
    print("log dir is set to ./logs/%s_%s/" % (cfg.model_name, cfg.data_inst))

    best_loss = 1e10
    best_acc = 0
    
    ## build dataset
    dataset_train, dataset_val = build_dataset()

    dataloaders = {
      'train': DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=True),
      'val': DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, 
                        num_workers=cfg.num_workers, pin_memory=True)
    }
    
    ## build models
    model = build_model().cuda()
#     model = nn.DataParallel(model)

    mi_estimator = CLUB(cfg.ClUB_sample_dim, cfg.ClUB_sample_dim, cfg.ClUB_hidden_size).cuda()
    mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr = cfg.CLUB_lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    
    if cfg.pretrained_path != "":
        checkpoint = torch.load(cfg.pretrained_path)
        print(model.load_state_dict(checkpoint, strict=False))
        model.load_state_dict(checkpoint, strict=False)
        print("load pretrained weights from %s" % cfg.pretrained_path)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    
    ## build HAM
    T21, T32 = build_HAMapping()
    
    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))

        since = time.time()
        train_phase(model, optimizer, mi_estimator, mi_optimizer, dataloaders['train'], epoch)
        best_acc = test_phase(model, dataloaders['val'], best_acc, epoch)
            
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        print(f"saving current model to {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)
        
        
