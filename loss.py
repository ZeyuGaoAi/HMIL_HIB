import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def _dice(true, pred, label):
    true = np.array(true == label, np.int32) 
    pred = np.array(pred == label, np.int32) 
    inter = (pred * true).sum()
    total = (pred + true).sum()
    return 2 * inter /  (total + 1.0e-8)

def _iou(true, pred, label):
    true = np.array(true == label, np.int32)
    pred = np.array(pred == label, np.int32)
    inter = (pred * true).sum()
    total = (pred + true).sum()
    return inter /  (total - inter + 1.0e-8)


def _acc(true, pred):
    matrixs = np.array(true == pred, np.int32)         
    correct = matrixs.sum()
    total = matrixs.size
    return correct/total

def pixel_acc(pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

def calc_sl_loss(pred, target, metrics):
    loss = torch.nn.CrossEntropyLoss()(pred, target)

    pred = nn.functional.softmax(pred, dim=1)
    
    acc = pixel_acc(pred, target)

    metrics['acc'] += acc.data.cpu().numpy() * target.size(0)
    metrics['loss_sl'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def calc_sl_loss_val(pred, target, metrics, tissue_type_dict):
    loss = torch.nn.CrossEntropyLoss()(pred, target)

    pred = nn.functional.softmax(pred, dim=1)
    acc = pixel_acc(pred, target)

    metrics['acc'] += acc.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    pred = torch.argmax(pred, axis=1)
    pred = pred.cpu().data.numpy()
    target = target.data.cpu().numpy()
    
    type_dice_list = []
    type_iou_list = []
    for type_name, type_id in tissue_type_dict.items():
        dice_val = _dice(target, pred, type_id)
        iou_val = _iou(target, pred, type_id)
        type_dice_list.append(dice_val)
        type_iou_list.append(iou_val)
        metrics['dice_'+type_name] += dice_val * target.shape[0]
        metrics['iou_'+type_name] += iou_val * target.shape[0]
        
    metrics['dice'] += np.mean(type_dice_list) * target.shape[0]
    metrics['iou'] += np.mean(type_iou_list) * target.shape[0]

    return loss

def calc_gc_loss(outputs, metrics, cfg):
    outputs_norm = outputs / outputs.norm(dim=1)[:, None]
    affinity_matrix = torch.mm(outputs_norm, outputs_norm.t())
    loss_gc = 0
    batch_size = int(outputs.size(0)/2)
    for i in range(outputs.size(0)):
        cos_sim_i = affinity_matrix[:,i]
        if i < batch_size:
            cos_sim_pos_i = cos_sim_i[i + batch_size]
        else:
            cos_sim_pos_i = cos_sim_i[i - batch_size]
        e_sim = torch.exp(cos_sim_pos_i/cfg.temperature)
        e_all = 0
        for j in range(outputs.size(0)):
            if i == j: # skip self
                continue
            else:
                e_all += torch.exp(cos_sim_i[j]/cfg.temperature)
        loss_gc += -torch.log(e_sim/e_all)
    loss_gc = loss_gc / batch_size
    
    metrics['loss_gc'] += loss_gc.data.cpu().numpy() * batch_size
    return loss_gc

def calc_nc_loss(outputs, metrics, cfg):
    outputs1, outputs2 = outputs
    s, b, c = outputs1.shape
    loss_nc = 0

    for bi in range(b):
        one_outputs = torch.cat((outputs1[:,bi,:], outputs2[:,bi,:]), 0)
        one_outputs_norm = one_outputs / one_outputs.norm(dim=1)[:, None]
        one_affinity_matrix = torch.mm(one_outputs_norm, one_outputs_norm.t())

        for i in range(s*2):
            cos_sim_i = one_affinity_matrix[:,i]
            if i < s:
                cos_sim_pos_i = cos_sim_i[i + s]
            else:
                cos_sim_pos_i = cos_sim_i[i - s]
            e_sim = torch.exp(cos_sim_pos_i/cfg.temperature)
            e_all = 0
            for j in range(s*2):
                if i == j: # skip self
                    continue
                else:
                    e_all += torch.exp(cos_sim_i[j]/cfg.temperature)
            loss_nc += -torch.log(e_sim/e_all)
    
    loss_nc = loss_nc / (b*s)
    
    metrics['loss_nc'] += loss_nc.data.cpu().numpy() * b * s
    return loss_nc

def calc_dc_loss_sp(outputs, sps, metrics, cfg):
    outputs1, outputs2 = outputs
    sps1, sps2 = sps
    s, b, c, w, h = outputs1.shape
    loss_dc = 0

    def get_sps_features(outputs, sps, sps_list):
        pos_list = []
        target_pos_list = []
        one_sps = sps[0, bi, ...] # w, h
        one_outputs = outputs[0, bi, ...] # c, w, h
        one_sps_feature_list = []
        for spi in sps_list:
            sp_index = torch.where(one_sps==spi)
            sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
            sp_feature = torch.mean(sp_feature, -1)
            one_sps_feature_list.append(sp_feature)
        sps_feature_list = torch.stack(one_sps_feature_list)
        return sps_feature_list

    def get_sp_contrast_loss(target_sps_list1, target_sps_list2, affinity_matrix):
            loss_dc_sp = 0

            for i in range(len(target_sps_list1)):

                if target_sps_list1[i] not in repeat_sps_list:
                    continue

                cos_smi_sp = affinity_matrix[i, :]

                e_all = torch.sum(cos_smi_sp)

                e_sim = 0
                sp_indexes = torch.where(target_sps_list2==target_sps_list1[i])
                e_sim += cos_smi_sp[sp_indexes[0][0]]
                loss_dc_sp += -torch.log(e_sim/e_all)
            return loss_dc_sp/len(repeat_sps_list)
    for bi in range(b): # for each case

        target_sps1 = sps1[0,bi,...] # w, h
        target_sps2 = sps2[0,bi,...] # w, h

        target_sps_list1 = torch.unique(target_sps1)
        target_sps_list2 = torch.unique(target_sps2)

        # find same sps in two target lists
        repeat_sps_list = torch.stack([sp for sp in target_sps_list1 if sp in target_sps_list2]).cuda()

        sps_feature_list1 = get_sps_features(outputs1, sps1, target_sps_list1).cuda()
        sps_feature_list2 = get_sps_features(outputs2, sps2, target_sps_list2).cuda()

        sps_feature_list1_norm = sps_feature_list1 / sps_feature_list1.norm(dim=1)[:, None]
        sps_feature_list2_norm = sps_feature_list2 / sps_feature_list2.norm(dim=1)[:, None]
    #     sps_features_norm = torch.cat([sps_feature_list1_norm, sps_feature_list2_norm], 0)
        affinity_matrix = torch.mm(sps_feature_list1_norm, sps_feature_list2_norm.t())
        affinity_matrix = torch.exp(affinity_matrix/cfg.temperature)
        
        loss_sp1 = get_sp_contrast_loss(target_sps_list1, target_sps_list2, affinity_matrix)
        loss_sp2 = get_sp_contrast_loss(target_sps_list2, target_sps_list1, affinity_matrix.t())
        
        loss_dc += (loss_sp1 + loss_sp2)
    
    loss_dc = loss_dc / b
    
    metrics['loss_dc'] += loss_dc.data.cpu().numpy() * b
    return loss_dc

def calc_dc_loss_spp(outputs, sps, metrics, cfg):
    outputs1, outputs2 = outputs
    sps1, sps2 = sps
    s, b, c, w, h = outputs1.shape
    loss_dc = 0
    loss_sp = 0
    loss_px = 0

    def get_sps_features(outputs, sps, sps_list):
        pos_list = []
        target_pos_list = []
        one_sps = sps[0, bi, ...] # w, h
        one_outputs = outputs[0, bi, ...] # c, w, h
        one_sps_feature_list = []
        for spi in sps_list:
            sp_index = torch.where(one_sps==spi)
            sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
            sp_feature = torch.mean(sp_feature, -1)
            one_sps_feature_list.append(sp_feature)
        sps_feature_list = torch.stack(one_sps_feature_list)
        return sps_feature_list

    def get_sp_contrast_loss(target_sps_list, index):
            loss_dc_sp = 0

            for i in range(len(target_sps_list)):

                if target_sps_list[i] not in repeat_sps_list:
                    continue

                cos_smi_sp = affinity_matrix[:, i+index]

                e_all = torch.sum(cos_smi_sp) - cos_smi_sp[i+index]

                e_sim = 0
                sp_indexes = torch.where(sps_list==target_sps_list[i])
                for j in sp_indexes[0]:
                    if j != (i+index):
                        e_sim += cos_smi_sp[j]
                loss_dc_sp += -torch.log(e_sim/e_all)
            return loss_dc_sp/len(repeat_sps_list)
        
    
    def get_loss_per_pixels(outputs, sps, sps_feature_list_norm, affinity_matrix, target_sps_list):
        loss_dc_pixel = 0
        
        one_sps = sps[0, bi, ...] # w, h
        one_outputs = outputs[0, bi, ...] # c, w, h
        
        affinity_matrix_sum = torch.sum(affinity_matrix, 0)
        for i in range(len(target_sps_list)):
            sp_index = torch.where(one_sps==target_sps_list[i])
            pixels_feature = one_outputs[:, sp_index[0], sp_index[1]].T
            sp_feature_norm = sps_feature_list_norm[i:i+1]
            pixels_feature_norm = pixels_feature / pixels_feature.norm(dim=1)[:, None]
            affinity_matrix_pixels = torch.mm(sp_feature_norm, pixels_feature_norm.t())
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels/cfg.temperature)
            e_sim = affinity_matrix_pixels.mean()
            e_dis = affinity_matrix_sum[i] - affinity_matrix[i,i]
            loss_dc_pixel += -torch.log(e_sim/(e_sim+e_dis))
        return loss_dc_pixel/len(target_sps_list)
        

    for bi in range(b): # for each case

        target_sps1 = sps1[0,bi,...] # w, h
        target_sps2 = sps2[0,bi,...] # w, h

        target_sps_list1 = torch.unique(target_sps1)
        target_sps_list2 = torch.unique(target_sps2)

        # find same sps in two target lists
        repeat_sps_list = torch.stack([sp for sp in target_sps_list1 if sp in target_sps_list2]).cuda()

        sps_feature_list1 = get_sps_features(outputs1, sps1, target_sps_list1).cuda()
        sps_feature_list2 = get_sps_features(outputs2, sps2, target_sps_list2).cuda()

        sps_feature_list1_norm = sps_feature_list1 / sps_feature_list1.norm(dim=1)[:, None]
        sps_feature_list2_norm = sps_feature_list2 / sps_feature_list2.norm(dim=1)[:, None]
        sps_features_norm = torch.cat([sps_feature_list1_norm, sps_feature_list2_norm], 0)
        affinity_matrix = torch.mm(sps_features_norm, sps_features_norm.t())
        affinity_matrix = torch.exp(affinity_matrix/cfg.temperature)
        
        affinity_matrix_1 = affinity_matrix[:len(target_sps_list1),:len(target_sps_list1)]
        affinity_matrix_2 = affinity_matrix[len(target_sps_list1):,len(target_sps_list1):]

        sps_list = torch.cat([target_sps_list1, target_sps_list2], 0)

        loss_sp1 = get_sp_contrast_loss(target_sps_list1, 0)
        loss_sp2 = get_sp_contrast_loss(target_sps_list2, target_sps_list1.size(0))
        
        loss_px1 = get_loss_per_pixels(outputs1, sps1, sps_feature_list1_norm, affinity_matrix_1, target_sps_list1)
        loss_px2 = get_loss_per_pixels(outputs2, sps2, sps_feature_list2_norm, affinity_matrix_2, target_sps_list2)

        loss_sp += (loss_sp1 + loss_sp2)
        loss_px += (loss_px1 + loss_px2)
    
    loss_dc = (loss_sp+loss_px) / b
    
    metrics['loss_sp'] += loss_sp.data.cpu().numpy()
    metrics['loss_px'] += loss_px.data.cpu().numpy()
    metrics['loss_dc'] += loss_dc.data.cpu().numpy() * b
    return loss_dc

def calc_dc_loss_mag(outputs, sps, metrics, cfg):
    outputs1, outputs2 = outputs
    sps1, sps2 = sps
    s, b, c, w, h = outputs1.shape
    loss_dc = 0
    
    def get_sps_features(outputs, sps, sps_list):
        sps_feature_list = []
        pos_list = []
        for si in range(s):
            one_sps = sps[si, bi, ...] # w, h
            one_outputs = outputs[si, bi, ...] # c, w, h
            one_sps_feature_list = []
            one_post_list = []
            index_sp = 0
            for spi in sps_list:
                sp_index = torch.where(one_sps==spi)
                if sp_index[0].size(0) == 0:
                    one_post_list.append(-1)
                    continue
                sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
                sp_feature = torch.mean(sp_feature, -1)
                one_sps_feature_list.append(sp_feature)
                one_post_list.append(index_sp)
                index_sp += 1
            one_sps_feature_list = torch.stack(one_sps_feature_list)
            sps_feature_list.append(one_sps_feature_list)
            pos_list.append(one_post_list)
        return sps_feature_list, pos_list
    
    def mag_contrast_loss(sps_feature_list, pos_list):
        loss_dc_mag = 0

        target_feature_list = sps_feature_list[0]
        target_feature_norm = target_feature_list / target_feature_list.norm(dim=1)[:, None]
        mag_feature_list = torch.cat(sps_feature_list[1:], 0)
        mag_feature_norm = mag_feature_list / mag_feature_list.norm(dim=1)[:, None]
        affinity_matrix = torch.mm(target_feature_norm, mag_feature_norm.t())
        affinity_matrix = torch.exp(affinity_matrix/cfg.temperature)

        target_pos_list = pos_list[0]
        for pos in target_pos_list:
            cos_smi = affinity_matrix[pos, :]
            e_all = torch.sum(cos_smi)
            e_sim = 0
            bias = 0
            for si in range(s-1):
                mag_post_list = pos_list[si+1]
                mag_pos = mag_post_list[pos]
                if mag_pos != -1:
                    e_sim += cos_smi[mag_post_list[pos]+bias]
                bias += (len(mag_post_list) - mag_post_list.count(-1))
            loss_dc_mag += -torch.log(e_sim/e_all)
        
        return loss_dc_mag/len(target_pos_list)
    
    for bi in range(b):
        target_sps1 = sps1[0,bi,...] # w, h
        target_sps2 = sps2[0,bi,...] # w, h

        target_sps_list1 = torch.unique(target_sps1)
        target_sps_list2 = torch.unique(target_sps2)
        
        sps_feature_list1, pos_list1 = get_sps_features(outputs1, sps1, target_sps_list1)
        sps_feature_list2, pos_list2 = get_sps_features(outputs2, sps2, target_sps_list2)
        
        loss_mag1 = mag_contrast_loss(sps_feature_list1, pos_list1)
        loss_mag2 = mag_contrast_loss(sps_feature_list2, pos_list2)
        
        loss_dc += (loss_mag1 + loss_mag2)
    
    loss_dc = loss_dc / b
    
    metrics['loss_dc'] += loss_dc.data.cpu().numpy() * b
    return loss_dc
    

def calc_dc_loss(outputs, sps, metrics, cfg):
    outputs1, outputs2 = outputs
    sps1, sps2 = sps
    s, b, c, w, h = outputs1.shape
    loss_dc = 0

    def get_sps_features(outputs, sps, repeat_sps_list, sps_feature_list):
        pos_list = []
        target_pos_list = []
        for si in range(s): # for each mag image with different mag
            one_sps = sps[si, bi, ...] # w, h
            one_outputs = outputs[si, bi, ...] # c, w, h
            one_sps_feature_list = []
            for spi in range(repeat_sps_list.size(0)):
                sp_index = torch.where(one_sps==repeat_sps_list[spi])
                if sp_index[0].size(0) == 0:
                    continue
                sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
                sp_feature = torch.mean(sp_feature, -1)
                one_sps_feature_list.append(sp_feature)
                pos_list.append(spi)
                if si == 0:
                    target_pos_list.append(spi)
            sps_feature_list = torch.cat([sps_feature_list, torch.stack(one_sps_feature_list)], 0)
        return sps_feature_list, pos_list, target_pos_list

    def get_loss_per_sps(repeat_sps_list, affinity_matrix, pos_list, target_pos_list, index):
        loss_dc_sp = 0
        
        for spi in range(len(target_pos_list)):
            cos_smi_sp = affinity_matrix[:, spi+index]

            e_all = torch.sum(cos_smi_sp) - cos_smi_sp[spi+index]

            e_sim = 0
            sp_indexes = torch.where(pos_list==target_pos_list[spi])
            for spj in sp_indexes[0]:
                if spj != (spi+index):
                    e_sim += cos_smi_sp[spj]
            loss_dc_sp += -torch.log(e_sim/e_all)
        return loss_dc_sp/len(target_pos_list)
        

    for bi in range(b): # for each case

        target_sps1 = sps1[0,bi,...] # w, h
        target_sps2 = sps2[0,bi,...] # w, h

        target_sps_list1 = torch.unique(target_sps1)
        target_sps_list2 = torch.unique(target_sps2)

        # find same sps in two target lists
        repeat_sps_list = torch.stack([sp for sp in target_sps_list1 if sp in target_sps_list2]).cuda()

        sps_feature_list1 = torch.empty((0, c)).cuda()
        sps_feature_list2 = torch.empty((0, c)).cuda()

        sps_feature_list1, pos_list1, target_pos_list1 = get_sps_features(outputs1, sps1, repeat_sps_list, sps_feature_list1)
        sps_feature_list2, pos_list2, target_pos_list2 = get_sps_features(outputs2, sps2, repeat_sps_list, sps_feature_list2)

        sps_feature_list_all = torch.cat([sps_feature_list1, sps_feature_list2], 0)
        sps_features_norm = sps_feature_list_all / sps_feature_list_all.norm(dim=1)[:, None]
        affinity_matrix = torch.mm(sps_features_norm, sps_features_norm.t())
        affinity_matrix = torch.exp(affinity_matrix/cfg.temperature)
        
        pos_list = pos_list1 + pos_list2
        pos_list = torch.Tensor(pos_list)
        
        loss_dc1 = get_loss_per_sps(repeat_sps_list, affinity_matrix, pos_list, target_pos_list1, 0)
        loss_dc2 = get_loss_per_sps(repeat_sps_list, affinity_matrix, pos_list, target_pos_list2, len(pos_list1))

        loss_dc += (loss_dc1 + loss_dc2)
    
    loss_dc = loss_dc / b
    
    metrics['loss_dc'] += loss_dc.data.cpu().numpy() * b
    return loss_dc


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))