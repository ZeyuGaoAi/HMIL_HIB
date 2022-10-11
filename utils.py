import os
import shutil
import numpy as np
from glob import glob
import scipy.io as sio
import cv2
from histomicstk.preprocessing.color_normalization import reinhard
import torch

def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx
    return new_pred

def watershed_image(mask, thresh=0.5):
    mask = mask*255
    mask = mask.astype('uint8')
    # gray\binary image
    gray = mask
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # morphology operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    # distance transform
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    ret, surface = cv2.threshold(dist, dist.max()*thresh, 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(surface)
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)
    # watershed transfrom
    markers += 1
    markers[unknown == 255] = 0
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask, markers=markers)
    markers[markers == -1] = 0
    markers = markers-1
    markers[markers == -1] = 0
    return markers

def cut_cells(img_nd, labels):
    cells = []
    cell_masks = []
    for i in np.unique(labels):
        if i == 0:
            continue
        cell_x, cell_y = np.where(labels==i)
        min_x = cell_x.min()
        max_x = cell_x.max()
        min_y = cell_y.min()
        max_y = cell_y.max()
        x_dist = max_x - min_x
        y_dist = max_y - min_y
        pad_x = int(x_dist*0.2)
        pad_y = int(y_dist*0.2)
        dist = max(x_dist, y_dist)
        if dist>128:
            continue
        if (x_dist + 2*pad_x) > 128:
            min_x = max(min_x - (128-x_dist)//2, 0)
            max_x = min(max_x + (128-x_dist)//2, 599)
        else:
            min_x = max(min_x - pad_x, 0)
            max_x = min(max_x + pad_x, 599)
        if (y_dist + 2*pad_y) > 128:
            min_y = max(min_y - (128-y_dist)//2, 0)
            max_y = min(max_y + (128-y_dist)//2, 717)
        else:
            min_y = max(min_y - pad_y, 0)
            max_y = min(max_y + pad_y, 717)
        x = img_nd.copy()
        x = x[min_x : max_x, min_y : max_y]
        y = labels.copy()
        y = y[min_x : max_x, min_y : max_y]
        y[y!=i] = 0
        y[y==i] = 1
        cells.append(x)
        cell_masks.append(y)
    return cells, cell_masks

def make_data(img_path, mat_path, patient_id, label, cell_rate):
    image_list = glob(img_path + patient_id + '*')
    image_list.sort()
    patient_data={}
    patient_data['p_label'] = label
    patient_data['c_rate'] = cell_rate
    p_cells=[]
    p_cells_insts=[]
    for image_file in image_list:
        basename = os.path.basename(image_file)
        image_ext = basename.split('.')[-1]
        basename = basename[:-(len(image_ext)+1)]
        
        # get the corresponding `.mat` file
        if not os.path.exists(mat_path + basename + '.mat'):
            continue
        result_mat = sio.loadmat(mat_path + basename + '.mat')
        inst_map = result_mat['inst_map'] 
        labels = watershed_image(inst_map)
        num_labels, labels_connect, stats, centroids = cv2.connectedComponentsWithStats(labels.astype('uint8'), connectivity=8, ltype=None)

        for i in range(num_labels):
            if stats[i][4]<500:
                labels_connect[labels_connect==i]=0
                
        img_nd = np.array(cv2.imread(image_file))
        cells, cells_masks = cut_cells(img_nd, labels_connect)
        
        if len(cells)==0:
            continue
        p_cells+=cells
        p_cells_insts+=cells_masks
            
    patient_data['c_images'] = p_cells
    patient_data['c_insts'] = p_cells_insts
    return patient_data

def padding(x):
    if x.shape[1]<128:
        left = int((128-x.shape[1])/2)
        right = int(128-x.shape[1]-left)
        x = np.pad(x,((0,0),(left,right),(0,0)),'constant',constant_values = (0,0))
    if x.shape[0]<128:
        up = int((128-x.shape[0])/2)
        down = int(128-x.shape[0]-up)
        x = np.pad(x,((up,down),(0,0),(0,0)),'constant',constant_values = (0,0))
    return x

def load_model(model, model_path):
    pretrained_path = model_path
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint, strict=False)
    return model

def get_from_patient(record_path, max_length):
    cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
    }

    record = np.load(record_path, allow_pickle=True)

    patient_label = record[()]['p_label']
    if patient_label == 0:
        patient_label = [0,0,0]
    elif patient_label == 1:
        patient_label = [0,0,1]
    elif patient_label == 2:
        patient_label = [0,1,2]
    elif patient_label == 3:
        patient_label = [0,1,3]
    elif patient_label == 4:
        patient_label = 4
    elif patient_label == 5:
        patient_label = [0,2,4]
    elif patient_label == 6:
        patient_label = [0,2,5]
    elif patient_label == 7:
        patient_label = [0,3,6]
    elif patient_label == 8:
        patient_label = [1,4,7]
    elif patient_label == 9:
        patient_label = [1,4,8]
    elif patient_label == 10:
        patient_label = [1,5,9]
    elif patient_label == 11:
        patient_label = [2,6,10]
    else:
        print('error patient label!')
    
    patient_label = torch.from_numpy(np.array(patient_label))
    cell_imgs = record[()]['c_images']
    cell_rates = record[()]['c_rate']/100
    if 'c_labels' in record[()].keys():
        cell_types = record[()]['c_labels']
        cell_types = [int(i/2) for i in cell_types]
    else:
        cell_types = []

    cell_imgs_augmented = []
    
    # augmentations
    for img in cell_imgs:
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = reinhard(img, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])
        img = padding(img)
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(np.array(img))
        cell_imgs_augmented.append(img)

    return [cell_imgs_augmented, patient_label, torch.from_numpy(np.array(cell_rates)), torch.from_numpy(np.array(cell_types)), cell_imgs]