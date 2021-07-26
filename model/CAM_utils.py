import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
import os
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image, ImageDraw
import cv2
from scipy import ndimage

import pickle
from torch.autograd import Variable
import json

def load_img(img_dir):
    img = np.load(img_dir)
    # print("img.shape", img.shape)
    img = np.expand_dims(np.expand_dims(np.squeeze(img), axis=0), axis=0)
    # print("img2.shape", img.shape)
    input = torch.autograd.Variable(torch.from_numpy(img.copy()).float())
    input = input.cuda()
    return input

def show_cam_on_image_plt(img, mask, file_path="cam.jpg", raw_path=None):
    img = np.transpose(img, (1, 2, 0))

    # mask = mask / np.max(mask)
    # print("mask[0]:", mask[0][0], mask_max)
    # mask[0][0] = 100
    mask_max = np.max(mask)
    mask_min = np.min(mask)

    # Show the raw image
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    plt.subplots_adjust(wspace=0.4)
    plt.axis('off')
    plt.imshow(img)
    plt.title('raw img')

    # Show the raw heatmap
    a = fig.add_subplot(1, 3, 2)
    mask = (mask-mask_min) / (mask_max - mask_min)
    plt.imshow(mask, cmap='jet')
    plt.title('attention map')
    plt.colorbar(cmap='jet', fraction=0.046, pad=0.04)

    # normalize the heatmap to [0,1], and add it with the image
    # mask = (mask-mask_min)/(mask_max - mask_min)
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cam = 0.8*heatmap + np.float32(img)

    a = fig.add_subplot(1, 3, 3)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(cam)
    plt.title('weighted img')

    # plt.savefig(file_path.replace('.jpg', '_Map_max_{}_min_{}.jpg'.format(mask_max, mask_min)))
    plt.savefig(file_path)

    if raw_path is not None:
        raw = np.float32(img)
        raw = raw / np.max(raw)
        cv2.imwrite(raw_path, np.uint8(255 * raw))  # raw图像


def save_mask_as_pdf_bb(mask, img, img_dim, fig, N_layer, name, j, KP_BB_dict, mask_BB_dict):
    '''
    In this function, we draw the attention map with bounding box
    '''
    mask = cv2.resize(mask, (img_dim, img_dim))
    # print('\nmask:\n', mask)
    mask_max = np.max(mask)
    mask_min = np.min(mask)
    # save_mask_as_pdf_bbhow the raw heatmap
    a = fig.add_subplot(N_layer, 2, 2 * j + 1)
    mask = (mask - mask_min) / (mask_max - mask_min)
    plt.imshow(np.uint8(255 * mask), cmap='jet')
    plt.title('{} activation map'.format(name))
    plt.colorbar(cmap='jet', fraction=0.046, pad=0.04)
    rect_KP = plt.Rectangle((KP_BB_dict['y2'], KP_BB_dict['x2']), KP_BB_dict['y1']-KP_BB_dict['y2'],
                         KP_BB_dict['x1']-KP_BB_dict['x2'], fill=False, edgecolor='red', linewidth=1)
    a.add_patch(rect_KP)
    rect_mask = plt.Rectangle((mask_BB_dict['y2'], mask_BB_dict['x2']), mask_BB_dict['y1'] - mask_BB_dict['y2'],
                            mask_BB_dict['x1'] - mask_BB_dict['x2'], fill=False, edgecolor='white', linewidth=1)
    a.add_patch(rect_mask)

    # normalize the heatmap to [0,1], and add it with the image
    # mask = (mask-mask_min)/(mask_max - mask_min)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cam = 0.8 * heatmap + np.float32(img)
    b = fig.add_subplot(N_layer, 2, 2 * j + 2)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(cam)
    plt.title('{} weighted img'.format(name))
    rect_KP = plt.Rectangle((KP_BB_dict['y2'], KP_BB_dict['x2']), KP_BB_dict['y1'] - KP_BB_dict['y2'],
                            KP_BB_dict['x1'] - KP_BB_dict['x2'], fill=False, edgecolor='red', linewidth=1)
    b.add_patch(rect_KP)
    rect_mask = plt.Rectangle((mask_BB_dict['y2'], mask_BB_dict['x2']), mask_BB_dict['y1'] - mask_BB_dict['y2'],
                              mask_BB_dict['x1'] - mask_BB_dict['x2'], fill=False, edgecolor='white', linewidth=1)
    b.add_patch(rect_mask)


def save_mask_as_pdf(mask, img, img_dim, fig, N_layer, name, j):
    mask = cv2.resize(mask, (img_dim, img_dim))
    # print('\nmask:\n', mask)
    mask_max = np.max(mask)
    mask_min = np.min(mask)
    # Show the raw heatmap
    a = fig.add_subplot(N_layer, 2, 2 * j + 1)
    mask = (mask - mask_min) / (mask_max - mask_min)
    plt.imshow(np.uint8(255 * mask), cmap='jet')
    plt.title('{} activation map'.format(name))
    plt.colorbar(cmap='jet', fraction=0.046, pad=0.04)

    # normalize the heatmap to [0,1], and add it with the image
    # mask = (mask-mask_min)/(mask_max - mask_min)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cam = 0.8 * heatmap + np.float32(img)
    a = fig.add_subplot(N_layer, 2, 2 * j + 2)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(cam)
    plt.title('{} weighted img'.format(name))

def read_BB(dir):
    KP_BBs = np.load(dir + '.npy')
    bird_BB = np.load(dir + '_bird_BB.npy')
    return KP_BBs, bird_BB

def read_KP(dir):
    KPs = np.load(dir + '.npy')
    bird_BB = np.load(dir + '_bird_BB.npy')
    # print("KPs.shape:", KPs.shape)
    # print("bird_BB.shape:", bird_BB.shape)
    return KPs, bird_BB

def generate_mask_BB_mass_center(mask, bird_BB, scale=4):
    """
    :return   a dict {'x1', 'x2', 'y1', 'y2'}
    """
    bird_w = bird_BB[2] - bird_BB[0]
    bird_h = bird_BB[3] - bird_BB[1]
    mask_w = int(bird_w/scale)
    mask_h = int(bird_h/scale)

    center_x, center_y = ndimage.measurements.center_of_mass(mask)
    print("center_x:{}, center_y:{}".format(center_x, center_y))
    x1 = int(center_x - mask_w/2)
    x2 = int(center_x + mask_w/2)
    y1 = int(center_y - mask_h/2)
    y2 = int(center_y + mask_h/2)
    if x1 < 0:
        x1 = 0
    if x2 > 223:
        x2 = 223
    if y1 < 0:
        y1 = 0
    if y2 > 223:
        y2 = 223
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}


def generate_mask_BB_max_center(mask, bird_BB):
    """
    The attention bounding box is centerd by the max value in the attention map.
    :return   a dict {'x1', 'x2', 'y1', 'y2'}
    """
    bird_w = bird_BB[2] - bird_BB[0]
    bird_h = bird_BB[3] - bird_BB[1]

    mask_w = int(bird_w/4)
    mask_h = int(bird_h/4)

    max_x, max_y = np.where(mask == np.amax(mask))

    if len(max_x) > 1:
        max_id = int(len(max_x)/2)
        max_x = max_x[max_id]
        max_y = max_y[max_id]

    print(mask_w, mask_h, max_x, max_y)
    x1 = int(max_x - mask_w/2)
    x2 = int(max_x + mask_w/2)
    y1 = int(max_y - mask_h/2)
    y2 = int(max_y + mask_h/2)
    if x1 < 0:
        x1 = 0
    if x2 > 223:
        x2 = 223
    if y1 < 0:
        y1 = 0
    if y2 > 223:
        y2 = 223

    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

# def generate_mask_BB(mask, bird_BB, scale=4):
#     """
#     :return   a dict {'x1', 'x2', 'y1', 'y2'}
#     """
#     inter = 8
#     bird_w = bird_BB[2] - bird_BB[0]
#     bird_h = bird_BB[3] - bird_BB[1]
#     mask_w = int(bird_w / scale)
#     mask_h = int(bird_h / scale)
#
#     best_x1 = 0
#     best_y1 = 0
#
#     best_BB_sum = 0
#
#     for x in range(0, 224-mask_w, inter):
#         for y in range(0, 224-mask_h, inter):
#             BB_sum = np.sum(mask[x:x + mask_w, y:y + mask_h])
#             if BB_sum > best_BB_sum:
#                 best_BB_sum = BB_sum
#                 best_x1 = x
#                 best_y1 = y
#
#     mask_c_x = best_x1 + int(mask_w/2)
#     mask_c_y = best_y1 + int(mask_h/2)
#     return {'x1': best_x1, 'x2': best_x1 + mask_w, 'y1': best_y1, 'y2': best_y1 + mask_h}, (mask_c_x, mask_c_y)

def generate_mask_BB(mask, bird_BB, scale=4, KNOW_BIRD_BB=False):
    """
    :return   a dict {'x1', 'x2', 'y1', 'y2'}
    """
    bird_w = bird_BB[2] - bird_BB[0]
    bird_h = bird_BB[3] - bird_BB[1]
    mask_w = int(bird_w / scale)
    mask_h = int(bird_h / scale)
    # np. np.max(mask)
    (mask_c_x, mask_c_y) = np.unravel_index(np.argmax(mask), np.array(mask).shape)
    mask_BB = get_KP_BB((mask_c_x, mask_c_y), mask_h, mask_w, bird_BB, KNOW_BIRD_BB)
    return mask_BB, (mask_c_x, mask_c_y)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0 and iou <= 1.0
    return iou


def get_KP_BB(gt_point, mask_h, mask_w, bird_BB, KNOW_BIRD_BB=False):
    KP_best_x, KP_best_y = gt_point[0], gt_point[1]
    KP_x1 = KP_best_x - int(mask_w / 2)
    KP_x2 = KP_best_x + int(mask_w / 2)
    KP_y1 = KP_best_y - int(mask_h / 2)
    KP_y2 = KP_best_y + int(mask_h / 2)
    if KNOW_BIRD_BB:
        Bound = bird_BB
    else:
        Bound = [0, 0, 223, 223]
    if KP_x1 < Bound[0]:
        KP_x1, KP_x2 = Bound[0], Bound[0] + mask_w
    elif KP_x2 > Bound[2]:
        KP_x1, KP_x2 = Bound[2] - mask_w, Bound[2]
    if KP_y1 < Bound[1]:
        KP_y1, KP_y2 = Bound[1], Bound[1] + mask_h
    elif KP_y2 > Bound[3]:
        KP_y1, KP_y2 = Bound[3] - mask_h, Bound[3]
    return {'x1': KP_x1, 'x2': KP_x2, 'y1': KP_y1, 'y2': KP_y2}


def get_IoU_Image(idx, imgs, maps, save_dir, save_att_idx, names, groups,
                  attri_names, KPs, bird_BB, scale, resize_WH, KNOW_BIRD_BB):
    BB_parts = ['head', 'breast', 'belly', 'back', 'wing', 'tail', 'leg']
    # {'x1': KP_x1, 'x2': KP_x2, 'y1': KP_y1, 'y2': KP_y2}
    attri_to_group_dict = {
        'head': [1, 4, 5, 6, 9, 10],
        'breast': [3, 14],
        'belly': [2],
        'back': [0],
        'wing': [8, 12],
        'tail': [13],
        'leg': [7, 11],
    }

    bird_w = bird_BB[2] - bird_BB[0]
    bird_h = bird_BB[3] - bird_BB[1]
    mask_w = int(bird_w / scale)
    mask_h = int(bird_h / scale)

    if resize_WH:
        mask_h = max(int(mask_w / 2), mask_h)
        mask_w = max(int(mask_h / 2), mask_w)

    img_IoU = {}
    for group_name, group_dims in groups.items():
        # print("group_name, group_dims:", group_name, group_dims)
        if group_name == 'others':
            continue

        img_IoU[group_name] = []
        # if the body part exist:
        # generate the iou for each attention map from each subgroup
        for group_dim in group_dims:
            for j, name in enumerate(names):
                mask = maps[name][idx, group_dim, :, :]
                mask = cv2.resize(mask, (224, 224))
                mask_BB_dict, (mask_c_x, mask_c_y) = generate_mask_BB(mask, bird_BB, scale, KNOW_BIRD_BB)

                KP_idxs = attri_to_group_dict[group_name]
                KPs_sub = [KPs[KP_idx][:2] for KP_idx in KP_idxs if KPs[KP_idx][2] != 0]
                if len(KPs_sub) == 0:
                    continue
                dis = [(point[0] - mask_c_x) ** 2 + (point[1] - mask_c_y) ** 2 for point in KPs_sub]
                gt_point = KPs_sub[np.argmin(dis)]
                KP_BB_dict = get_KP_BB(gt_point, mask_h, mask_w, bird_BB, KNOW_BIRD_BB)
                # print("KP_BB_dict", KP_BB_dict, mask_BB_dict)
                IoU = get_iou(KP_BB_dict, mask_BB_dict)
                img_IoU[group_name].append(IoU)
    return img_IoU


def calculate_atten_IoU(input, impath, save_att_idx, maps, names, vis_groups, KP_root=None, save_att=False, scale=4,
                        resize_WH=False, KNOW_BIRD_BB=False):
    """
    :param input: input image
    :param impath: image paths
    :param maps: maps with size 64, 312, 7, 7
    :param names: layer names
    :param vis_groups: vis_groups is a list of size image_num. each item is a dict,
                       including the attention index for each subgroup
    :param KP_root: is the root of KP_centers
    :return:
    """
    # print("resize_WH:{}, out_of_edge:{}, max_area_center:{}".format(resize_WH, out_of_edge, max_area_center))
    img_raw_show = tensor_imshow(input)  # (64, 3, 224, 224)
    attri_names = refine_attri_names('./data/vis/files/attri_name.txt')
    batch_IoU = []
    for i, path in enumerate(impath):
        # time_1 = time.time()
        # print(i, path)
        if type(vis_groups) is dict:
            vis_group = vis_groups
        elif type(vis_groups) is list:
            vis_group = vis_groups[i]
        else:
            exit('ERROR FOR vis_groups!')
        tmp = path.split('/')[-2:]
        this_dir = os.path.join(KP_root, tmp[0], tmp[1][:-4])
        if save_att:
            save_dir = os.path.join(save_att, tmp[0], tmp[1][:-4])
        else:
            save_dir = False
        # KP_BBs, bird_BB = read_BB(this_dir)
        KPs, bird_BB = read_KP(this_dir)
        # print("this dir:", this_dir)
        # time_2 = time.time()
        # print('time for load BB:', time_2 - time_1)
        img_IoU = get_IoU_Image(i, img_raw_show, maps, save_dir, save_att_idx, names, vis_group, attri_names,
                                KPs, bird_BB, scale, resize_WH, KNOW_BIRD_BB)
        # time_3 = time.time()
        # print('time for calculate IoU:', time_3 - time_2)
        batch_IoU.append(img_IoU)
        # exit()
    return batch_IoU


def draw_attribute_activation(input, impath, maps, names, vis_groups, vis_root):
    img_raw_show = tensor_imshow(input)  # (64, 3, 224, 224)
    attri_names = refine_attri_names('../data/vis/files/attri_name.txt')

    for i, path in enumerate(impath):
        if type(vis_groups) == dict:
            vis_group = vis_groups
        elif type(vis_groups) == list:
            vis_group = vis_groups[i]
        else:
            exit('ERROR FOR vis_groups!')
        tmp = path.split('/')[-2:]
        this_dir = os.path.join(vis_root, tmp[0], tmp[1][:-4])
        show_maps_on_image(i, img_raw_show, maps, this_dir, names, vis_group, attri_names)


def show_maps_on_image(idx, imgs, maps, file_dir, names, groups, attri_names):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    img = np.transpose(imgs[idx, :, :, :], (1, 2, 0))
    img_dim = img.shape[0]
    N_attri = maps[names[0]].shape[1]
    N_layer = len(names)
    # Show the raw image
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4)
    plt.axis('off')
    # plt.imshow(np.uint8(255 * img))
    plt.imshow(img)
    plt.title('raw img')
    plt.savefig(os.path.join(file_dir, "raw.pdf"))

    fig = plt.figure(figsize=(8, 3 * N_layer))  # set the size for activation map
    for group_name, group_dims in groups.items():
        group_dir = os.path.join(file_dir, group_name)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        # 暂时写下这段代码进行观察，整体不和谐
        for j, name in enumerate(names):
            plt.clf()
            mask = np.mean(maps[name][idx, group_dims, :, :], axis=0)
            # print(mask)
            save_mask_as_pdf(mask, img, img_dim, fig, N_layer, name, j)
        # a higher dpi could let pdf clear, but need more storage.
        plt.savefig(os.path.join(group_dir, "Overall_{}.pdf".format(group_name)), dpi=100)

        for i in group_dims:
            plt.clf()
            for j, name in enumerate(names):
                mask = maps[name][idx, i, :, :]
                save_mask_as_pdf(mask, img, img_dim, fig, N_layer, name, j)
            # a higher dpi could let pdf clear, but need more storage.
            plt.savefig(os.path.join(group_dir, "attri_{:0>3}_{}.pdf".format(i, attri_names[i])), dpi=100)
    plt.close('all')


def show_cam_on_image(img, mask, file_path="cam.jpg", raw_path=None):
    # normalize the heatmap to [0,1]
    mask_max = np.max(mask)
    mask_min = np.min(mask)
    mask = (mask - mask_min)/(mask_max - mask_min)
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    img = np.transpose(img, (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)
    cv2.imwrite(file_path, np.uint8(255 * cam))  # grad-cam图像
    # print("saving images to:", file_path)

    if raw_path is not None:
        raw = np.float32(img)
        raw = raw / np.max(raw)
        cv2.imwrite(raw_path, np.uint8(255 * raw))  # raw图像


def show_image_pixels(img, mask, file_path="cam.jpg", raw_path=None, perc=0.3):
    # 为展示出效果, 将mask做初步处理
    # img is numpy array, shape [3, 224, 224]
    input_dim = img.shape[2]
    # print('img.shape:', input_dim)
    HW = input_dim * input_dim
    blur_img = np.float32(np.zeros(img.shape))
    start = blur_img
    finish = img
    # from large to small
    salient_order = np.flip(np.argsort(mask.reshape(-1, HW), axis=1), axis=-1)
    coords = salient_order[:, 0: int(HW * perc)]
    start = start.reshape(3, HW)
    start[:, coords] = finish.reshape(3, HW)[:, coords]
    start = start.reshape(3, input_dim, input_dim)
    start = np.transpose(start, (1, 2, 0))
    start = cv2.cvtColor(start, cv2.COLOR_RGB2BGR)
    start = np.float32(start)
    start = start / np.max(start)
    cv2.imwrite(file_path, np.uint8(255 * start))  # grad-cam图像

    if raw_path is not None:
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        raw = np.float32(img)
        raw = raw / np.max(raw)
        cv2.imwrite(raw_path, np.uint8(255 * raw))  # raw图像


def draw_imgs(opt, mask_path, vis_path, raw_path, img_raw_show, mask):
    # print('mask_path:', mask_path)
    # print('vis_path:', vis_path)
    ###### here control if we save the mask ######
    with open(mask_path, 'wb') as pickle_file:
        pickle.dump(mask, pickle_file)
    # if vis_path has something: save image; else: don't save image
    if vis_path:
        if opt.visimage:
            # 以 cv2.imwrite 的方式保存图片
            show_image_pixels(img_raw_show, mask, file_path=vis_path, raw_path=raw_path, perc=opt.visimage)
        else:
            # 以 plt.savefig 的方式保存图片
            show_cam_on_image_plt(img_raw_show, mask, file_path=vis_path, raw_path=raw_path)


def refine_attri_names(fn):
    attri_names = open(fn).readlines()
    assert len(attri_names) == 312
    for i in range(len(attri_names)):
        attri_names[i] = attri_names[i].strip().replace('::', '_').replace('(', '').replace(')', '')
    return attri_names


# def draw_attribute_actention(input, impath, maps, names, vis_groups, vis_root):
#     img_raw_show = tensor_imshow(input)  # (64, 3, 224, 224)
#     attri_names = refine_attri_names('../data/vis/files/attri_name.txt')
#     pre_attri, attention = maps
#
#     for i, path in enumerate(impath):
#         tmp = path.split('/')[-2:]
#         this_dir = os.path.join(vis_root, tmp[0], tmp[1][:-4])
#         show_maps_on_image(i, img_raw_show, maps, this_dir, names, vis_groups, attri_names)


def tensor_imshow(inp):
    """Imshow for Tensor."""
    # inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = inp.detach().squeeze().cpu().numpy()
    # print("inp.shape:", inp.shape)
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(3):
        inp[:, i] = inp[:, i] * std[i] + mean[i]
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    # plt.imshow(inp, **kwargs)
    # if title is not None:
    #     plt.title(title)
    # print(inp.shape)
    return inp


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # print('x.shape:', x, x.shape, x.requires_grad, x.is_leaf)
        for param in self.model.parameters():
            param.requires_grad = True

        print(self.model)
        # 可能需要根据模型做修改
        for name, module in self.model._modules.items():
            print("name:", name)
            print("module:", module)
            # print('x.shape:', name, x.requires_grad, x.is_leaf)
            x = x.requires_grad_()
            x = module(x)
            # if name in self.target_layers:
                # print('x.shape:', x.shape)
                # outputs += [x]
        exit('end!')
        return outputs, x  # 通过循环逐层跑模型，最后输出的是 提取到的变量 和 模型输出


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, args, model):
        """
        Let the code know which layer's activations need to be output
        :param args:
        :param model: the exact model for activations
        """
        # 此处需要传入参数
        self.model = model
        self.extractor_feature = FeatureExtractor(self.model, [args.extract])

    def __call__(self, x, attribute):
        # code here need to corresponding to the exact model
        feature_activations, x_middle = self.extractor_feature(x)
        x_middle = x_middle.view(-1, 2048)
        attri = self.model.model2(x_middle)
        output = attri.mm(attribute)
        return feature_activations, attri, output


def Global_max_pooling(matrix, kernel_size = 4):
    if kernel_size == None:
        return matrix
    else:
        w, h = matrix.shape
        matrix = matrix.reshape(w*h)
        idx = np.argsort(matrix)
        idx = idx[::-1]
        b = np.zeros(w*h)
        for i in range(kernel_size):
            b[idx[i]] = matrix[idx[i]]
        b = b.reshape(w, h)
        return b


class CAM:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.img_dim = self.args.img_size
        self.extractor = dict()
        for name in self.model.extract:
            self.extractor[name] = ModelOutputs(args, self.model.layers_ALE[name])

    def forward(self, input, attribute):
        return self.model(input, attribute)

    def __call__(self, x, attribute, kernel_size=None):
        record_features = {}
        batch_size = x.size(0)
        x = self.model.resnet[0:5](x)  # layer 1
        record_features['layer1'] = x  # [64, 256, 56, 56]
        x = self.model.resnet[5](x)  # layer 2
        record_features['layer2'] = x  # [64, 512, 28, 28]
        x = self.model.resnet[6](x)  # layer 3
        record_features['layer3'] = x  # [64, 1024, 14, 14]
        x = self.model.resnet[7](x)  # layer 4
        record_features['layer4'] = x  # [64, 2048, 7, 7]
        x = self.model.resnet[8](x)
        record_features['avg_pool'] = x  # [64, 2048, 1, 1]

        activation_layers = dict()
        output_layers = dict()
        attri_layers = dict()
        for name in self.model.extract:
            activation_layers[name], output_layers[name], attri_layers[name] \
                = self.extractor[name](record_features[name], attribute)
        #self.layers_ALE[name](record_features[name], attribute)
        # features, attri, output = self.extractor(input.cuda(), attribute)

        attri_dim = attribute.shape[0]

        for name, module in self.model.model2._modules.items():
            if name == 'fc':
                weight = module.weight.cpu().data.numpy()

        features = features[0].cpu().data.numpy()
        features = np.squeeze(features)
        cam_features = []
        for i in range(attri_dim):
            index = i
            mask = np.sum(weight[index, j]*features[j] for j in range(2048))
            mask = Global_max_pooling(mask, kernel_size=kernel_size)
            cam_features.append(cv2.resize(mask, (self.img_dim, self.img_dim)))  # 将（14,14）插值为（448, 448）
        return cam_features


