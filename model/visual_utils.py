#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import torch.utils.data
import os
from PIL import Image
import numpy as np
import h5py
import torch
import torch.utils.data
import scipy.io as sio
import matplotlib.pyplot as plt

def plot_acc(seen_classes_names, unseen_classes_names, class_name, acc, name, c, color=None):
    # Choose the dot per inch
    my_dpi = 96
    # Choose the dimensions for the figure (here 480x480)
    figure = plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)
    show_acc = []
    show_name = []
    show_color = []
    for i in range(len(acc)):
        if acc[i] > 0:
            show_acc.append(acc[i])
            show_name.append(name[i])
            if name[i] in seen_classes_names:
                show_color.append('blue')
            elif name[i] in unseen_classes_names:
                show_color.append('orange')

    height = show_acc
    # Choose the names of the bars
    bars = show_name

    y_pos = np.arange(len(bars))

    # Create bars
    plt.barh(y_pos, height, color=show_color)
    plt.xlim(0,1)
    # Create names on the x-axis
    plt.yticks(y_pos, bars, fontsize=15)

    # Save the figure and choose a name
    plt.savefig('{}.png'.format(c), bbox_inches='tight')
    # print("saving figure:", '{}.png'.format(c))
    # Show graphic
    # plt.show()

def sort_acc(acc, name):
    sorted_index = sorted(range(len(acc)), key=lambda k: acc[k], reverse=False)
    name_sort = []
    for i in range(len(name)):
        name_sort.append(name[sorted_index[i]])
    acc.sort()
    return acc, name_sort
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def L2_normalize(array):
    # L2 normalize
    norm = np.linalg.norm(array)
    array = array / norm
    return array

def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
                
    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)


    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # print("using the matcontent:", opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

        feature = matcontent['features'].T
        self.label = matcontent['labels'].astype(int).squeeze() - 1
        self.image_files = matcontent['image_files'].squeeze()
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        # print("matcontent.keys:", matcontent.keys())
        self.trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        if opt.dataset == 'CUB':
            self.train_loc = matcontent['train_loc'].squeeze() - 1
            self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            # self.train_unseen_loc = matcontent['train_unseen_loc'].squeeze() - 1

        # self.train_loc = matcontent['train_loc'].squeeze() - 1
        # self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        self.test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.allclasses_name = matcontent['allclasses_names']
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attri_name = matcontent['attri_name']

        # self.train_unseen_loc = matcontent['train_unseen_loc'].squeeze() - 1
        # self.test_unseen_small_loc = matcontent['test_unseen_small_loc'].squeeze() - 1
        # print("allclasses_names", self.allclasses_name)
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[self.trainval_loc])
                _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[self.test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[self.trainval_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[self.test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[self.test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[self.train_loc]).float()
            self.train_label = torch.from_numpy(self.label[self.train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[self.val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(self.label[self.val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        # print("self.test_unseen_label:", list(set(self.test_unseen_label.numpy())))
        # print("self.unseenclasses:", list(set(self.unseenclasses.numpy())))
        # print("self.test_seen_label:", list(set(self.test_seen_label.numpy())))
        # print("self.seenclasses:", list(set(self.seenclasses.numpy())))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(opt, image_files, img_loc, image_labels, dataset):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    for image_file, image_label in zip(image_files, image_labels):
        if dataset == 'CUB':
            image_file = opt.image_root + image_file[0].split("MSc/")[1]
        elif dataset == 'AWA1':
            image_file = opt.image_root + image_file[0].split("databases/")[1]
        elif dataset == 'AWA2':
            image_file = opt.image_root + '/AwA2/JPEGImages/' + image_file[0].split("JPEGImages")[1]
        elif dataset == 'SUN':
            image_file = os.path.join(opt.image_root, image_file[0].split("data/")[1])
        else:
            exit(1)
        imlist.append((image_file, int(image_label)))
    return imlist

class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, opt, data_inf=None, transform=None, target_transform=None, dataset=None,
                 flist_reader=default_flist_reader, loader=default_loader, image_type=None, select_num=None):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if image_type == 'test_unseen_small_loc':
            self.img_loc = data_inf.test_unseen_small_loc
        elif image_type == 'test_unseen_loc':
            self.img_loc = data_inf.test_unseen_loc
        elif image_type == 'test_seen_loc':
            self.img_loc = data_inf.test_seen_loc
        elif image_type == 'trainval_loc':
            self.img_loc = data_inf.trainval_loc
        elif image_type == 'train_loc':
            self.img_loc = data_inf.train_loc
        else:
            try:
                sys.exit(0)
            except:
                print("choose the image_type in ImageFileList")


        if select_num != None:
            # select_num is the number of images that we want to use
            # shuffle the image loc and choose #select_num images
            np.random.shuffle(self.img_loc)
            self.img_loc = self.img_loc[:select_num]

        self.image_files = data_inf.image_files
        self.image_labels = data_inf.label
        self.dataset = dataset
        self.imlist = flist_reader(opt, self.image_files, self.img_loc, self.image_labels, self.dataset)
        self.allclasses_name = data_inf.allclasses_name
        self.attri_name = data_inf.attri_name

        self.image_labels = self.image_labels[self.img_loc]
        label, idx = np.unique(self.image_labels, return_inverse=True)
        self.image_labels = torch.tensor(idx)
        # train_label = torch.tensor(train_label)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, impath

    def __len__(self):
        num = len(self.imlist)
        return num

def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = np.array(test_label)
    predicted_label = np.array(predicted_label)
    acc_per_class = []
    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in range(len(nclass)):
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class)/len(acc_per_class)


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
        acc_per_class = []

        acc = np.sum(test_label == predicted_label) / len(test_label)
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
        return acc, sum(acc_per_class)/len(acc_per_class)

def prepare_attri_label(attribute, classes):
    # print("attribute.shape", attribute.shape)
    classes_dim = classes.size(0)
    attri_dim = attribute.shape[1]
    output_attribute = torch.FloatTensor(classes_dim, attri_dim)
    for i in range(classes_dim):
        output_attribute[i] = attribute[classes[i]]
    return torch.transpose(output_attribute, 1, 0)


def save_correct_imgs(test_label, predicted_label, img_paths, img_locs):
    # print("label", test_label[2500:2900])
    # print("predict_label", predicted_label[2500:2900])
    test_label = np.array(test_label)
    print('len(test_label:', len(test_label))
    predicted_label = np.array(predicted_label)
    correct_idx  = [i for i in range(len(test_label)) if  test_label[i]== predicted_label[i]]
    acc = len(correct_idx) / len(test_label)
    print('correct_impaths', len(correct_idx))

    correct_impaths = [img_paths[correct_idx[i]] for i in range(len(correct_idx))]
    correct_imlocs = [img_locs[correct_idx[i]] for i in range(len(correct_idx))]
    print('correct_impaths', correct_impaths[0])
    print('correct_imlocs', len(correct_imlocs))

    print('overall acc:', acc)
    return correct_imlocs, correct_impaths


def save_predict_txt(output, predict_txt_path, attri, predicted_label, target, class_attribute, attri_names, class_names):
    file = open(predict_txt_path, 'w')
    output = output.cpu().numpy()[0]
    predicted_class = [class_names[index][0][0] for index in np.argsort(output)[::-1]]
    file.write('predicted classes:{}\n'.format(predicted_class))
    file.write('predict class:{}\n'.format(class_names[predicted_label][0][0]))
    file.write('ground truth class:{}\n'.format(class_names[target][0][0]))
    predict_att_mul = attri * class_attribute[predicted_label][0]
    # print("predict_att_mul:", predict_att_mul.shape)
    # print('attri_names:', attri_names.shape)
    target_att_mul = attri * class_attribute[target][0]
    predict_attri_idx = np.argsort(predict_att_mul)[::-1]
    target_attri_idx = np.argsort(target_att_mul)[::-1]
    file.write('predict attri_name for {}, '.format(class_names[predicted_label][0][0]) + 'the value is {}:\n \n'.format(sum(predict_att_mul)))
    for idx in predict_attri_idx:
        file.write('attri_name:{}, attri_mul_value:{:.4f}\n'.format(attri_names[idx].strip(), predict_att_mul[idx]))
    file.write('\n')
    file.write('predict attri_name for {}, '.format(class_names[target][0][0]) + 'the value is {}:\n \n'.format(sum(target_att_mul)))

    for idx in target_attri_idx:
        file.write('attri_name:{}, attri_mul_value:{:.4f}\n'.format(attri_names[idx].strip(), target_att_mul[idx]))
    file.write('\n')
    file.close()




def add_paths(opt, img_path, correct, predicted_label, attri_name, allclasses_name, attri_id=None, attri_rank=None, attri_weight = None, dataset = None):
    if dataset == "CUB":
        img_path = img_path.split('images/')[1]
    elif dataset == 'AWA1':
        img_path = img_path.split('JPEGImages/')[1]
    sub_path = img_path.split('/')[0]
    mask_path = os.path.join(opt.image_root, 'visual/{}/{}/masks/'.format(opt.vis_type, opt.train_id))
    mask_path = os.path.join(mask_path, sub_path)
    vis_path = mask_path.replace('masks', '{}_attri_images'.format(opt.image_type.strip('_loc')))
    final_vis_path = mask_path.replace('masks', '{}_all_images'.format(opt.image_type.strip('_loc')))
    raw_path = mask_path.replace('masks', 'raw_images')
    predict_txt_path = raw_path.replace('raw_images', 'predict_results')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(predict_txt_path):
        os.makedirs(predict_txt_path)
    if not os.path.exists(final_vis_path):
        os.makedirs(final_vis_path)
    final_mask_name = img_path.split('/')[1].replace('.jpg', '.pkl')
    final_mask_path = os.path.join(mask_path, final_mask_name)
    mask_name = final_mask_name.replace('.pkl', '_{}_{}_{}.pkl'.format(attri_rank, attri_id, attri_weight))
    mask_path = os.path.join(mask_path, mask_name)

    final_vis_name = img_path.split('/')[1].replace('.jpg', '_all.jpg')
    # print('attri:', attri_name[attri_id].strip(), len(attri_name[attri_id]))
    final_vis_path = os.path.join(final_vis_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + final_vis_name).replace('_{}.png'.format(attri_id), '_all.png')
    vis_name = img_path.split('/')[1].replace('.jpg', '_rank{}_{}_weight{:.3}.jpg'.format(attri_rank, attri_name[attri_id].strip(), attri_weight))
    vis_path = os.path.join(vis_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + vis_name)
    raw_path = os.path.join(raw_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + final_vis_name)
    predict_txt_path = os.path.join(predict_txt_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + final_vis_name.replace('.jpg', '.txt'))

    return predict_txt_path, mask_path, vis_path, raw_path, final_mask_path, final_vis_path

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def average(x):
    return x / np.sum(x, axis=0)


# def add_attri_selection_paths(opt, img_path, correct, predicted_label, attri_name, allclasses_name, attri_id=None, attri_rank=None, attri_weight = None, dataset = None):
#
#     if dataset == "CUB":
#         img_path = img_path.split('images/')[1]
#     elif dataset == 'AWA1':
#         img_path = img_path.split('JPEGImages/')[1]
#
#     mask_path = os.path.join(opt.imagelist, 'visual/{}/{}/masks/{}/'.format(opt.vis_type, opt.train_id, attri_name[attri_id].strip()))
#
#     vis_path = mask_path.replace('masks', '{}_attri_images'.format(opt.image_type.strip('_loc')))
#
#     final_vis_path = mask_path.replace('masks', '{}_all_images'.format(opt.image_type.strip('_loc')))
#     raw_path = mask_path.replace('masks', 'raw_images')
#
#     if not os.path.exists(mask_path):
#         os.makedirs(mask_path)
#     if not os.path.exists(vis_path):
#         os.makedirs(vis_path)
#     if not os.path.exists(raw_path):
#         os.makedirs(raw_path)
#     if not os.path.exists(final_vis_path):
#         os.makedirs(final_vis_path)
#     final_mask_name = img_path.split('/')[1].replace('.jpg', '.pkl')
#     final_mask_path = os.path.join(mask_path, final_mask_name)
#     # mask_name = final_mask_name.replace('.pkl', '_{}_{}.pkl'.format(attri_id, attri_weight))
#     mask_name = final_mask_path
#     mask_path = os.path.join(mask_path, mask_name)
#
#     final_vis_name = img_path.split('/')[1].replace('.jpg', '_all.jpg')
#     # print('attri:', attri_name[attri_id].strip(), len(attri_name[attri_id]))
#     final_vis_path = os.path.join(final_vis_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + final_vis_name).replace('_{}.png'.format(attri_id), '_all.png')
#     vis_name = img_path.split('/')[1].replace('.jpg', '_rank{}_{}_weight{:.3}.jpg'.format(attri_rank, attri_name[attri_id].strip(), attri_weight))
#     vis_path = os.path.join(vis_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + vis_name)
#     raw_path = os.path.join(raw_path, correct + "_" + allclasses_name[predicted_label][0][0] + "_" + final_vis_name)
#     # print("vis_path:", vis_path)
#     print("mask_path:", vis_path)
#     return mask_path, vis_path, raw_path, final_mask_path, final_vis_path


def per_dim_dis(A, B):
    """
    input: A, B with size of N*1
    purpose: calculate the per_dim distance of A and B
    :return: dis with same size as A
    """
    dis = np.abs(A - B)
    return dis


def add_image_attri_L2_path(opt, img_path, correct, predicted_label, attri_name, allclasses_name, attri_id=None, attri_rank=None, attri_weight=None, attri_dis=None, dataset = None):
    """ This path is divided by image class.
    Args:
        attri_id: the id of this attribute
        attri_rank: The rank of the attribute distance of this attri, 0 means the distance is smallest
        attri_weight: The distance

    Returns:
        The paths.
    """
    if dataset == "CUB":
        img_path = img_path.split('images/')[1]
    elif dataset == 'AWA1':
        img_path = img_path.split('JPEGImages/')[1]
    # print("img_path:", img_path)
    # sub_path = img_path.split('/')[0]
    # print("sub_path:", sub_path)

    mask_path = os.path.join(opt.image_root, 'visual/{}/{}/masks/{}'.format(opt.vis_type, opt.train_id, img_path.split('/')[0]))
    # print("mask_path:", mask_path)

    # mask_path = os.path.join(mask_path, sub_path)
    # print("mask_path:", mask_path)

    vis_path = mask_path.replace('masks', '{}_attri_images'.format(opt.image_type.strip('_loc')))
    # print("vis_path:", vis_path)

    final_vis_path = mask_path.replace('masks', '{}_all_images'.format(opt.image_type.strip('_loc')))
    raw_path = mask_path.replace('masks', 'raw_images')

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(final_vis_path):
        os.makedirs(final_vis_path)
    final_mask_name = img_path.split('/')[1].replace('.jpg', '.pkl')
    final_mask_path = os.path.join(mask_path, final_mask_name)
    mask_name = final_mask_name.replace('.pkl', '_{}_{}.pkl'.format(attri_id, attri_weight))
    mask_path = os.path.join(mask_path, mask_name)
    final_vis_name = img_path.split('/')[1].replace('.jpg', '_all.jpg')
    # print('attri:', attri_name[attri_id].strip(), len(attri_name[attri_id]))
    final_vis_path = os.path.join(final_vis_path, correct + "_" + allclasses_name[predicted_label] + "_" + final_vis_name).replace('_{}.png'.format(attri_id), '_all.png')
    vis_name = img_path.split('/')[1].replace('.jpg', '_rank{}_{}_dis{:.3}_weight{:.3}.jpg'.format(attri_rank, attri_name[attri_id].strip(), attri_dis, attri_weight))
    vis_path = os.path.join(vis_path, "{}_pred_{}_GT_{}". format(correct, allclasses_name[predicted_label], vis_name))
    raw_path = os.path.join(raw_path, "{}_pred_{}_GT_{}". format(correct, allclasses_name[predicted_label], final_vis_name))
    # print("vis_path:", vis_path)
    # print("final_vis_path:", final_vis_path)
    final_mask_path_pos = final_mask_path.replace('.pkl', '_pos.pkl')
    final_mask_path_neg = final_mask_path.replace('.pkl', '_neg.pkl')

    final_vis_path_pos = final_vis_path.replace('.jpg', '_pos.jpg')
    final_vis_path_neg = final_vis_path.replace('.jpg', '_neg.jpg')

    return mask_path, vis_path, raw_path, final_mask_path, final_vis_path, final_mask_path_pos, final_vis_path_pos, final_mask_path_neg, final_vis_path_neg

def get_group(fn):
    group_data = np.loadtxt(fn)
    num = int(max(group_data))
    groups = [[] for _ in range(num)]
    for i, id in enumerate(group_data):
        groups[int(id)-1].append(i)
    return groups

def add_glasso(var, group):
    return var[group, :].pow(2).sum(dim=0).add(1e-8).sum().pow(1/2.)

def add_dim_glasso(var, group):
    loss = var[group, :].pow(2).sum(dim=1).add(1e-8).pow(1/2.).sum()
    return loss