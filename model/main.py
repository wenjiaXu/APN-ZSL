from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import visual_utils
import sys
import random
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, \
    prepare_attri_label, add_glasso, add_dim_glasso
from model_proto import resnet_proto_IoU
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import json
from main_utils import test_zsl, calibrated_stacking, test_gzsl, \
    calculate_average_IoU, test_with_IoU
from main_utils import set_randomseed, get_loader, get_middle_graph, Loss_fn, Result
from opt import get_opt

cudnn.benchmark = True

opt = get_opt()
# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)



def main():
    # load data
    data = visual_utils.DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label  # weird

    # define test_classes
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")

    # prepare the attribute labels
    class_attribute = data.attribute
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()

    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    # define attribute groups

    if opt.dataset == 'CUB':
        parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8.json')))
        sub_group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8_layer.json')))
        opt.resnet_path = './pretrained_models/resnet101_c.pth.tar'
    elif opt.dataset == 'AWA2':
        parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat',
                 'character']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_9.json')))
        sub_group_dic = {}
        if opt.awa_finetune:
            opt.resnet_path = './pretrained_models/resnet101_awa2.pth.tar'
        else:
            opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'
    elif opt.dataset == 'SUN':
        parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_4.json')))
        sub_group_dic = {}
        opt.resnet_path = './pretrained_models/resnet101_sun.pth.tar'
    else:
        opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'
    # initialize model
    print('Create Model...')
    model = resnet_proto_IoU(opt)
    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()

    # optimzation weight, only ['final'] + model.extract are used.
    reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri, 'regular': opt.regular},
                  'layer4': {'l_xe': opt.l_xe, 'attri': opt.l_attri, 'regular': opt.l_regular,
                             'cpt': opt.cpt},  # l denotes layer
                  }
    reg_lambdas = {}
    for name in ['final'] + model.extract:
        reg_lambdas[name] = reg_weight[name]
    # print('reg_lambdas:', reg_lambdas)

    if torch.cuda.is_available():
        model.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_seen = attribute_seen.cuda()
        attribute_gzsl = attribute_gzsl.cuda()

    layer_name = model.extract[0]  # only use one layer currently
    # compact loss configuration, define middle_graph
    middle_graph = get_middle_graph(reg_weight[layer_name]['cpt'], model)

    # train and test
    result_zsl = Result()
    result_gzsl = Result()


    if opt.only_evaluate:
        print('Evaluate ...')
        model.load_state_dict(torch.load(opt.resume))
        model.eval()
        # test zsl
        if not opt.gzsl:
            acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
            print('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
        else:
            # test gzsl
            acc_GZSL_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
            acc_GZSL_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

            if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                acc_GZSL_H = 0
            else:
                acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
                        acc_GZSL_unseen + acc_GZSL_seen)

            print('GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'.format(acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H))
    else:
        print('Train and test...')
        for epoch in range(opt.nepoch):
            # print("training")
            model.train()
            current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
            realtrain = epoch > opt.pretrain_epoch
            if epoch <= opt.pretrain_epoch:   # pretrain ALE for the first several epoches
                optimizer = optim.Adam(params=[model.prototype_vectors[layer_name], model.ALE_vector],
                                       lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))
            else:
                optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=current_lr, betas=(opt.beta1, 0.999))
            # loss for print
            loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                        'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}

            batch = len(trainloader)
            for i, (batch_input, batch_target, impath) in enumerate(trainloader):
                model.zero_grad()
                # map target labels
                batch_target = visual_utils.map_label(batch_target, data.seenclasses)
                input_v = Variable(batch_input)
                label_v = Variable(batch_target)
                if opt.cuda:
                    input_v = input_v.cuda()
                    label_v = label_v.cuda()
                output, pre_attri, attention, pre_class = model(input_v, attribute_seen)
                label_a = attribute_seen[:, label_v].t()

                loss = Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,
                               output, pre_attri, attention, pre_class, label_a, label_v,
                               realtrain, middle_graph, parts, group_dic, sub_group_dic)
                loss_log['ave_loss'] += loss.item()
                loss.backward()
                optimizer.step()
            # print('\nLoss log: {}'.format({key: loss_log[key] / batch for key in loss_log}))
            print('\n[Epoch %d, Batch %5d] Train loss: %.3f '
                  % (epoch+1, batch, loss_log['ave_loss'] / batch))

            if (i + 1) == batch or (i + 1) % 200 == 0:
                ###### test #######
                # print("testing")
                model.eval()
                # test zsl
                if not opt.gzsl:
                    acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
                    if acc_ZSL > result_zsl.best_acc:
                        # save model state
                        model_save_path = os.path.join('./out/{}_ZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                        torch.save(model.state_dict(), model_save_path)
                        print('model saved to:', model_save_path)
                    result_zsl.update(epoch+1, acc_ZSL)
                    print('\n[Epoch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch+1, acc_ZSL, result_zsl.best_acc, result_zsl.best_iter))

                else:
                    # test gzsl
                    acc_GZSL_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
                    acc_GZSL_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

                    if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                        acc_GZSL_H = 0
                    else:
                        acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
                                acc_GZSL_unseen + acc_GZSL_seen)
                    if acc_GZSL_H > result_gzsl.best_acc:
                        # save model state
                        model_save_path = os.path.join('./out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                        torch.save(model.state_dict(), model_save_path)
                        print('model saved to:', model_save_path)

                    result_gzsl.update_gzsl(epoch+1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H)

                    print('\n[Epoch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                          '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                          format(epoch+1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H, result_gzsl.best_acc_U, result_gzsl.best_acc_S,
                    result_gzsl.best_acc, result_gzsl.best_iter))




if __name__ == '__main__':
    main()

