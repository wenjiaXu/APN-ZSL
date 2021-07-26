import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
import copy


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINEAR_SOFTMAX_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LINEAR_SOFTMAX_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class LINEAR_SOFTMAX(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LINEAR_SOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class LAYER_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LAYER_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        batch_size = x.size(0)
        x = torch.mean(x, dim=1)
        x = x.view(batch_size, -1)
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class resnet_proto_IoU(nn.Module):
    def __init__(self, opt):
        super(resnet_proto_IoU, self).__init__()
        resnet = models.resnet101()
        num_ftrs = resnet.fc.in_features
        num_fc_dic = {'cub':150, 'awa2': 40, 'sun': 645}

        if 'c' in opt.resnet_path:
            num_fc = num_fc_dic['cub']
        elif 'awa2' in opt.resnet_path:
            num_fc = num_fc_dic['awa2']
        elif 'sun' in opt.resnet_path:
            num_fc = num_fc_dic['sun']
        else:
            num_fc = 1000
        resnet.fc = nn.Linear(num_ftrs, num_fc)

        # 01 - load resnet to model1
        if opt.resnet_path != None:
            state_dict = torch.load(opt.resnet_path)
            resnet.load_state_dict(state_dict)
            # print("resnet load state dict from {}".format(opt.resnet_path))

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune(True)

        # 02 - load cls weights
        # we left the entry for several layers, but here we only use layer4
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 7*7, 'avg_pool': 1*1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        if opt.dataset == 'CUB':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [312, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([312, 2048, 1, 1]), requires_grad=True)
        elif opt.dataset == 'AWA1':
            exit(1)
            self.ALE = LINEAR_SOFTMAX_ALE(input_dim=self.channel_dict['avg_pool'], attri_dim=85)
        elif opt.dataset == 'AWA2':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [85, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
        elif opt.dataset == 'SUN':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [102, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([102, 2048, 1, 1]), requires_grad=True)
        self.avg_pool = opt.avg_pool

    def forward(self, x, attribute, return_map=False):
        """out: predict class, predict attributes, maps, out_feature"""
        # print('x.shape', x.shape)
        record_features = {}
        batch_size = x.size(0)
        x = self.resnet[0:5](x)  # layer 1
        record_features['layer1'] = x  # [64, 256, 56, 56]
        x = self.resnet[5](x)  # layer 2
        record_features['layer2'] = x  # [64, 512, 28, 28]
        x = self.resnet[6](x)  # layer 3
        record_features['layer3'] = x  # [64, 1024, 14, 14]
        x = self.resnet[7](x)  # layer 4
        record_features['layer4'] = x  # [64, 2048, 7, 7]

        attention = dict()
        pre_attri = dict()
        pre_class = dict()

        if self.avg_pool:
            pre_attri['final'] = F.avg_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        else:
            pre_attri['final'] = F.max_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        # print("pre_attri['final'].shape:", pre_attri['final'].shape)
        # print("attribute.shape:", attribute.shape)
        # exit()
        output_final = self.softmax(pre_attri['final'].mm(attribute))

        for name in self.extract:
            # print("hererererere:", record_features[name].shape)
            attention[name] = F.conv2d(input=record_features[name], weight=self.prototype_vectors[name])  # [64, 312, W, H]
            pre_attri[name] = F.max_pool2d(attention[name], kernel_size=self.kernel_size[name]).view(batch_size, -1)
            pre_class[name] = self.softmax(pre_attri[name].mm(attribute))
        return output_final, pre_attri, attention, pre_class

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def _l2_convolution(self, x, prototype_vector, one):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2  # [64, C, W, H]
        x2_patch_sum = F.conv2d(input=x2, weight=one)

        p2 = prototype_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vector)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast  [64, 312,  W, H]
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # [64, 312,  W, H]
        return distances

