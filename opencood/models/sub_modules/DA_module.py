
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import pdb


class DA_feature_Discriminator(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature

        """
        super(DA_feature_Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 10, 4, 2, 1)
        # self.bn2 = nn.BatchNorm2d(1)
        # self.conv3 = nn.Conv2d(128, 1, 4, 2, 1)
        self.fc1 = nn.Linear(10 * 12 * 44, 1)
        # self.fc2 = nn.Linear(8 * 6 * 22, 1)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.sig = nn.Sigmoid()

        for l in [self.conv1, self.conv2, self.fc1]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        # x = self.bn2(x)
        # x = self.relu(self.conv3(x))
        x = x.contiguous().view(-1, 12 * 44*10)
        x = self.fc1(x)
        x = self.sig(x)
        return x

class DA_feature_Discriminator_V1(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature

        """
        super(DA_feature_Discriminator_V1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 10, 4, 2, 1)
        # self.bn2 = nn.BatchNorm2d(1)
        # self.conv3 = nn.Conv2d(128, 1, 4, 2, 1)
        self.fc1 = nn.Linear(10 * 12 * 44, 5*6*22)
        self.fc2 = nn.Linear(5*6*22, 1)
        # self.fc3 = nn.Linear(5*11, 1)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.sig = nn.Sigmoid()

        for l in [self.conv1, self.conv2, self.fc1]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        # x = self.bn2(x)
        # x = self.relu(self.conv3(x))
        x = x.contiguous().view(-1, 12 * 44*10)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.sig(x)
        return x

class DA_feature_Discriminator_V2(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature

        """
        super(DA_feature_Discriminator_V2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 4, 2, 1)#[24, 88]
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 4, 2, 1)#[12, 44]
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 16, 4, 2, 1)#[6, 22]
        self.bn3 = nn.BatchNorm2d(16)# [bs, 16, 6,22]

        # self.fc1 = nn.Linear(10 * 12 * 44, 1)
        # self.fc2 = nn.Linear(8 * 6 * 22, 1)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        # self.sig = nn.Sigmoid()

        for l in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)

        return x


class DA_feature_Discriminator_V3(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature

        """
        super(DA_feature_Discriminator_V3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 512, 4, 2, 1)#[24, 88]
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 4, 2, 1)#[12, 44]
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 64, 4, 2, 1)#[6, 22]
        self.bn3 = nn.BatchNorm2d(64)# [bs, 16, 6,22]

        # self.fc1 = nn.Linear(10 * 12 * 44, 1)
        # self.fc2 = nn.Linear(8 * 6 * 22, 1)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        # self.sig = nn.Sigmoid()

        for l in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)

        return x

class DA_feature_Discriminator_V4(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature

        """
        super(DA_feature_Discriminator_V4, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 512, 4, 2, 1)#[24, 88]
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 4, 2, 1)#[12, 44]
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, 4, 2, 1)#[6, 22]
        self.bn3 = nn.BatchNorm2d(128)# [bs, 16, 6,22]

        # self.fc1 = nn.Linear(10 * 12 * 44, 1)
        # self.fc2 = nn.Linear(8 * 6 * 22, 1)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        # self.sig = nn.Sigmoid()

        for l in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)

        return x

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    # def __repr__(self):
    #     tmpstr = self.__class__.__name__ + "("
    #     tmpstr += "weight=" + str(self.weight)
    #     tmpstr += ")"
    #     return tmpstr



class DA_feature_Head(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature

        """
        super(DA_feature_Head, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, feature):

        t = F.relu(self.conv1_da(feature))
        img_features=self.conv2_da(t)
        return img_features



class DA_instance_Head(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DA_instance_Head, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)
        self.in_channels=in_channels

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(nn.Module):
    
    def __init__(self, args):
        super(DomainAdaptationModule, self).__init__()


        self.feature_head = DA_feature_Head(args['DA_feature_head'])
        self.instance_head = DA_instance_Head(args['DA_instance_head'])

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fea_weight = args['DA_feature_weight'] 
        self.ins_weight = args['DA_instance_weight'] 
        self.args = args
        if args['AdvGRL']:
            self.bce = F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.7,0.3]]), torch.FloatTensor([[1,0]]))
            self.advGRL_threshold = args['advGRL_threshold']
            print("AdvGRL Training!!!!!!")
    
        self.grl_feature = GradientScalarLayer(-1.0*args['grl_feature_weight']) 
        self.grl_instance = GradientScalarLayer(-1.0*args['grl_instance_weight']) 

        self.DA_feature_Discriminator = DA_feature_Discriminator(args['DA_feature_head'])

    def ADDA_classifier(self, output):

        # source_fea = source_output['fused_feature']
        # target_fea = target_output['fused_feature']
        fea = output['fused_feature']

        return  self.DA_feature_Discriminator(fea)


    def __call__(self, output_dict):
        # source_fea: [B, C, H, W] [B, 384, 48, 176]
        # target_fea: [B, C, H, W] [B, 384, 48, 176]
        # source_ins: [B, C, H, W][B, 2, 48, 176]
        # target_ins: [B, C, H, W][B, 2, 48, 176]

        source_fea = output_dict['source_feature']
        target_fea = output_dict['target_feature']

        source_psm = output_dict['psm']
        target_psm = output_dict['target_psm']

        # source_rm = output_dict['rm']
        # target_rm = output_dict['target_rm']
        # pdb.set_trace()

        ################################## Loss of DA feature component#################


        ####1) GRL component
        if not self.args['AdvGRL']:
            source_grl_feature = self.grl_feature(source_fea)
            target_grl_feature = self.grl_feature(target_fea)
        else:
            source_grl_feature = source_fea
            target_grl_feature = target_fea


        da_source_feature = self.feature_head(source_grl_feature)
        da_target_feature = self.feature_head(target_grl_feature)

        da_source_fea_label = torch.ones_like(da_source_feature, dtype=torch.float32)#### source label is 1
        da_target_fea_label = torch.zeros_like(da_target_feature, dtype=torch.float32)#### target label is 0
        

        da_source_fea_level = da_source_feature.reshape(da_source_feature.shape[0], -1) #[B, C*H*W]
        da_target_fea_level = da_target_feature.reshape(da_target_feature.shape[0], -1) #[B, C*H*W]

        da_source_fea_label = da_source_fea_label.reshape(da_source_fea_label.shape[0], -1)##[B, C*H*W]
        da_target_fea_label = da_target_fea_label.reshape(da_target_fea_label.shape[0], -1) #[B, C*H*W]

        da_fea = torch.cat([da_source_fea_level, da_target_fea_level], dim=0) #[B*2, C*H*W]
        da_fea_label = torch.cat([da_source_fea_label, da_target_fea_label], dim=0) #[B*2, C*H*W]


        ####2) AdvGRL component
        if self.args['AdvGRL']:
            with torch.no_grad():
                da_fea_loss = F.binary_cross_entropy_with_logits(da_fea, da_fea_label) 
            if da_fea_loss <= self.bce:
                adv_threshold = min(self.advGRL_threshold, 1/da_fea_loss)
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.args['grl_feature_weight']*adv_threshold.numpy())
                da_fea = self.advGRL_optimized(da_fea)
                da_fea_loss = F.binary_cross_entropy_with_logits(da_fea, da_fea_label) 
            else:
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.args['grl_feature_weight'])
                da_fea = self.advGRL_optimized(da_fea)
                da_fea_loss = F.binary_cross_entropy_with_logits(da_fea, da_fea_label) 

        else:
            # da feature loss
            da_fea_loss = F.binary_cross_entropy_with_logits(da_fea, da_fea_label) 





        ###############################Loss of DA instance component#############################



        ####1) GRL component
        if not self.args['AdvGRL']:
            source_psm_grl = self.grl_instance(source_psm)
            target_psm_grl = self.grl_instance(target_psm)
        else:
            source_psm_grl = source_psm
            target_psm_grl = target_psm


        # source_psm_grl = self.grl_instance(source_rm)
        # target_psm_grl = self.grl_instance(target_rm)

        cls_preds_source = source_psm_grl.permute(0, 2, 3, 1).contiguous() ### refer to PointPillarLoss------->[B, 48, 176, 2]
        cls_preds_target = target_psm_grl.permute(0, 2, 3, 1).contiguous()###refer to  PointPillarLoss------->[B, 48, 176, 2]
        # print("cls_preds_source: ", cls_preds_source.size())
        cls_preds_source = self.avgpool(cls_preds_source)
        cls_preds_target = self.avgpool(cls_preds_target)##[B, 48, 88, 1]

        cls_preds_source = cls_preds_source.view(source_psm.shape[0]*source_psm.shape[2], -1)##[B*H, C *W]====[B*H, 88*1]
        cls_preds_target = cls_preds_target.view(target_psm.shape[0]*source_psm.shape[2], -1)##[B*H, C *W]====[B*H, 88*1]

        da_ins_source  = self.instance_head(cls_preds_source)###[B*H, 2* 88]----> [B*H, 1]
        da_ins_target = self.instance_head(cls_preds_target)###[B*H, 2* 88]----> [B*H, 1]


        da_source_ins_label = torch.ones_like(da_ins_source, dtype=torch.float32)#### source label is 1
        da_target_ins_label = torch.zeros_like(da_ins_target, dtype=torch.float32)#### target label is 0


        da_ins = torch.cat([da_ins_source, da_ins_target],  dim=0)###[2*B*H, 1]
        da_ins_label = torch.cat([da_source_ins_label, da_target_ins_label], dim=0) ###[2*B*H, 1]

        # # da instance loss
        # da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label) 
        
        ####2) AdvGRL component
        if self.args['AdvGRL']:
            with torch.no_grad():
                da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label) 
            if da_fea_loss <= self.bce:
                adv_threshold = min(self.advGRL_threshold, 1/da_fea_loss)
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.args['grl_instance_weight']*adv_threshold.numpy())
                da_ins = self.advGRL_optimized(da_ins)
                da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label) 
            else:
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.args['grl_instance_weight'])
                da_ins = self.advGRL_optimized(da_ins)
                da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label) 
        else:
            # da instance loss
            da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label) 

        losses = {}
        if self.fea_weight > 0:
            losses['fea_loss'] = da_fea_loss * self.fea_weight
        if self.ins_weight > 0:
            losses['ins_loss'] = da_ins_loss * self.ins_weight

        return losses



