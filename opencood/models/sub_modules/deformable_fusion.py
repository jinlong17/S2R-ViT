"""
This class is about swap fusion applications
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.base_transformer import FeedForward, PreNormResidual

from opencood.models.sub_modules.self_attn import ScaledDotProductAttention

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat

import pdb
import math


# KPN基本网路单元
class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False,bn=True):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if bn :
        #    self.conv1 = nn.Sequential(
        #             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
        #             nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
        #             nn.ReLU(),
        #             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
        #             nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
        #             nn.ReLU(),
        #             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
        #             nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
        #             nn.ReLU()
        #         )
           self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
                    nn.ReLU()
                    # nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
                    # nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
                    # nn.ReLU(),
                    # nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
                    # nn.BatchNorm2d(out_ch,eps=1e-5, momentum=0.01, affine=True),
                    # nn.ReLU()
                )

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, color, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, color, height, width)

            core_out[K] = torch.einsum('ijklnop,ijlmnop->ijkmnop', [t1, t2]).view(batch_size, N, K * K, 1, height, width)

            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        # pdb.set_trace()
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        # for key, data in core.items():
        #     print(key, data.size())
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            torch.cuda.empty_cache()
            if len(img_stack) == 0:
                frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
                for i in range(K):
                    for j in range(K):
                        img_stack.append(frame_pad[..., i:i + height, j:j + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1]**2 - kernel[index]**2) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias

        if color == 1:
            pred_img_i = pred_img_i.squeeze(2)

        try:
            while len(pred_img_i.size()) > len(white_level.size()):
                white_level = white_level.unsqueeze(-1)
            white_level = white_level.type_as(pred_img_i).expand_as(pred_img_i)
        except:
            pass

        pred_img_i = pred_img_i / white_level
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
        # print('pred_img:', pred_img.size())
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        # return pred_img_i, pred_img
        return pred_img_i 
class KPN(nn.Module):
    def __init__(self,in_channel=3, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias

        self.color_channel = 3 if color else 1

        self.in_channel = in_channel
        self.out_channel = in_channel


        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # pdb.set_trace()

        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(self.in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, self.out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # pdb.set_trace()
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5,size=conv4.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6,size=conv3.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7,size=conv2.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        core = self.outc(F.interpolate(conv8, size=data.size()[-2:], mode=self.upMode, align_corners=True))
        # pdb.set_trace()

        # pdb.set_trace()
        # return self.kernel_pred(data, core, white_level)
        return data*core

class KPN_S(nn.Module):
    def __init__(self,in_channel=3, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN_S, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias

        self.color_channel = 3 if color else 1

        self.in_channel = in_channel
        self.out_channel = in_channel


        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # pdb.set_trace()

        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(self.in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, self.out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # pdb.set_trace()
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5,size=conv4.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6,size=conv3.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7,size=conv2.size()[-2:],  mode=self.upMode, align_corners=True)], dim=1))
        core = self.outc(F.interpolate(conv8, size=data.size()[-2:], mode=self.upMode, align_corners=True))
        # pdb.set_trace()

        # pdb.set_trace()
        # return self.kernel_pred(data, core, white_level)
        # return data*core
        return core



class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        # print("     ",in_channels,kernel_size, stride, padding, dilation)
        # print(self.depthwise)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW): 
        flops = 0
        flops += HW*self.in_channels*self.kernel_size**2/self.stride**2
        flops += HW*self.in_channels*self.out_channels
        print("SeqConv2d:{%.2f}"%(flops/1e9))
        return flops
        
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        # print("dim, inner_dim, kernel_size, q_stride, pad, bias   ", dim, inner_dim, kernel_size, q_stride, pad, bias)
        # self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        # self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        # self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        # pdb.set_trace()
        q = self.to_q(x)
        # q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        q = rearrange(q, 'b c l w -> b (l w) c')
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        # k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        # v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        k = rearrange(k, 'b c l w -> b (l w) c')
        v = rearrange(v, 'b c l w -> b (l w) c')
        return q,k,v    
    
    def flops(self, q_L, kv_L=None): 
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

    def flops(self, q_L, kv_L=None): 
        kv_L = kv_L or q_L
        flops = q_L*self.dim*self.inner_dim+kv_L*self.dim*self.inner_dim*2
        return flops 



class DW_Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads=32, attn_drop=0.,agent_size=6, window_size=7, shift_size=0 , act_layer=0,
                 qkv_bias=True, qk_scale=None, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww

        self.n_group = len(self.window_size)
        self.channel = self.dim // self.n_group  
        assert self.dim == self.channel * self.n_group
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.gnum_heads = num_heads // len(self.window_size)  
        assert num_heads == self.gnum_heads * len(self.window_size)
        self.gchannel = self.channel // self.gnum_heads  
        assert self.channel == self.gchannel * self.gnum_heads
        self.qk_scale = qk_scale

        # define a parameter table of relative position bias
        # print(self.window_size)

        self.relative_position_bias_table = []
        self.relative_position_index = []
        for i, window_s in enumerate(self.window_size):
            relative_position_bias_params = nn.Parameter(
                torch.zeros((2 * window_s - 1) * (2 * window_s - 1), self.gnum_heads))
            trunc_normal_(relative_position_bias_params, std=.02)
            setattr(self,'relative_position_bias_params_{}'.format(i),relative_position_bias_params) 
            self.relative_position_bias_table.append(getattr(self,'relative_position_bias_params_{}'.format(i)))

            # get pair-wise relative position index for each token inside the window
            Window_size = to_2tuple(window_s)
            coords_h = torch.arange(Window_size[0]) #[0,1,2,3,..,window_s]
            coords_w = torch.arange(Window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += Window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += Window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * Window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # print(relative_position_index.shape)
            self.register_buffer("relative_position_index_{}".format(i), relative_position_index)
            self.relative_position_index.append(getattr(self, "relative_position_index_{}".format(i)))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        # self.sknet = SKConv(dim=dim, M=self.n_group, act_layer=act_layer)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print('----------',self.window_size)
        # exit()
        # print(x.shape) #torch.Size([64, 56, 56, 96])
        # B, H, W, C = x.shape
        # x = x.view(B, -1, C)  # [B,H*W,C]



        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device \
            = *x.shape, x.device
        
        W = width*window_width
        H = height*window_height
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')


        # print(x.shape) #torch.Size([64, 3136, 96])
        B, HW, C = x.shape
        qkv = self.qkv(x).reshape(B, HW, 3, C).permute(2, 0, 1, 3)  # [3,B,HW,C]
        # print(qkv.shape) #torch.Size([3, 64, 3136, 96])
        qkv = qkv.reshape(3, agent_size*batch, height*window_height, width*window_width, C)

        qkv_groups = qkv.chunk(len(self.window_size), -1)  
        x_groups = []
        for i, qkv_group in enumerate(qkv_groups):
            # print(qkv_group.shape) #torch.Size([3, 42, 56, 56, 32])
            window_s = self.window_size[i]
            # print(i)
            # print("self.window_size[i]: ", self.window_size[i])
            # print(q.shape) #[64, 56, 56, 32]
            # cyclic shift

            # padding
            pad_l = pad_t = 0
            pad_r = (window_s - W % window_s) % window_s
            pad_b = (window_s - H % window_s) % window_s
            qkv_group = F.pad(qkv_group, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp, _ = qkv_group.shape

            # cyclic shift
            # if self.shift_size[i] > 0:
            #     shifted_qkv_group = torch.roll(qkv_group, shifts=(-self.shift_size[i], -self.shift_size[i]),
            #                                    dims=(2, 3))
            # else:
            #     shifted_qkv_group = qkv_group
            shifted_qkv_group = qkv_group
            # print(shifted_qkv_group.shape) #[3, 64, 56, 56, 32]

            # partition windows
            qkv_windows = window_partition(shifted_qkv_group, window_s)  # nW*B, window_size, window_size, C
            # print(qkv_windows.shape) #torch.Size([3, 4096, 7, 7, 32])
            qkv_windows = qkv_windows.view(3, -1, window_s * window_s,
                                           self.channel)  # nW*B, window_size*window_size, C//n_group
            _, B_, N, _ = qkv_windows.shape  # [3, 9216, 25, 32]
  
            # pad_c = (self.num_heads - self.channel % self.num_heads) % self.num_heads
            qkv = qkv_windows.reshape(3, B_, N, self.gnum_heads, self.gchannel).permute(0, 1, 3, 2,
                                                                                        4)  # [3,B_,self.gnum_heads,N,self.gchannel]

            head_dim = qkv.shape[-1]
            [q, k, v] = [x for x in qkv]
            # print(q.shape) #torch.Size([B_, self.gnum_heads, N, self.gchannel])
            self.scale = self.qk_scale or head_dim ** -0.5
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            # print(attn.shape) #torch.Size([4608, 1, 25, 25])
            window_size = to_2tuple(window_s)
            # print(window_size, self.relative_position_index[i].view(-1).shape,self.relative_position_bias_table[i][self.relative_position_index[i].view(-1)].shape)
            relative_position_bias = self.relative_position_bias_table[i][
                self.relative_position_index[i].view(-1)].view(
                window_size[0] * window_size[1], window_size[0] * window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            # print(relative_position_bias.shape) #torch.Size([25, 25, 3])
            # exit()
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().cuda()  # nH, Wh*Ww, Wh*Ww

            # print(attn.shape, relative_position_bias.shape) #torch.Size([9216, gnum_heads, 25, 25]) torch.Size([gnum_heads, 25, 25])
            attn = attn + relative_position_bias.unsqueeze(0)  ##torch.Size([4608, gnum_heads, N, N])

            # if mask[i] is not None:
            #     nW = mask[i].shape[0]
            #     attn = attn.view(B_ // nW, nW, self.gnum_heads, N, N) + mask[i].unsqueeze(1).unsqueeze(0)
            #     attn = attn.view(-1, self.gnum_heads, N, N)
            #     attn = self.softmax(attn)
            # else:
            #     attn = self.softmax(attn)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            # print(attn.shape, v.shape)
            # print((attn @ v).transpose(1, 2).shape, B_, N, self.gchannel*self.gnum_heads)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, self.channel)

            # merge windows
            x = x.view(B_, window_s, window_s, self.channel)
            shifted_x = window_reverse(x, window_s, Hp, Wp)  # B H W C

            # reverse cyclic shift
            # if self.shift_size[i] > 0:
            #     x = torch.roll(shifted_x, shifts=(self.shift_size[i], self.shift_size[i]), dims=(1, 2))
            # else:
            #     x = shifted_x
            x = shifted_x
            # print(x.shape) #torch.Size([64, 72, 72, 32])
            # x = x.reshape(B, -1, self.channel)
            # pdb.set_trace()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x_groups.append(x)

        x = torch.cat(x_groups, -1)  # [B,H,W,C]
        # x = self.sknet(x) #[B,C,H,W]
        x = x.view(B, self.dim, HW).permute(0, 2, 1) #[B,HW,C]


        x = rearrange(x, '(b x y) (l w1 w2) d -> b l x y w1 w2 d',
                                b=batch, x=height, y=width,l=agent_size, w1=window_height, w2=window_width)

        
        return x


class Conv_Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7,
            token_projection='linear',
            qkv_bias=True
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        # self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        if token_projection =='conv':
            self.to_qkv = ConvProjection(dim,dim_head,dim//dim_head,bias=qkv_bias)
        elif token_projection =='linear':
            self.to_qkv = LinearProjection(dim,dim_head,dim//dim_head,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        


        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # flatten
        # pdb.set_trace()
        # x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values
        # q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y l) ( w1 w2) d')
        q, k, v = self.to_qkv(x)
        q, k, v = map(lambda t: rearrange(t, '(b  l) w d -> b  (l w) d', l=agent_size ),
                      (q, k, v))
        # pdb.set_trace()

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))
        
        # pdb.set_trace()

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if len(x.shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    elif len(x.shape) == 5:
        _, B, H, W, C = x.shape
        # print(x.shape, window_size) #torch.Size([3, 42, 56, 56, 32])
        x = x.view(3, B, H // window_size, window_size, W // window_size, window_size, C) 
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(3, -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class DW_Attention_M(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads=32, attn_drop=0.,agent_size=6, window_size=7, shift_size=0 , act_layer=0,
                 qkv_bias=True, qk_scale=None, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww

        self.n_group = len(self.window_size)
        self.channel = self.dim // self.n_group  
        assert self.dim == self.channel * self.n_group
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.gnum_heads = num_heads // len(self.window_size)  
        assert num_heads == self.gnum_heads * len(self.window_size)
        self.gchannel = self.channel // self.gnum_heads  
        assert self.channel == self.gchannel * self.gnum_heads
        self.qk_scale = qk_scale

        # define a parameter table of relative position bias
        # print(self.window_size)

        self.relative_position_bias_table = []
        self.relative_position_index = []

        self.att = ScaledDotProductAttention(self.dim)


        for i, window_s in enumerate(self.window_size):
            relative_position_bias_params = nn.Parameter(
                torch.zeros((2 * window_s - 1) * (2 * window_s - 1), self.gnum_heads))
            trunc_normal_(relative_position_bias_params, std=.02)
            setattr(self,'relative_position_bias_params_{}'.format(i),relative_position_bias_params) 
            self.relative_position_bias_table.append(getattr(self,'relative_position_bias_params_{}'.format(i)))

            # get pair-wise relative position index for each token inside the window
            Window_size = to_2tuple(window_s)
            coords_h = torch.arange(Window_size[0]) #[0,1,2,3,..,window_s]
            coords_w = torch.arange(Window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += Window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += Window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * Window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # print(relative_position_index.shape)
            self.register_buffer("relative_position_index_{}".format(i), relative_position_index)
            self.relative_position_index.append(getattr(self, "relative_position_index_{}".format(i)))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        # self.sknet = SKConv(dim=dim, M=self.n_group, act_layer=act_layer)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print('----------',self.window_size)
        # exit()
        # print(x.shape) #torch.Size([64, 56, 56, 96])
        # B, H, W, C = x.shape
        # x = x.view(B, -1, C)  # [B,H*W,C]



        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device \
            = *x.shape, x.device
        
        W = width*window_width
        H = height*window_height
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')


        # print(x.shape) #torch.Size([64, 3136, 96])
        B, HW, C = x.shape
        qkv = self.qkv(x).reshape(B, HW, 3, C).permute(2, 0, 1, 3)  # [3,B,HW,C]
        # print(qkv.shape) #torch.Size([3, 64, 3136, 96])
        qkv = qkv.reshape(3, agent_size*batch, height*window_height, width*window_width, C)

        qkv_groups = qkv.chunk(len(self.window_size), -1)  
        x_groups = []
        for i, qkv_group in enumerate(qkv_groups):
            # print(qkv_group.shape) #torch.Size([3, 42, 56, 56, 32])
            window_s = self.window_size[i]
            # print(i)
            # print("self.window_size[i]: ", self.window_size[i])
            # print(q.shape) #[64, 56, 56, 32]
            # cyclic shift

            # padding
            pad_l = pad_t = 0
            pad_r = (window_s - W % window_s) % window_s
            pad_b = (window_s - H % window_s) % window_s
            qkv_group = F.pad(qkv_group, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp, _ = qkv_group.shape

            # cyclic shift
            # if self.shift_size[i] > 0:
            #     shifted_qkv_group = torch.roll(qkv_group, shifts=(-self.shift_size[i], -self.shift_size[i]),
            #                                    dims=(2, 3))
            # else:
            #     shifted_qkv_group = qkv_group
            shifted_qkv_group = qkv_group
            # print(shifted_qkv_group.shape) #[3, 64, 56, 56, 32]

            # partition windows
            qkv_windows = window_partition(shifted_qkv_group, window_s)  # nW*B, window_size, window_size, C
            # print(qkv_windows.shape) #torch.Size([3, 4096, 7, 7, 32])
            qkv_windows = qkv_windows.view(3, -1, window_s * window_s,
                                           self.channel)  # nW*B, window_size*window_size, C//n_group
            _, B_, N, _ = qkv_windows.shape  # [3, 9216, 25, 32]
  
            # pad_c = (self.num_heads - self.channel % self.num_heads) % self.num_heads
            qkv = qkv_windows.reshape(3, B_, N, self.gnum_heads, self.gchannel).permute(0, 1, 3, 2,
                                                                                        4)  # [3,B_,self.gnum_heads,N,self.gchannel]

            head_dim = qkv.shape[-1]
            [q, k, v] = [x for x in qkv]
            # print(q.shape) #torch.Size([B_, self.gnum_heads, N, self.gchannel])
            self.scale = self.qk_scale or head_dim ** -0.5
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            # print(attn.shape) #torch.Size([4608, 1, 25, 25])
            window_size = to_2tuple(window_s)
            # print(window_size, self.relative_position_index[i].view(-1).shape,self.relative_position_bias_table[i][self.relative_position_index[i].view(-1)].shape)
            relative_position_bias = self.relative_position_bias_table[i][
                self.relative_position_index[i].view(-1)].view(
                window_size[0] * window_size[1], window_size[0] * window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            # print(relative_position_bias.shape) #torch.Size([25, 25, 3])
            # exit()
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().cuda()  # nH, Wh*Ww, Wh*Ww

            # print(attn.shape, relative_position_bias.shape) #torch.Size([9216, gnum_heads, 25, 25]) torch.Size([gnum_heads, 25, 25])
            attn = attn + relative_position_bias.unsqueeze(0)  ##torch.Size([4608, gnum_heads, N, N])

            # if mask[i] is not None:
            #     nW = mask[i].shape[0]
            #     attn = attn.view(B_ // nW, nW, self.gnum_heads, N, N) + mask[i].unsqueeze(1).unsqueeze(0)
            #     attn = attn.view(-1, self.gnum_heads, N, N)
            #     attn = self.softmax(attn)
            # else:
            #     attn = self.softmax(attn)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            # print(attn.shape, v.shape)
            # print((attn @ v).transpose(1, 2).shape, B_, N, self.gchannel*self.gnum_heads)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, self.channel)

            # merge windows
            x = x.view(B_, window_s, window_s, self.channel)
            shifted_x = window_reverse(x, window_s, Hp, Wp)  # B H W C

            # reverse cyclic shift
            # if self.shift_size[i] > 0:
            #     x = torch.roll(shifted_x, shifts=(self.shift_size[i], self.shift_size[i]), dims=(1, 2))
            # else:
            #     x = shifted_x
            x = shifted_x
            # print(x.shape) #torch.Size([64, 72, 72, 32])
            # x = x.reshape(B, -1, self.channel)
            # pdb.set_trace()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x_groups.append(x)

        x = torch.cat(x_groups, -1)  # [B,H,W,C]
        #TODO:dynamic fusion
        x =  rearrange(x,'B H W C -> B (H W) C')
        x = self.att(x,x,x)
        x =  rearrange(x,'B (H W) C -> B H W C', H=H)
        # x = self.sknet(x) #[B,C,H,W]
        x = x.view(B, self.dim, HW).permute(0, 2, 1) #[B,HW,C]


        x = rearrange(x, '(b x y) (l w1 w2) d -> b l x y w1 w2 d',
                                b=batch, x=height, y=width,l=agent_size, w1=window_height, w2=window_width)

        
        return x




# swap attention -> max_vit
class Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # flatten
        # pdb.set_trace()
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))
        
        # pdb.set_trace()

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)


class DFFusionBlockMask(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention with
    mask enabled for multi-vehicle cooperation.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out,
                 args):
        super(DFFusionBlockMask, self).__init__()

        self.window_size = window_size
        self.args = args
        drop_path = 1
        if args['win_att']:
            print("loaded the win_att")
            self.window_attention = PreNormResidual(input_dim,
                                                    Attention(input_dim, dim_head,
                                                            drop_out,
                                                            agent_size,
                                                            window_size))
            self.window_ffd = PreNormResidual(input_dim,
                                            FeedForward(input_dim, mlp_dim,
                                                        drop_out))
        if args['grid_att']:
            print("loaded the grid_att")                              
            self.grid_attention = PreNormResidual(input_dim,
                                                Attention(input_dim, dim_head,
                                                            drop_out,
                                                            agent_size,
                                                            window_size))
            self.grid_ffd = PreNormResidual(input_dim,
                                            FeedForward(input_dim, mlp_dim,
                                                        drop_out))
        ##TODO:
        if args['cwin_att']:
            print("loaded the cwin_att")
            self.cwin_attention = PreNormResidual(input_dim,
                                                Conv_Attention(input_dim,
                                                             dim_head,
                                                            drop_out,
                                                            agent_size,
                                                            window_size,
                                                            token_projection='conv'))
            self.cwin_ffd = PreNormResidual(input_dim,
                                            FeedForward(input_dim, mlp_dim,
                                                        drop_out))

        self.muti_window_size = args['muti_window_size']
        if args['dw_att']:
            print("loaded the dw_att")
            # self.dw_attention = PreNormResidual(input_dim,
            #                                     DW_Attention(input_dim,
            #                                                  dim_head,
            #                                                 drop_out,
            #                                                 agent_size,
            #                                                 self.muti_window_size,
            #                                                 ))
            self.dw_attention = PreNormResidual(input_dim,
                                                DW_Attention_M(input_dim,
                                                             dim_head,
                                                            drop_out,
                                                            agent_size,
                                                            self.muti_window_size,
                                                            ))
            self.dwwin_ffd = PreNormResidual(input_dim,
                                            FeedForward(input_dim, mlp_dim,
                                                        drop_out))

            self.drop_path = DropPath(drop_path)


        # if args['kpn']:
        #     print("loaded the lightweight kpn module")
        #     self.KPN = KPN(in_channel=input_dim)

        if args['select_nn']:
            print("loaded the confident selection module")
            self.KPN = KPN_S(in_channel=input_dim)
            channels=args['agent_size']*args['b_size']
            # self.cov_cav = torch.nn.Conv2d(in_channels=channels, out_channels=args['b_size'], kernel_size=1)

        self.depth = self.args['depth']
    def forward(self, x, mask):
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention


        #window attention
        if self.args['win_att']:
            mask_swap = mask

            # mask b h w 1 l -> b x y w1 w2 1 L
            mask_swap = rearrange(mask_swap,
                                'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                                w1=self.window_size, w2=self.window_size)
            x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                        w1=self.window_size, w2=self.window_size)
            x = self.window_attention(x, mask=mask_swap)
            x = self.window_ffd(x)
            x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')


        # grid attention
        if self.args['grid_att']:  

            mask_swap = mask
            mask_swap = rearrange(mask_swap,
                                'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                                w1=self.window_size, w2=self.window_size)
            x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                        w1=self.window_size, w2=self.window_size)
            x = self.grid_attention(x, mask=mask_swap)
            x = self.grid_ffd(x)
            x = rearrange(x, 'b m x y w1 w2 d -> b m d (w1 x) (w2 y)')


        #deliated conv attention 
        if self.args['cwin_att']:
            mask_swap = mask

            # mask b h w 1 l -> b x y w1 w2 1 L
            mask_swap = rearrange(mask_swap,
                                'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                                w1=self.window_size, w2=self.window_size)
            x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                        w1=self.window_size, w2=self.window_size)
            x = self.cwin_attention(x, mask=mask_swap)


            x = self.cwin_ffd(x)
            x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        #dynamic attention 
        if self.args['dw_att']:
            mask_swap = mask

            # mask b h w 1 l -> b x y w1 w2 1 L
            mask_swap = rearrange(mask_swap,
                                'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                                w1=self.window_size, w2=self.window_size)
            x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                        w1=self.window_size, w2=self.window_size)
            # shortcut = x.clone()
            x = self.dw_attention(x, mask=mask_swap)
            x = self.dwwin_ffd(x)
            # FFN
    
            # x = shortcut + self.drop_path(x)  

            x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        ## KPN: data * core
        # if self.args['kpn']:
        #     # pdb.set_trace()
        #     batch_size = x.shape[0]
        #     x = rearrange(x, 'b l c h w-> (b l) c h w')
        #     x = self.KPN(x,x)
        #     x = rearrange(x, '(b l) c h w-> b l c h w', b=batch_size)


        #confident selection
        if self.args['select_nn']:
            # pdb.set_trace()
            batch_size = x.shape[0]
            ego = x[:,0:1,:,:,:].clone()
            cav = x[:,1:,:,:,:].clone()
            cav = rearrange(cav, 'b l c h w-> (b l) c h w')
            cav = self.KPN(cav,cav)
            cav = rearrange(cav, '(b l) c h w-> b l c h w',b=batch_size)
            med_v = torch.median(cav)
            # med_v = torch.mean(cav)

            # zero_v = torch.zeros_like(cav)
            # select_v = torch.where(cav > med_v, cav, zero_v)
            # x[:,0:1,:,:,:] = ego+select_v

            one_v = torch.ones_like(cav)
            select_v = torch.where(cav > med_v, cav, one_v)
            x[:,0:1,:,:,:] = ego*select_v                                                  

        return x


class DFFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out,
                 args):
        super(DFFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),
            Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
        )

    def forward(self, x, mask=None):
        # todo: add mask operation later for mulit-agents
        x = self.block(x)
        return x


class DeformFusionEncoder(nn.Module):
    """
    Data rearrange -> swap block -> mlp_head
    """

    def __init__(self, args):
        super(DeformFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.args = args
        self.depth = args['depth']

        # block related
        input_dim = args['input_dim']
        mlp_dim = args['mlp_dim']
        agent_size = args['agent_size']
        window_size = args['window_size']
        drop_out = args['drop_out']
        dim_head = args['dim_head']

        self.mask = False
        if 'mask' in args:
            self.mask = args['mask']

        for i in range(self.depth):
            if self.mask:
                block = DFFusionBlockMask(input_dim,
                                    mlp_dim,
                                    dim_head,
                                    window_size,
                                    agent_size,
                                    drop_out,
                                    args)

            else:
                block = DFFusionBlock(input_dim,
                                        mlp_dim,
                                        dim_head,
                                        window_size,
                                        agent_size,
                                        drop_out,
                                        args)
            self.layers.append(block)

        # mlp head
        # self.mlp_head = nn.Sequential(
        #     Reduce('b m d h w -> b d h w', 'mean'),
        #     Rearrange('b d h w -> b h w d'),
        #     nn.LayerNorm(input_dim),
        #     nn.Linear(input_dim, input_dim),
        #     Rearrange('b h w d -> b d h w')
        # )

        self.max_operation = Reduce('b m d h w -> b d h w', 'max')
        self.mean_operation = Reduce('b m d h w -> b d h w', 'mean')
        self.mlp_head = nn.Sequential(
            # Reduce('b m d h w -> b d h w', 'mean'),
            # Reduce('b m d h w -> b d h w', 'max'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )
        print("self.args['max_min']: ", self.args['max_min'])
    def forward(self, x, mask=None):
        for stage in self.layers:
            x = stage(x, mask=mask)

        x1 = self.mean_operation(x)
        x2 = self.max_operation(x)
        # 
        if self.args['max_min']:
            x= x1+x2
        else:
            x = x1

        # x=x1+x2

        return x


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = {'input_dim': 512,
            'mlp_dim': 512,
            'agent_size': 4,
            'window_size': 8,
            'dim_head': 4,
            'drop_out': 0.1,
            'depth': 2,
            'mask': True
            }
    block = DeformFusionEncoder(args)
    block.cuda()
    test_data = torch.rand(1, 4, 512, 32, 32)
    test_data = test_data.cuda()
    mask = torch.ones(1, 32, 32, 1, 4)
    mask = mask.cuda()

    output = block(test_data, mask)
    print(output)
