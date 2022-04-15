import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np

from timm.models.layers import trunc_normal_, DropPath


class Wangji(nn.Module):
    def __init__(self, cfg, scale_factor, in_chans = 3):
        super(Wangji, self).__init__()

        if cfg.convNext_type == 'T':
            depths = [3, 3, 9]
            dims = [96, 192, 384]
        elif cfg.convNext_type == 'S':
            depths = [3, 3, 27]
            dims = [96, 192, 384]
        # elif cfg.convNext_type == 'B': # Не сработает, т.к. нужно, чтобы dims[2] делилось на 3.
        #     depths = [3, 3, 27]
        #     dims = [128, 256, 512]
        elif cfg.convNext_type == 'L':
            depths = [3, 3, 27]
            dims = [192, 384, 768]
        # elif cfg.convNext_type == 'XL': # Не сработает, т.к. нужно, чтобы dims[2] делилось на 3.
        #     depths = [3, 3, 27]
        #     dims = [256, 512, 1024]
        else:
            depths = [3, 3, 9]
            dims = [96, 192, 384]

        #LongSkip

        self.convLS0 = nn.Conv2d(in_channels = in_chans, out_channels = 32, kernel_size = 2, stride=[2,3], dilation=[1,3])
        self.convLS1 = nn.Conv2d(in_channels = 32, out_channels = (dims[2] * scale_factor - dims[2]), kernel_size = [2,3], stride=[4,2], dilation=3)

        #Blue Branch

        self.convBlue1 = nn.Conv2d(in_channels = dims[0], out_channels = dims[1], kernel_size = 2, padding=[0,1], stride=2, dilation=[1,3])
        self.convBlue2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=2, padding=[0, 1], stride=2, dilation=[1, 3])

        #ConvNext (Red Branch) https://paperswithcode.com/paper/a-convnet-for-the-2020s

        drop_path_rate = 0.
        layer_scale_init_value = 1e-6
        self.convnext_downsample_layers = nn.ModuleList()  # stem and 2 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.convnext_downsample_layers.append(stem)
        for i in range(2):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.convnext_downsample_layers.append(downsample_layer)

        self.convnext_blocks = nn.ModuleList()  # 3 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.convnext_blocks.append(stage)
            cur += depths[i]

        #PreProcessing Branch

        self.convPreProc00 = nn.Conv2d(in_channels = dims[0], out_channels = 32, kernel_size = 3, padding=1, stride=2)
        self.convPreProc01 = nn.Conv2d(in_channels=32, out_channels=int(dims[2]/3), kernel_size=3, padding=1, stride=2)
        self.convPreProc1 = nn.Conv2d(in_channels=dims[1], out_channels=int(dims[2]/3), kernel_size=3, padding=1, stride=2)
        self.convPreProc2 = nn.Conv2d(in_channels=dims[2], out_channels=int(dims[2]/3), kernel_size=1)

        # correlation matrix/Attation from HAN https://paperswithcode.com/paper/single-image-super-resolution-via-a-holistic

        self.LAM = LAM_Module(dims[2])

        # Upsampler from HAN https://paperswithcode.com/paper/a-convnet-for-the-2020s

        # conv_HAN = default_conv_form_HAN
        # n_feats = dims[2] * scale_factor
        # n_colors = 3
        # ks = 3
        # upsample = [Upsampler(conv_HAN, scale_factor, n_feats, act=False),
        #             conv_HAN(n_feats, n_colors, ks)]
        #
        # self.upsample = nn.Sequential(*upsample)
        self.upsample = sub_pixel(scale_factor)
        self.finalConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=True)

        # self.TGT_VOCAB_SIZE = 88 # большие и маленькие английские буквы + 10 цифр + 25 символов + пробел
        # self.BOS, self.EOS, self.PAD, self.BLANK = self.TGT_VOCAB_SIZE+1, self.TGT_VOCAB_SIZE+2, self.TGT_VOCAB_SIZE+3, 0
        self.FULL_VOCAB_SIZE = cfg.FULL_VOCAB_SIZE

        self.linear = nn.Linear(dims[2], self.FULL_VOCAB_SIZE)

        # LSTM

        input_size = cfg.FULL_VOCAB_SIZE
        hidden_size = input_size
        tagset_size = self.FULL_VOCAB_SIZE

        self.LSTM = LSTMTagger(input_size, hidden_size, tagset_size)


    def forward(self, x):
        img = x

        y_green_1 = self.convLS0(x)
        y_green_2 = self.convLS1(y_green_1)

        y_downsample0 = self.convnext_downsample_layers[0](x)
        y_block0 = self.convnext_blocks[0](y_downsample0)
        y_convPreProc00 = self.convPreProc00(y_block0)
        y_convPreProc01 = self.convPreProc01(y_convPreProc00)

        y_blue_conv1 = self.convBlue1(y_block0)
        y_downsample1 = self.convnext_downsample_layers[1](y_block0)
        y_block1 = self.convnext_blocks[1](y_downsample1)
        y_convPreProc1 = self.convPreProc1(y_block1)
        y_summ1 = y_blue_conv1 + y_block1

        y_blue_conv2 = self.convBlue2(y_summ1)
        y_downsample2 = self.convnext_downsample_layers[2](y_block1)
        y_block2 = self.convnext_blocks[2](y_downsample2)
        y_convPreProc2 = self.convPreProc2(y_block2)
        y_summ2 = y_blue_conv2 + y_block2

        y_concat1 = torch.cat([y_summ2, y_green_2], 1)
        # y_concat2 = torch.cat([y_convPreProc01, y_convPreProc1, y_convPreProc2], 1)
        y_concat2 = torch.stack([y_convPreProc01, y_convPreProc1, y_convPreProc2], 1)

        y_upsample = self.upsample(y_concat1)
        y_sr = self.finalConv(y_upsample)

        y_attention = self.LAM(y_concat2)

        y_summ3 = y_block2 + y_attention
        # y_summ3 = y_block2
        # y_summ3 = y_attention

        y_flat = rearrange(y_summ3, 'b c h w -> b (h w) c')

        y_lin = self.linear(y_flat)

        tag_scores = self.LSTM(y_lin)

        tag_scores = rearrange(tag_scores, 'n t c -> t n c') # для стс лосса

        return y_sr, tag_scores



class ConvNeXt(nn.Module):
        r""" ConvNeXt
            A PyTorch impl of : `A ConvNet for the 2020s`  -
              https://arxiv.org/pdf/2201.03545.pdf

        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """

        def __init__(self, in_chans=3, num_classes=1000,
                     depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                     # depths - кол-во блоков после каждого снижения разрешения
                     layer_scale_init_value=1e-6, head_init_scale=1.,
                     ):
            super().__init__()

            self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(stem)
            for i in range(3):
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
            dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            cur = 0
            for i in range(4):
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]

            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
            self.head = nn.Linear(dims[-1], num_classes)

            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

        def forward_features(self, x):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

        def forward(self, x):
            x = self.forward_features(x)
            x = self.head(x)
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LAM_Module(nn.Module):
    """ Layer attention module from HAN https://paperswithcode.com/paper/single-image-super-resolution-via-a-holistic"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1) # proj_query = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale*8))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class LSTMTagger(nn.Module):

    def __init__(self, input_size, hidden_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_size

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 1, batch_first=True, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_size*2, tagset_size) #!!! hidden_size*2 из-за bidirectional=True. Уточни у Саши МОЖЕТ ВЕРНУТЬ
        self.hidden2tag = nn.Linear(hidden_size, tagset_size) #!!! hidden_size*2 из-за bidirectional=True. Уточни у Саши МОЖЕТ ВЕРНУТЬ

    def forward(self, input):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # embeds = self.word_embeddings(sentence)
        # print("embeds.shape:",embeds.shape)
        # print("embeds.view(len(sentence), 1, -1).shape:",embeds.view(len(sentence), 1, -1).shape)
        lstm_out, _ = self.lstm(input)
        # print("lstm_out:",lstm_out)
        hidden2tag = self.hidden2tag(lstm_out) #МОЖЕТ ВЕРНУТЬ
        # print("tag_space:",tag_space)
        tag_scores = F.log_softmax(lstm_out, dim=1) #hidden2tag
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return tag_scores

# https://paperswithcode.com/paper/single-image-super-resolution-via-a-holistic
def default_conv_form_HAN(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# https://paperswithcode.com/paper/single-image-super-resolution-via-a-holistic
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        scale = scale*8
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)