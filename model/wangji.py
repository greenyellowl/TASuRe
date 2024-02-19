import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np
from utils import exceptions
from timm.models.layers import trunc_normal_, DropPath

from model.transformer import Transformer
from model.moran.asrn_res import Attention

from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead

class Wangji(nn.Module):
    def __init__(self, cfg, scale_factor, in_chans = 3):
        super(Wangji, self).__init__()
        self.cfg = cfg

        if self.cfg.input_mask:
            in_chans += 1

        # Statial Transformation Network
        if self.cfg.stn:
            self.tps_inputsize = [32, 64]
            tps_outputsize = [self.cfg.height//self.cfg.scale_factor,
                              self.cfg.width//self.cfg.scale_factor]
            num_control_points = 20
            tps_margins = [0.05, 0.05]

            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_chans,
                num_ctrlpoints=num_control_points,
                activation='none')


        if self.cfg.convNext_type == 'TTT':
            depths = [1, 1, 1]
            dims = [96, 192, 384]
            finalConv_in_channels = 3 if cfg.scale_factor == 2 else None
        elif self.cfg.convNext_type == 'TT':
            depths = [3, 3, 3]
            dims = [96, 192, 384]
            finalConv_in_channels = 3 if cfg.scale_factor == 2 else None
        elif self.cfg.convNext_type == 'T':
            depths = [3, 3, 9]
            dims = [96, 192, 384]
            finalConv_in_channels = 3 if cfg.scale_factor == 2 else None
        elif self.cfg.convNext_type == 'S':
            depths = [3, 3, 27]
            dims = [96, 192, 384]
            finalConv_in_channels = 3
        # elif self.cfg.convNext_type == 'B': # Не сработает, т.к. нужно, чтобы dims[2] делилось на 3.
        #     depths = [3, 3, 27]
        #     dims = [128, 256, 512]
        elif self.cfg.convNext_type == 'L':
            depths = [3, 3, 27]
            dims = [192, 384, 768]
            finalConv_in_channels = 6 if cfg.scale_factor == 2 else 3
        elif self.cfg.convNext_type == 'XL': # Не сработает, т.к. нужно, чтобы dims[2] делилось на 3.
            depths = [3, 3, 27]
            dims = [256, 512, 1024]
            finalConv_in_channels = 8
        else:
            depths = [3, 3, 9]
            dims = [96, 192, 384]  
        self.dims = dims

        if self.cfg.enable_ConvNext or self.cfg.train_after_sr:
            #ConvNext (Red Branch) https://paperswithcode.com/paper/a-convnet-for-the-2020s

            drop_path_rate = 0.
            layer_scale_init_value = 1e-6
            self.convnext_downsample_layers = nn.ModuleList()  # stem and 2 intermediate downsampling conv layers
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first") if cfg.convnext_layernorm else nn.BatchNorm2d(dims[0])
            )
            self.convnext_downsample_layers.append(stem)
            for i in range(2):
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first") if cfg.convnext_layernorm else nn.BatchNorm2d(dims[i]),
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
            
            if self.cfg.freeze_convnext:
                for param in self.convnext_downsample_layers.parameters():
                    param.requires_grad = False

                for param in self.convnext_blocks.parameters():
                    param.requires_grad = False

        self.act = nn.GELU()
        self.act_sigm = nn.Sigmoid()
        
        if self.cfg.enable_sr or self.cfg.train_after_sr:
            #LongSkip

            if cfg.scale_factor == 2:
                out_channels = 32
                self.convLS0 = nn.Conv2d(in_channels = in_chans, out_channels = out_channels, kernel_size = 2, stride=[2,3], dilation=[1,3])
                if self.cfg.batch_norm:
                    self.bn_ls0 = nn.BatchNorm2d(out_channels)
                out_channels = dims[2] * scale_factor - dims[2]
                self.convLS1 = nn.Conv2d(in_channels = 32, out_channels = out_channels, kernel_size = [2,3], stride=[4,2], dilation=3)
                if self.cfg.batch_norm:
                    self.bn_ls1 = nn.BatchNorm2d(out_channels)
            elif cfg.scale_factor == 4:
                out_channels = 32
                self.convLS0 = nn.Conv2d(in_channels = in_chans, out_channels = out_channels, kernel_size = 2, stride=2, dilation=[3,1])
                if self.cfg.batch_norm:
                    self.bn_ls0 = nn.BatchNorm2d(out_channels)
                out_channels = dims[2]
                self.convLS1 = nn.Conv2d(in_channels = 32, out_channels = out_channels, kernel_size = 3, stride=[2,3], dilation=[2,3])
                if self.cfg.batch_norm:
                    self.bn_ls1 = nn.BatchNorm2d(out_channels)
                

            #Blue Branch

            if cfg.scale_factor == 2 or cfg.scale_factor == 1:

                out_channels = dims[1]
                self.convBlue1 = nn.Conv2d(in_channels = dims[0], out_channels = out_channels, kernel_size = 2, padding=[0,1], stride=2, dilation=[1,3])
                if self.cfg.batch_norm:
                    self.bn_blue1 = nn.BatchNorm2d(out_channels)
                out_channels = dims[2]
                self.convBlue2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=2, padding=[0, 1], stride=2, dilation=[1, 3])
                if self.cfg.batch_norm:
                    self.bn_blue2 = nn.BatchNorm2d(out_channels)
                # if not cfg.transpose_upsample:
                #     self.convBlue3 = nn.Conv2d(in_channels=dims[2]*2, out_channels=dims[2]*scale_factor, kernel_size=1)
            
            if cfg.scale_factor == 4:

                out_channels = dims[1]
                self.convBlue1 = nn.Conv2d(in_channels = dims[0], out_channels = out_channels, kernel_size = [3,2], stride=[1,2], dilation=[2,1])
                if self.cfg.batch_norm:
                    self.bn_blue1 = nn.BatchNorm2d(out_channels)
                out_channels = dims[2]
                self.convBlue2 = nn.Conv2d(in_channels=dims[1], out_channels=out_channels, kernel_size=[3,5], stride=1, dilation=1)
                if self.cfg.batch_norm:
                    self.bn_blue2 = nn.BatchNorm2d(out_channels)
                
                if not cfg.transpose_upsample:
                    if self.cfg.convNext_type == 'S': out_channels = 1024 * 3
                    else: out_channels = dims[2]*scale_factor
                    self.convBlue3 = nn.Conv2d(in_channels=dims[2]*2, out_channels=out_channels, kernel_size=1)
                    if self.cfg.batch_norm:
                        self.bn_blue3 = nn.BatchNorm2d(out_channels)
            
            if cfg.transpose_upsample:
                if cfg.transpose_type == 'big':
                    self.convUpsample1 = nn.ConvTranspose2d(in_channels=dims[2]*2, out_channels=dims[1], kernel_size=[2,5], padding=0, output_padding=0, stride=[2,1], dilation=1)
                    self.convUpsample2 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=dims[0], kernel_size=[5,2], padding=0, output_padding=0, stride=[1,2], dilation=1)
                    self.convUpsample3 = nn.ConvTranspose2d(in_channels=dims[0], out_channels=dims[1], kernel_size=[9,2], padding=0, output_padding=0, stride=[1,2], dilation=1)
                    if cfg.scale_factor >= 2:
                        self.convUpsample4 = nn.ConvTranspose2d(in_channels=dims[1]+in_chans, out_channels=dims[1], kernel_size=[17,33], padding=0, output_padding=0, stride=1, dilation=1)
                    if cfg.scale_factor == 4:
                        self.convUpsample5 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=dims[0], kernel_size=[33,65], padding=0, output_padding=0, stride=1, dilation=1)
                elif cfg.transpose_type == 'med':
                    self.convUpsample1 = nn.ConvTranspose2d(in_channels=dims[2]*2, out_channels=dims[1], kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)
                    self.convUpsample2 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=dims[0], kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)
                    self.convUpsample3 = nn.ConvTranspose2d(in_channels=dims[0], out_channels=dims[1], kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)
                    if cfg.scale_factor >= 2:
                        self.convUpsample4 = nn.ConvTranspose2d(in_channels=dims[1]+in_chans, out_channels=dims[1], kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)
                    if cfg.scale_factor == 4:
                        self.convUpsample5 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=dims[0], kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)
                elif cfg.transpose_type == 'small':
                    self.convUpsample1 = nn.ConvTranspose2d(in_channels=dims[2]*2, out_channels=dims[1], kernel_size=2, padding=0, output_padding=0, stride=2, dilation=1)
                    self.convUpsample2 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=dims[0], kernel_size=2, padding=0, output_padding=0, stride=2, dilation=1)
                    self.convUpsample3 = nn.ConvTranspose2d(in_channels=dims[0], out_channels=dims[1], kernel_size=2, padding=0, output_padding=0, stride=2, dilation=1)
                    if cfg.scale_factor >= 2:
                        self.convUpsample4 = nn.ConvTranspose2d(in_channels=dims[1]+in_chans, out_channels=dims[1], kernel_size=2, padding=0, output_padding=0, stride=2, dilation=1)
                    if cfg.scale_factor == 4:
                        self.convUpsample5 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=dims[0], kernel_size=2, padding=0, output_padding=0, stride=2, dilation=1)
                finalConv_in_channels = dims[0] if cfg.scale_factor == 4 else dims[1]
            else:
                self.upsample = sub_pixel(scale_factor)
            
            self.finalConv = nn.Conv2d(in_channels=finalConv_in_channels, out_channels=3, kernel_size=3, padding=1, bias=True)

            if self.cfg.freeze_sr:
                self.convLS0.weight.requires_grad = False
                self.convLS0.bias.requires_grad = False

                self.convLS1.weight.requires_grad = False
                self.convLS1.bias.requires_grad = False

                self.convBlue1.weight.requires_grad = False
                self.convBlue1.bias.requires_grad = False

                self.convBlue2.weight.requires_grad = False
                self.convBlue2.bias.requires_grad = False

                self.finalConv.weight.requires_grad = False
                self.finalConv.bias.requires_grad = False
        
        
        if self.cfg.enable_rec and not self.cfg.recognizer == 'transformer':
            #PreProcessing Branch
            out_channels = int(dims[2]/3)
            self.convPreProc0 = nn.Conv2d(in_channels=dims[0], out_channels=out_channels, kernel_size=1)
            # Нужно upscale Nearest Neighbors или Bi-Linear Interpolation или Transposed Convolutions
            self.convPreProc1 = nn.ConvTranspose2d(in_channels=dims[1], out_channels=out_channels, kernel_size=2, stride=2)
            self.convPreProc2 = nn.ConvTranspose2d(in_channels=dims[2], out_channels=out_channels, kernel_size=4, stride=4)
            if self.cfg.batch_norm:
                self.bn_PreProc = nn.BatchNorm2d(out_channels)

            # correlation matrix/Attation from HAN https://paperswithcode.com/paper/single-image-super-resolution-via-a-holistic
            self.LAM = LAM_Module(dims[2])

            self.FULL_VOCAB_SIZE = cfg.FULL_VOCAB_SIZE

            # self.linear = nn.Linear((16*dims[2]), self.FULL_VOCAB_SIZE)

            # LSTM

            if self.cfg.recognizer_input == 'att':
                in_features_linear = int((cfg.height/scale_factor/2)*dims[2])
                input_size_lstm = cfg.lstm_hidden_size
                # input_size_lstm = in_features_linear
            elif self.cfg.recognizer_input == 'convnext':
                # input_size = dims[self.cfg.recognizer_input_convnext]
                in_features_linear = (768 if self.cfg.convNext_type == 'T' or self.cfg.convNext_type == 'S' else 1536) if self.cfg.recognizer_input_convnext == 1 else dims[2]
                # self.last_convNext_block = eval("y_block"+str(self.cfg.recognizer_input_convnext))
            num_layers = cfg.num_lstm_layers
            batch_first = True
            bidirectional = cfg.lstm_bidirectional
            
            if cfg.moran_att:
                tagset_size_lstm = input_size_lstm
            else:
                tagset_size_lstm  = self.cfg.FULL_VOCAB_SIZE

            if cfg.recognizer == 'lstm':
                self.pre_lstm_linear = nn.Linear(in_features=in_features_linear, out_features=input_size_lstm)
                self.LSTM = LSTMTagger(input_size_lstm, input_size_lstm, num_layers, batch_first, bidirectional, tagset_size_lstm)

                # MORAN attention
                if cfg.moran_att:
                    nh = cfg.lstm_hidden_size
                    nclass = self.cfg.FULL_VOCAB_SIZE
                    num_embeddings = cfg.lstm_hidden_size
                    self.attention = Attention(input_size=nh, hidden_size=nh, num_classes=nclass, num_embeddings=num_embeddings, CUDA=True)
                
            elif cfg.recognizer == 'transformer':
                self.build_up_transformer()
            else:
                raise exceptions.WrongRecognizer

            if self.cfg.freeze_rec:
                self.convPreProc0.weight.requires_grad = False
                self.convPreProc0.bias.requires_grad = False

                self.convPreProc1.weight.requires_grad = False
                self.convPreProc1.bias.requires_grad = False

                self.convPreProc2.weight.requires_grad = False
                self.convPreProc2.bias.requires_grad = False

                for param in self.LAM.parameters():
                    param.requires_grad = False

                for param in self.LSTM.parameters():
                    param.requires_grad = False

                for param in self.transformer.parameters():
                    param.requires_grad = False
                

    def build_up_transformer(self):
        transformer = Transformer().cuda()
        transformer = nn.DataParallel(transformer)
        transformer.load_state_dict(torch.load('/home/helen/code/TextSR/pretrained/pretrain_transformer.pth'))
        transformer.eval()
        self.transformer = transformer
    
    
    def forward(self, x, label_len, label_strs):
        # Rectify input image
        if self.cfg.stn:
            if x.shape[2] != self.tps_inputsize[0] or x.shape[3] != self.tps_inputsize[1]:
                x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        # img = x
        batch_size = x.shape[0]

        if self.cfg.enable_ConvNext:
            # Red ConvNext Branch

            y_downsample0 = self.convnext_downsample_layers[0](x)
            y_block0 = self.convnext_blocks[0](y_downsample0)
            del y_downsample0

            y_downsample1 = self.convnext_downsample_layers[1](y_block0)
            y_block1 = self.convnext_blocks[1](y_downsample1)
            del y_downsample1

            if self.cfg.recognizer_input == 'att' or self.cfg.enable_sr or \
                (self.cfg.recognizer_input == 'convnext' and self.cfg.recognizer_input_convnext == 2):
                y_downsample2 = self.convnext_downsample_layers[2](y_block1)
                y_block2 = self.convnext_blocks[2](y_downsample2)
                del y_downsample2
        
        y_sr = None
        if self.cfg.enable_sr:
            y_green_1 = self.convLS0(x)
            if self.cfg.batch_norm:
                y_green_1 = self.bn_ls0(y_green_1)
            y_green_1 = self.act(y_green_1)
            y_green_2 = self.convLS1(y_green_1)
            if self.cfg.batch_norm:
                y_green_2 = self.bn_ls1(y_green_2)
            y_green_2 = self.act(y_green_2)
            # del y_green_1

            y_blue_conv1 = self.convBlue1(y_block0)
            if self.cfg.batch_norm:
                y_blue_conv1 = self.bn_blue1(y_blue_conv1)
            y_blue_conv1 = self.act(y_blue_conv1)
            y_summ1 = y_blue_conv1 + y_block1
            # del y_blue_conv1
            
            y_blue_conv2 = self.convBlue2(y_summ1)
            if self.cfg.batch_norm:
                y_blue_conv2 = self.bn_blue2(y_blue_conv2)
            y_blue_conv2 = self.act(y_blue_conv2)
            y_summ2 = y_blue_conv2 + y_block2
            # del y_blue_conv2

            y_concat1 = torch.cat([y_summ2, y_green_2], 1)
            # del y_green_2
            # del y_summ2

            if self.cfg.transpose_upsample:
                y_up_conv1 = self.convUpsample1(y_concat1)
                y_up_conv1 = self.act(y_up_conv1)
                y_summ_t1 = y_up_conv1 + y_summ1
                y_up_conv2 = self.convUpsample2(y_summ_t1)
                y_up_conv2 = self.act(y_up_conv2)
                y_summ_t2 = y_up_conv2 + y_block0
                if self.cfg.scale_factor == 1:
                    y_up_conv3 = self.convUpsample3(y_summ_t2)
                    y_up_conv3 = self.act(y_up_conv3)
                    y_to_final_conv = torch.cat([y_up_conv3, x], 1)
                elif self.cfg.scale_factor == 2:
                    y_up_conv3 = self.convUpsample3(y_summ_t2)
                    y_up_conv3 = self.act(y_up_conv3)
                    y_concat3 = torch.cat([y_up_conv3, x], 1)
                    y_up_conv4 = self.convUpsample4(y_concat3)
                    y_to_final_conv = self.act(y_up_conv4)
                elif self.cfg.scale_factor == 4:
                    y_up_conv4 = self.convUpsample4(y_concat3)
                    y_up_conv4 = self.act(y_up_conv4)
                    y_up_conv5 = self.convUpsample5(y_up_conv4)
                    y_to_final_conv = self.act(y_up_conv5)
            else:
                y_blue_conv3 = self.convBlue3(y_concat1) if self.cfg.scale_factor == 4 else y_concat1
                if self.cfg.scale_factor == 4:
                    if self.cfg.batch_norm:
                        y_blue_conv3 = self.bn_blue3(y_blue_conv3)
                y_blue_conv3 = self.act(y_blue_conv3)
                y_to_final_conv = self.upsample(y_blue_conv3)
            y_sr = self.finalConv(y_to_final_conv)
            # print('before:',f'{y_to_final_conv.min().item():.2f}', '|',
            #         f'{y_to_final_conv.max().item():.2f}', '|',
            #         f'{torch.mean(y_to_final_conv).item():.2f}', '|',
            #         f'{torch.std(y_to_final_conv).item():.2f}', '|',
            #         '\nafter:',f'{y_sr.min().item():.2f}', '|',
            #         f'{y_sr.max().item():.2f}', '|',
            #         f'{torch.mean(y_sr).item():.2f}', '|',
            #         f'{torch.std(y_sr).item():.2f}',)
            y_sr = self.act_sigm(y_sr)
            # del y_concat1


        tag_scores = None
        if self.cfg.enable_rec and not self.cfg.recognizer == 'transformer':
            if self.cfg.recognizer_input == 'convnext':
                # block = eval("y_block"+str(self.cfg.recognizer_input_convnext))
                block = y_block1 if self.cfg.recognizer_input_convnext == 1 else y_block2
                vector_len = int(self.dims[self.cfg.recognizer_input_convnext] * block.shape[2] * block.shape[3] / 32)
                y_flat = torch.zeros(batch_size, 32, vector_len, dtype=torch.float32, device=block.device)
                len_slice = int(self.dims[self.cfg.recognizer_input_convnext]/32)
                for i in range(32):
                    vector = block[:,i*len_slice:(i+1)*len_slice,:,:]
                    y_flat[:,i] = rearrange(vector, 'b c h w -> b (c h w)')
            elif self.cfg.recognizer_input == 'att':
                y_convPreProc0 = self.convPreProc0(y_block0)
                if self.cfg.batch_norm:
                    y_convPreProc0 = self.bn_PreProc(y_convPreProc0)
                y_convPreProc0 = self.act(y_convPreProc0)
                y_convPreProc1 = self.convPreProc1(y_block1)
                if self.cfg.batch_norm:
                    y_convPreProc1 = self.bn_PreProc(y_convPreProc1)
                y_convPreProc1 = self.act(y_convPreProc1)
                y_convPreProc2 = self.convPreProc2(y_block2)
                if self.cfg.batch_norm:
                    y_convPreProc2 = self.bn_PreProc(y_convPreProc2)
                y_convPreProc2 = self.act(y_convPreProc2)

                y_concat2 = torch.stack([y_convPreProc0, y_convPreProc1, y_convPreProc2], 1)

                y_attention = self.LAM(y_concat2)

                # y_summ3 = y_block2 + y_attention
                # y_summ3 = y_block2
                y_summ3 = y_attention

                y_flat = rearrange(y_summ3, 'b c h w -> b w (h c)')

                # y_linear = self.linear(lstm_out)
            y_flat = self.pre_lstm_linear(y_flat)
            y_flat = self.act(y_flat)
            tag_scores = self.LSTM(y_flat)
            tag_scores = rearrange(tag_scores, 'n t c -> t n c')
            if self.cfg.moran_att:
                tag_scores = self.attention(tag_scores, label_len, label_strs, False)
                tag_scores_batch = torch.zeros((x.shape[0], x.shape[3]//2, self.cfg.FULL_VOCAB_SIZE), dtype=torch.float32)
                start_ind = 0
                for b, l in enumerate(label_len):
                    tag_scores_batch[b, 0:l, :] = tag_scores[start_ind:start_ind+l, :]
                    start_ind += l
                tag_scores = rearrange(tag_scores_batch, 'n t c -> t n c')
                tag_scores = F.log_softmax(tag_scores, dim=2)
                
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
        PixelShuffle = nn.PixelShuffle(scale*8)
        modules.append(PixelShuffle)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class LSTMTagger(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_size

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, batch_first=batch_first, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_size*2, tagset_size) #!!! hidden_size*2 из-за bidirectional=True. Уточни у Саши МОЖЕТ ВЕРНУТЬ
        hidden_size = hidden_size*2 if bidirectional else hidden_size
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
        tag_scores = F.log_softmax(hidden2tag, dim=2) #hidden2tag
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