import dataset.dataset as dataset
from dataset.dataset import MyDataset
import torch
import torch.nn as nn
import torch.optim as optim
from model import wangji
from model import bicubic, srcnn, vdsr, srresnet, edsr, esrgan, rdn, lapsrn, tsrn
from model import recognizer
from model import moran
from model import crnn
from loss import gradient_loss, percptual_loss, image_loss, wangji_loss
from torch.utils.tensorboard import SummaryWriter
from utils.labelmaps import get_vocabulary, labels2strs
from utils import util, ssim_psnr, utils_moran, utils_crnn
import os
import logging
from datetime import datetime


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.cfg = config
        self.args = args
        self.load_dataset = MyDataset
        self.batch_size = args.batch_size if args.batch_size is not None else self.cfg.batch_size
        # self.align_collate = alignCollate_real
        # self.align_collate = alignCollate_syn
        # self.mask = self.args.mask
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir_name = 'log_' + str(datetime.now()).replace(':', '_').replace(' ', '_')
        self.make_writer()
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.resume = args.resume if args.resume is not None else config.resume
        self.logging = logging
        self.scale_factor = self.cfg.scale_factor


    def get_train_data(self):
        cfg = self.cfg # инициализация конфига
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir: # Обычные изображния
                el_index = cfg.train_data_dir.index(data_dir_)
                train_data_annotations_file = cfg.train_data_annotations_file[el_index]
                print('collect dataset: '+data_dir_)
                dataset_list.append(
                    self.load_dataset(cfg=cfg,
                                      isTextZoom=False,
                                      data_dir=data_dir_,
                                      data_annotations_file=train_data_annotations_file,
                                      # voc_type=cfg.voc_type,
                                      # max_len=cfg.max_len,
                                      ).load_dataset()) # создаётся объект класса loadDataset
            if len(cfg.train_data_textzoom_dir) > 0:
                for data_dir_ in cfg.train_data_textzoom_dir: # TextZoom
                    print('collect dataset: '+data_dir_)
                    dataset_list.append(
                        self.load_dataset(cfg=cfg,
                                          isTextZoom=True,
                                          data_dir=data_dir_,
                                          # voc_type=cfg.voc_type,
                                          # max_len=cfg.max_len,
                                          ).load_dataset()) # создаётся объект класса loadDataset
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=self.batch_size,
        #     shuffle=False, num_workers=int(0),
        #     collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
        #                                   mask=self.mask),
        #     drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            drop_last=True, pin_memory=True)
        return train_dataset, train_loader


    def get_test_val_data(self):
        cfg = self.cfg

        if isinstance(cfg.test_val_data_dir, list):
            dataset_list = []
            loader_list = []

            for data_dir_ in cfg.test_val_data_dir: # Обычные изображния
                print('collect dataset: '+data_dir_)
                el_index = cfg.test_val_data_dir.index(data_dir_)
                data_annotations_file = cfg.test_val_data_annotations_file[el_index]

                test_val_dataset = self.load_dataset(cfg=cfg,
                                      isTextZoom=False,
                                      data_dir=data_dir_,
                                      data_annotations_file=data_annotations_file,
                                      # voc_type=cfg.voc_type,
                                      # max_len=cfg.max_len,
                                      ).load_dataset()
                dataset_list.append(test_val_dataset) # создаётся объект класса loadDataset

                # test_val_loader = torch.utils.data.DataLoader(
                #     test_val_dataset, batch_size=self.batch_size,
                #     shuffle=True, num_workers=int(cfg.workers),
                #     collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width,
                #                                   down_sample_scale=cfg.down_sample_scale,
                #                                   mask=self.mask),
                #                                   drop_last=True)

                test_val_loader = torch.utils.data.DataLoader(
                    test_val_dataset, batch_size=self.batch_size,
                    shuffle=False, num_workers=int(cfg.workers),drop_last=True, pin_memory=True)
                loader_list.append(test_val_loader)

            if len(cfg.test_val_textzoom_data_dir)>0:
                for data_dir_ in cfg.test_val_textzoom_data_dir: # TextZoom
                    print('collect dataset: '+data_dir_)
                    test_val_dataset = self.load_dataset(cfg=cfg,
                                          isTextZoom=True,
                                          data_dir=data_dir_,
                                          # voc_type=cfg.voc_type,
                                          # max_len=cfg.max_len,
                                          ).load_dataset()
                    dataset_list.append(test_val_dataset) # создаётся объект класса loadDataset

                    # test_val_loader = torch.utils.data.DataLoader(
                    #     test_val_dataset, batch_size=self.batch_size,
                    #     shuffle=True, num_workers=int(0),
                    #     collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width,
                    #                                   down_sample_scale=cfg.down_sample_scale,
                    #                                   mask=self.mask),
                    #     drop_last=True)
                    test_val_loader = torch.utils.data.DataLoader(
                        test_val_dataset, batch_size=self.batch_size,
                        shuffle=False, num_workers=int(cfg.workers),drop_last=True, pin_memory=True)
                    loader_list.append(test_val_loader)

            # test_val_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        return dataset_list, loader_list


    def generator_init(self):
        cfg = self.cfg
        if self.args.arch == 'wangji':
            model = wangji.Wangji(cfg, scale_factor=cfg.scale_factor)
            image_crit = wangji_loss.WangjiLoss(args=self.args, cfg=self.cfg)
        elif self.args.arch == 'tsrn':
            model = tsrn.TSRN(scale_factor=cfg.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == 'bicubic' and self.args.test:
            model = bicubic.BICUBIC(scale_factor=cfg.scale_factor)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srcnn':
            model = srcnn.SRCNN(scale_factor=cfg.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'vdsr':
            model = vdsr.VDSR(scale_factor=cfg.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srres':
            model = srresnet.SRResNet(scale_factor=cfg.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'esrgan':
            model = esrgan.RRDBNet(scale_factor=cfg.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'rdn':
            model = rdn.RDN(scale_factor=cfg.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'edsr':
            model = edsr.EDSR(scale_factor=cfg.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'lapsrn':
            model = lapsrn.LapSRN(scale_factor=cfg.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = lapsrn.L1_Charbonnier_loss()
        else:
            raise ValueError
        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            image_crit.to(self.device)
            if cfg.ngpu > 1:
                model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))
                image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))
            if self.resume != '':
                print('loading pre-trained model from %s ' % self.resume)
                if self.cfg.ngpu == 1:
                    model.load_state_dict(torch.load(self.resume)['state_dict_G'])
                else:
                    model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
        return {'model': model, 'crit': image_crit}


    def optimizer_init(self, model):
        cfg = self.cfg
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999), amsgrad=True)
        # optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0)
        return optimizer


    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.cfg
        aster_info = AsterInfo(cfg.voc_type)
        model_path = self.cfg.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model, aster_info


    def make_writer(self):
        self.writer = SummaryWriter('checkpoint/{}'.format(self.log_dir_name))


    def save_checkpoint(self, netG, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, exp_name):
        # ckpt_path = os.path.join('checkpoint', exp_name, self.vis_dir)
        ckpt_path = os.path.join('checkpoint', exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'state_dict_G': netG.state_dict(),
            'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in netG.parameters()]),
            'converge': converge_list
        }
        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
        else:
            torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))


    def parse_crnn_data(self, imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor



class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)