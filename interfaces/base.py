from collections import OrderedDict
import string
import dataset.dataset as dataset
from dataset.dataset import MyDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_warmup_scheduler import WarmupScheduler # https://github.com/hysts/pytorch_warmup-scheduler
from model import wangji
from model import bicubic, srcnn, vdsr, srresnet, edsr, esrgan, rdn, lapsrn, tsrn
from model import recognizer
from model import moran
from model import crnn
from loss import gradient_loss, percptual_loss, image_loss, wangji_loss, text_focus_loss
from torch.utils.tensorboard import SummaryWriter
from utils.labelmaps import get_vocabulary, labels2strs
from utils import util, ssim_psnr, utils_moran, utils_crnn, exceptions
import os
import logging
from datetime import datetime
from string import ascii_uppercase, ascii_lowercase


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.cfg = config
        self.args = args
        self.load_dataset = MyDataset
        self.batch_size = self.cfg.batch_size
        # self.align_collate = alignCollate_real
        # self.align_collate = alignCollate_syn
        # self.mask = self.args.mask
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir_name = 'log_' + str(datetime.now()).replace(':', '_').replace(' ', '_')
        self.make_writer()
        self.make_multi_writer()
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.resume = args.resume if args.resume is not None else config.resume
        self.logging = logging
        self.scale_factor = self.cfg.scale_factor

        self.Letters, self.Symbols, self.FullVocab, self.FullVocabList = util.get_vocab(self.cfg)

        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)


    def get_train_data(self):
        cfg = self.cfg # инициализация конфига
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            if len(cfg.train_data_dir) > 0:
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

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True, num_workers=int(cfg.workers),
            drop_last=True, pin_memory=True)
        return train_dataset, train_loader


    def get_test_val_data(self):
        cfg = self.cfg

        if isinstance(cfg.test_val_data_dir, list):
            dataset_list = []
            loader_list = []

            if len(cfg.train_data_dir) > 0:
                for data_dir_ in cfg.test_val_data_dir: # Обычные изображния
                    print('collect dataset: '+data_dir_)
                    el_index = cfg.test_val_data_dir.index(data_dir_)
                    data_annotations_file = cfg.test_val_data_annotations_file[el_index]

                    test_val_dataset = self.load_dataset(cfg=cfg,
                                        isTextZoom=False,
                                        data_dir=data_dir_,
                                        data_annotations_file=data_annotations_file,
                                        isEval=True,
                                        # voc_type=cfg.voc_type,
                                        # max_len=cfg.max_len,
                                        ).load_dataset()
                    dataset_list.append(test_val_dataset) # создаётся объект класса loadDataset

                    batch_size = min(cfg.batch_size_val, len(test_val_dataset))
                    # batch_size = min(int(self.batch_size / 3), int(len(test_val_dataset) / 3))
                    test_val_loader = torch.utils.data.DataLoader(
                        test_val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=int(cfg.workers),drop_last=False, pin_memory=True)
                    loader_list.append(test_val_loader)

            if len(cfg.test_val_textzoom_data_dir)>0:
                for data_dir_ in cfg.test_val_textzoom_data_dir: # TextZoom
                    print('collect dataset: '+data_dir_)
                    test_val_dataset = self.load_dataset(cfg=cfg,
                                          isTextZoom=True,
                                          data_dir=data_dir_,
                                          isEval=True,
                                          # voc_type=cfg.voc_type,
                                          # max_len=cfg.max_len,
                                          ).load_dataset()
                    dataset_list.append(test_val_dataset) # создаётся объект класса loadDataset

                    batch_size = min(cfg.batch_size_val, len(test_val_dataset))
                    # batch_size = min(int(self.batch_size / 3), int(len(test_val_dataset) / 3))
                    test_val_loader = torch.utils.data.DataLoader(
                        test_val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=int(cfg.workers),drop_last=False, pin_memory=True)
                    loader_list.append(test_val_loader)
        else:
            raise TypeError('check trainRoot')

        return dataset_list, loader_list


    def generator_init(self):
        cfg = self.cfg
        optimizer = None
        scheduler = None
        scheduler_warmup = None
        if self.args.arch == 'wangji':
            model = wangji.Wangji(cfg, scale_factor=cfg.scale_factor)
            epoch = 0
            if self.cfg.recognizer == 'transformer':
                image_crit = text_focus_loss.TextFocusLoss(self.args, cfg=self.cfg)
            elif self.cfg.recognizer == 'lstm':
                image_crit = wangji_loss.WangjiLoss(args=self.args, cfg=self.cfg)
            else:
                raise exceptions.WrongRecognizer            
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
                    checkpoint = torch.load(self.resume)
                    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict_G'], strict=False)
                    epoch = checkpoint['info']['epochs']
                    if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                        optimizer = self.optimizer_init(model)
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                        scheduler = self.scheduler_init(optimizer)
                        scheduler.load_state_dict(checkpoint['scheduler'])
                    if 'scheduler_warmup' in checkpoint and checkpoint['scheduler_warmup'] is not None:
                        scheduler_warmup = self.scheduler_warmup_init(optimizer, epoch)
                        scheduler_warmup.load_state_dict(checkpoint['scheduler_warmup'])
                        epoch += 1
                else:
                    model.load_state_dict(
                        {'module.' + k: v for k, v in checkpoint['model'].items()})
                    optimizer = self.optimizer_init(model)
                    optimizer.load_state_dict(
                        {'module.' + k: v for k, v in checkpoint['optimizer'].items()})
        return {'model': model, 'crit': image_crit, 'optimizer': optimizer, 'scheduler': scheduler, 'scheduler_warmup': scheduler_warmup, 'last_epoch':epoch}


    def optimizer_init(self, model):
        cfg = self.cfg
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999), amsgrad=True, weight_decay=self.cfg.adam_weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0)
        return optimizer


    def scheduler_init(self, optimizer):
        cfg = self.cfg
        scheduler_plateau_patience = cfg.scheduler_plateau_patience
        scheduler_plateau_cooldown = cfg.scheduler_plateau_cooldown
        scheduler_plateau_factor = cfg.scheduler_plateau_factor
        scheduler = ReduceLROnPlateau(optimizer, 'min',
            patience=scheduler_plateau_patience, cooldown=scheduler_plateau_cooldown,
            min_lr=float(self.cfg.min_lr), verbose=True, factor=scheduler_plateau_factor)
        return scheduler


    def scheduler_warmup_init(self, optimizer, epoch):
        cfg = self.cfg
        scheduler_warmup_epoch = cfg.scheduler_plateau_cooldown
        scheduler_warmup = WarmupScheduler(optimizer, warmup_epoch=scheduler_warmup_epoch, last_epoch=epoch)
        return scheduler_warmup


    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.cfg
        aster_info = AsterInfo(cfg.voc_type)
        model_path = self.cfg.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model, aster_info
        
    @staticmethod
    def parse_crnn_data(imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor


    def MORAN_init(self):
        cfg = self.cfg
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(nc=1,
                            nclass=len(alphabet.split(':')),
                            nh=256,
                            targetH=32,
                            targetW=100,
                            BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor',
                            maxBatch=cfg.batch_size,
                            CUDA=True)
        model_path = self.cfg.moran_pretrained
        self.logging.info('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    @staticmethod
    def parse_moran_data(imgs_input, converter_moran):
        batch_size = imgs_input.shape[0]
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def Aster_init(self):
        cfg = self.cfg
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.cfg.rec_pretrained)['state_dict'])
        self.logging.info('load pred_trained aster model from %s' % self.cfg.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        return aster, aster_info

    @staticmethod
    def parse_aster_data(cfg, device, imgs_input):
        # cfg = self.cfg
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


    def make_writer(self):
        self.writer = SummaryWriter('checkpoint/{}'.format(self.log_dir_name))


    def make_multi_writer(self):
        self.multi_writer = SummaryWriter('checkpoint/{}'.format(self.log_dir_name))
        layout = {
            "MultiGraphs": {
                "MultiLoss": ["Multiline", ["MultiLoss/train", "MultiLoss/validation"]],
            },
        }
        self.multi_writer.add_custom_scalars(layout)


    def save_checkpoint(self, model, optimizer, epoch, iters, follow_metric_name, best_history_metric_values, best_model_info, is_best, exp_name, scheduler, scheduler_warmup):
        # ckpt_path = os.path.join('checkpoint', exp_name, self.vis_dir)
        ckpt_path = os.path.join('checkpoint', exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scheduler_warmup': scheduler_warmup.state_dict(),
            'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'up_scale_factor': self.scale_factor},
            'follow_metric_name': follow_metric_name,
            'best_history_res': best_history_metric_values,
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in model.parameters()]),
            'cfg': self.cfg,
            # 'converge': converge_list
        }
        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
        else:
            torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))



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