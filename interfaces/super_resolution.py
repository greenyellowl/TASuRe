import re
# from this import d
from turtle import update
from pyparsing import srange
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.cuda.amp import autocast, GradScaler

from interfaces import base
import copy
import logging
from datetime import datetime
import utils
from utils import util, ssim_psnr, UnNormalize, exceptions, utils_moran
import random
import os
import time
import tqdm

from utils.metrics import get_str_list, Accuracy
from utils.util import str_filt

# from warmup_scheduler import GradualWarmupScheduler
from pytorch_warmup_scheduler import WarmupScheduler # https://github.com/hysts/pytorch_warmup-scheduler
from einops import rearrange

from ctcdecode import CTCBeamDecoder

import time

from multiprocessing import Pool
from multiprocessing import set_start_method
from functools import partial

step = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0


class TextSR(base.TextBase):
    def train(self):
        print('train started')
        cfg = self.cfg

        # scheduler_plateau_patience = 2
        # scheduler_plateau_cooldown = 4
        # scheduler_warmup_epoch = scheduler_plateau_cooldown
        # scheduler_plateau_factor = 0.8

        config_txt = f"""Возможно номальное обучение  \n
                         width: {cfg.width}  \n
                         height: {cfg.height}  \n
                         batch_size: {cfg.batch_size}  \n
                         epochs: {cfg.epochs}  \n
                         lr: {cfg.lr}  \n
                         scale_factor: {cfg.scale_factor}  \n
                         workers: {cfg.workers}  \n
                         fp16: {cfg.fp16}  \n
                           \n
                         resume: {cfg.resume}  \n
                           \n
                         LOSS:  \n
                         lambda_mse: {cfg.lambda_mse}  \n
                         lambda_ctc: {cfg.lambda_ctc}  \n
                         clip_grad_norm_: {cfg.clip_grad_norm_}  \n
                         adam_weight_decay: {cfg.adam_weight_decay}  \n
                           \n
                         MODEL STRUCTURE:  \n
                         convNext_type: {cfg.convNext_type}  \n
                         acc_best_model: {cfg.acc_best_model}  \n
                         rec_best_model_save: {cfg.rec_best_model_save}  \n
                         recognizer: {cfg.recognizer}  \n
                         num_lstm_layers: {cfg.num_lstm_layers}  \n
                         lstm_bidirectional: {cfg.lstm_bidirectional}  \n
                         recognizer_input: {cfg.recognizer_input}  \n
                         recognizer_input_convnext: {cfg.recognizer_input_convnext}  \n
                         moran_att: {cfg.moran_att}  \n
                         transpose_upsample: {cfg.transpose_upsample}  \n
                         transpose_type: {cfg.transpose_type}  \n
                         batch_norm: {cfg.batch_norm}  \n
                           \n
                         BRANCHES:  \n
                         enable_sr: {cfg.enable_sr}  \n
                         enable_ConvNext: {cfg.enable_ConvNext}  \n
                         enable_rec: {cfg.enable_rec}  \n
                           \n
                         freeze_sr: {cfg.freeze_sr}  \n
                         freeze_rec: {cfg.freeze_rec}  \n
                         freeze_convnext: {cfg.freeze_convnext}  \n
                           \n
                         train_after_sr: {cfg.train_after_sr}  \n
                           \n
                         LETTERS:  \n
                         letters: {cfg.letters}  \n
                         allow_symbols: {cfg.allow_symbols}  \n
                         TGT_VOCAB_SIZE: {cfg.TGT_VOCAB_SIZE}  \n
                         FULL_VOCAB_SIZE: {cfg.FULL_VOCAB_SIZE}  \n
                           \n
                         SCHEDULER:  \n
                         scheduler_plateau_patience: {cfg.scheduler_plateau_patience}  \n
                         scheduler_plateau_cooldown: {cfg.scheduler_plateau_cooldown}  \n
                         scheduler_warmup_epoch: {cfg.scheduler_plateau_cooldown}  \n
                         scheduler_plateau_factor: {cfg.scheduler_plateau_factor}  \n
                         min_lr: {cfg.min_lr}  \n
                           \n
                         DATASETS:  \n
                         train_data_dir: {cfg.train_data_dir}  \n
                         train_data_annotations_file: {cfg.train_data_annotations_file}  \n
                         train_data_textzoom_dir: {cfg.train_data_textzoom_dir}  \n
                         test_val_data_dir: {cfg.test_val_data_dir}  \n
                         test_val_data_annotations_file: {cfg.test_val_data_annotations_file}  \n
                         test_val_textzoom_data_dir: {cfg.test_val_textzoom_data_dir}  \n
                         enable_lr_albumentations: {cfg.enable_lr_albumentations}  \n
                         overfitting: {cfg.overfitting}  \n"""
        self.writer.add_text('config', config_txt)

        train_dataset, train_loader = self.get_train_data()
        test_val_dataset_list, test_val_loader_list = self.get_test_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        num_params = util.get_num_params(model)
        self.writer.add_text('model_num_params', num_params)
        
        del model_dict['model']
        del model_dict['crit']

        # optimizer

        optimizer = model_dict['optimizer']
        del model_dict['optimizer']

        if optimizer is None:
            optimizer = self.optimizer_init(model)

        scaler = GradScaler()

        # scheduler

        scheduler_plateau = model_dict['scheduler']
        scheduler_warmup = model_dict['scheduler_warmup']
        del model_dict['scheduler']
        del model_dict['scheduler_warmup']

        if scheduler_plateau is None:
            scheduler_plateau_patience = cfg.scheduler_plateau_patience
            scheduler_plateau_cooldown = cfg.scheduler_plateau_cooldown
            scheduler_plateau_factor = cfg.scheduler_plateau_factor
            
            scheduler_plateau = ReduceLROnPlateau(optimizer, 'min',
                patience=scheduler_plateau_patience, cooldown=scheduler_plateau_cooldown,
                min_lr=float(self.cfg.min_lr), verbose=True, factor=scheduler_plateau_factor)
            
        if scheduler_warmup is None:            
            scheduler_warmup_epoch = cfg.scheduler_plateau_cooldown
            scheduler_warmup = WarmupScheduler(optimizer, warmup_epoch=scheduler_warmup_epoch)

        best_history_follow_metric_values = dict()
        for val_loader in test_val_loader_list:
            data_name = val_loader.dataset.dataset_name
            best_history_follow_metric_values[data_name] = {'epoch': -1, 'value': None}
        min_val_loss = 99999
        
        best_sum_follow_metric_value = None

        metrics_dict_datasets = {} # для сохранения первых чекпоинтов

        print(len(train_loader))

        # global step

        loss = 99999
        last_epoch = model_dict['last_epoch']
        del model_dict

        for epoch in tqdm.tqdm(range(last_epoch, cfg.epochs), desc='training'):
            scheduler_plateau.step(loss)
            scheduler_warmup.step()
            spend_time_epoch = ''
            
            pbar = tqdm.tqdm((enumerate(train_loader)), leave = False, desc='batch', total=len(train_loader))
            for j, data in pbar:
                spend_time = ''
                iters = len(train_loader) * epoch + j
                if not self.cfg.test_only:
                    start_batch_time = time.time() * 1000
                    spend_time = ''
                    start_time = time.time() * 1000
                    model.train()
                    # for p in model.parameters():
                    #     assert p.requires_grad == True
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'first block '+str(duration)+'  \n'

                    # Получение данных из датасетов

                    start_time = time.time() * 1000
                    images_hr, images_lr, label_len, label_strs, dataset_name = data
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Получение данных из датасетов '+str(duration)+'  \n'

                    # Прогон модели

                    length = ''
                    text = ''
                    t, l = self.converter_moran.encode(label_strs, scanned=True)
                    text = torch.LongTensor(cfg.batch_size * 5)
                    length = torch.IntTensor(cfg.batch_size)
                    utils_moran.loadData(text, t)
                    utils_moran.loadData(length, l)
                    
                    start_time = time.time() * 1000
                    if self.cfg.fp16:
                        with autocast():
                            images_sr, tag_scores = model(images_lr, length, text)
                    else:
                        images_sr, tag_scores = model(images_lr, length, text)
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Прогон модели '+str(duration)+'  \n'

                    # Лоссы
                    
                    start_time = time.time() * 1000
                    if cfg.recognizer == 'transformer':
                        loss, mse_loss, attention_loss, recognition_loss, word_decoder_result = image_crit(images_sr, images_hr, label_strs)
                        self.writer.add_scalar('loss/loss', loss, iters)
                        self.writer.add_scalar('loss/mse_loss', mse_loss, iters)
                        self.writer.add_scalar('loss/attention_loss', attention_loss, iters)
                        self.writer.add_scalar('loss/recognition_loss', recognition_loss, iters)
                    elif cfg.recognizer == 'lstm':
                        if self.cfg.fp16:
                            with autocast():
                                loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)
                        else:
                            loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)
                        self.writer.add_scalar('loss/loss', loss, iters)
                        self.writer.add_scalar('loss/mse_loss', mse_loss, iters)
                        self.writer.add_scalar('loss/ctc_loss', ctc_loss, iters)
                    
                        self.multi_writer.add_scalar('Multiloss/train', loss, iters)
                    else:
                        raise exceptions.WrongRecognizer
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Лоссы '+str(duration)+'  \n'

                    
                    # Loss и Optimizer

                    start_time = time.time() * 1000
                    optimizer.zero_grad()

                    if self.cfg.fp16:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if epoch <= cfg.cnt_first_epochs:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm_first_epochs)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm_)
                    if self.cfg.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    total_norm = 0
                    # i = 0
                    for p in model.parameters():
                        # i += 1
                        if p.requires_grad:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    self.writer.add_scalar('loss/grad_norm', total_norm, iters)
                    # print(i)
                    # quit()
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Loss и Optimizer '+str(duration)+'  \n'


                    # Логгирование лосов

                    start_time = time.time() * 1000
                    for ind, param_group in enumerate(optimizer.param_groups):
                        # lr = param_group['lr']
                        lr = scheduler_warmup.get_last_lr()[0]
                        self.writer.add_scalar('loss/learning rate', lr, epoch)
                        break
                    
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Логгирование лосов '+str(duration)+'  \n'

                    
                    # Вывод лоссов и ЛР в консоль

                    start_time = time.time() * 1000
                    if iters % cfg.displayInterval == 0:
                        
                        if cfg.recognizer == 'transformer':
                            loss, mse_loss, attention_loss, recognition_loss, word_decoder_result = image_crit(images_sr, images_hr, label_strs)
                            info_string = f"loss={float(loss.data):03.3f} | " \
                                    f"mse_loss={float(mse_loss):03.3f} | " \
                                    f"attention_loss={float(attention_loss):03.3f} | " \
                                    f"recognition_loss={recognition_loss:03.3f} | " \
                                    f"learning rate={lr:.10f}"
                        elif cfg.recognizer == 'lstm':
                            info_string = f"loss={float(loss.data):03.3f} | " \
                                    f"mse_loss={float(mse_loss):03.3f} | " \
                                    f"ctc_loss={float(ctc_loss):03.3f} | " \
                                    f"learning rate={lr:.10f}"
                        else:
                            raise exceptions.WrongRecognizer

                        pbar.set_description(info_string)
                    
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Вывод лоссов и ЛР в консоль '+str(duration)+'  \n'

                    
                    # Вывод изображений в TensorBoard на первом батче в эпохе
                    start_time = time.time() * 1000
                    if j == 1 and self.cfg.enable_sr:
                        index = random.randint(0, images_lr.shape[0]-1)
                        dataset = dataset_name[index]
                        show_lr = images_lr[index, ...].clone().detach()
                        show_hr = images_hr[index, ...].clone().detach()
                        show_sr = images_sr[index, ...].clone().detach()

                        # un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_lr_image_first_batch', torch.clamp(show_lr, min=0, max=1), iters)
                        self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_sr_image_first_batch', torch.clamp(show_sr, min=0, max=1), iters)
                        self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_hr_image_first_batch', torch.clamp(show_hr, min=0, max=1), iters)
                    
                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += 'Вывод изображений в TensorBoard на первом батче в эпохе '+str(duration)+'  \n'


                # ВАЛИДАЦИЯ

                spend_time_val = ''
                torch.cuda.empty_cache()
                metrics_dict_datasets = {}
                metrics_dict = {}
                start_time = time.time() * 1000
                if (j == len(train_loader)-1 and self.cfg.eval) or self.cfg.test_only:
                    if self.cfg.test_only:
                        torch.manual_seed(1234)
                    print('\n')
                    print('\n')
                    print('===================================================VALIDATION===================================================')

                    current_follow_metric_dict = {}

                    metrics_dict_datasets = {}

                    if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        aster_sr_accuracy_sum = 0
                        aster_sr_lev_dis_relation_avg_sum = 0
                        aster_lr_accuracy_sum = 0
                        aster_lr_lev_dis_relation_avg_sum = 0
                        aster_hr_accuracy_sum = 0
                        aster_hr_lev_dis_relation_avg_sum = 0

                    if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        crnn_sr_accuracy_sum = 0
                        crnn_sr_lev_dis_relation_avg_sum = 0
                        crnn_lr_accuracy_sum = 0
                        crnn_lr_lev_dis_relation_avg_sum = 0
                        crnn_hr_accuracy_sum = 0
                        crnn_hr_lev_dis_relation_avg_sum = 0

                    if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        moran_sr_accuracy_sum = 0
                        moran_sr_lev_dis_relation_avg_sum = 0
                        moran_lr_accuracy_sum = 0
                        moran_lr_lev_dis_relation_avg_sum = 0
                        moran_hr_accuracy_sum = 0
                        moran_hr_lev_dis_relation_avg_sum = 0

                    ctc_sr_accuracy_sum = 0
                    ctc_sr_lev_dis_relation_avg_sum = 0

                    psnr_avg_sum = 0
                    ssim_avg_sum = 0

                    cnt = 0
                    torch.cuda.empty_cache()
                    for k, val_loader in enumerate(test_val_loader_list):
                        dataset_name = val_loader.dataset.dataset_name

                        print('evaluation %s' % dataset_name)

                        aster = None
                        aster_info = None
                        crnn = None
                        moran = None

                        metrics_dict, spend_time_val = self.eval(model, val_loader, image_crit, iters, aster, crnn, moran, aster_info, dataset_name, iters, epoch, min_val_loss)
                        metrics_dict_datasets[dataset_name] = metrics_dict

                        # Сохраняем значения отслеживаемой метки из каждого датасета
                        follow_metric_name, _, _ = util.get_follow_metric_name(self.cfg)
                        current_follow_metric_dict[dataset_name] = metrics_dict[follow_metric_name]
                        
                        # Обновляем словарь с лучими значениями отслеживаемой метрики в каждом датасете
                        best_history_follow_metric_values = self.update_best_metric(metrics_dict, best_history_follow_metric_values, dataset_name, epoch)

                        # Рассчёт средних метрик за эпоху
                        if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                            aster_sr_accuracy_sum += metrics_dict['aster_sr_accuracy']
                            aster_sr_lev_dis_relation_avg_sum += metrics_dict['aster_sr_lev_dis_relation_avg']
                            aster_lr_accuracy_sum += metrics_dict['aster_lr_accuracy']
                            aster_lr_lev_dis_relation_avg_sum += metrics_dict['aster_lr_lev_dis_relation_avg']
                            aster_hr_accuracy_sum += metrics_dict['aster_hr_accuracy']
                            aster_hr_lev_dis_relation_avg_sum += metrics_dict['aster_hr_lev_dis_relation_avg']

                        if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                            crnn_sr_accuracy_sum += metrics_dict['crnn_sr_accuracy']
                            crnn_sr_lev_dis_relation_avg_sum += metrics_dict['crnn_sr_lev_dis_relation_avg']
                            crnn_lr_accuracy_sum += metrics_dict['crnn_lr_accuracy']
                            crnn_lr_lev_dis_relation_avg_sum += metrics_dict['crnn_lr_lev_dis_relation_avg']
                            crnn_hr_accuracy_sum += metrics_dict['crnn_hr_accuracy']
                            crnn_hr_lev_dis_relation_avg_sum += metrics_dict['crnn_hr_lev_dis_relation_avg']

                        if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                            moran_sr_accuracy_sum += metrics_dict['moran_sr_accuracy']
                            moran_sr_lev_dis_relation_avg_sum += metrics_dict['moran_sr_lev_dis_relation_avg']
                            moran_lr_accuracy_sum += metrics_dict['moran_lr_accuracy']
                            moran_lr_lev_dis_relation_avg_sum += metrics_dict['moran_lr_lev_dis_relation_avg']
                            moran_hr_accuracy_sum += metrics_dict['moran_hr_accuracy']
                            moran_hr_lev_dis_relation_avg_sum += metrics_dict['moran_hr_lev_dis_relation_avg']

                        ctc_sr_accuracy_sum += metrics_dict['ctc_sr_accuracy']
                        ctc_sr_lev_dis_relation_avg_sum += metrics_dict['ctc_sr_lev_dis_relation_avg']

                        psnr_avg_sum += metrics_dict['psnr_avg']
                        ssim_avg_sum += metrics_dict['ssim_avg']

                        cnt +=1
                    
                    # Сохранение модели
                    if self.cfg.saveBestModel:
                        best_sum_follow_metric_value = self.save_best_model(follow_metric_name, current_follow_metric_dict, best_history_follow_metric_values, 
                                                                            best_sum_follow_metric_value, metrics_dict_datasets, model, optimizer, scheduler_plateau, scheduler_warmup, iters, epoch)
                  
                # Закончилась эпоха
                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += 'ВАЛИДАЦИЯ '+str(duration)+'  \n'
                spend_time += spend_time_val +'  \n'
     
                # Сохранение чек-поинта
                start_time = time.time() * 1000 
                if iters % len(train_loader)-1 == 0 and iters != 0:
                    follow_metric_name, _, _ = util.get_follow_metric_name(self.cfg)
                    best_model_info = metrics_dict_datasets
                    self.save_checkpoint(model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        iters=iters,
                        follow_metric_name=follow_metric_name,
                        best_history_metric_values=best_history_follow_metric_values,
                        best_model_info=best_model_info,
                        is_best=False,
                        exp_name=self.log_dir_name,
                        scheduler=scheduler_plateau,
                        scheduler_warmup=scheduler_warmup)
                
                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += 'Сохранение чек-поинта '+str(duration)+'  \n'

                end_batch_time = time.time() * 1000
                duration_batch = end_batch_time - start_batch_time
                spend_time_epoch += 'batch '+str(j)+'  \n'+spend_time+'  \nduration batch '+str(duration_batch)

                # step += 1
            
            self.writer.add_text(f'spend_time/{epoch}_epoch', spend_time_epoch)

            # Лоссы по эпохам
                
            start_time = time.time() * 1000
            if cfg.recognizer == 'transformer':
                self.writer.add_scalar('loss_epoch/loss', loss, epoch)
                self.writer.add_scalar('loss_epoch/mse_loss', mse_loss, epoch)
                self.writer.add_scalar('loss_epoch/attention_loss', attention_loss, epoch)
                self.writer.add_scalar('loss_epoch/recognition_loss', recognition_loss, epoch)
            elif cfg.recognizer == 'lstm':
                self.writer.add_scalar('loss_epoch/loss', loss, epoch)
                self.writer.add_scalar('loss_epoch/mse_loss', mse_loss, epoch)
                self.writer.add_scalar('loss_epoch/ctc_loss', ctc_loss, epoch)
            else:
                raise exceptions.WrongRecognizer
            end_time = time.time() * 1000
            duration = end_time - start_time
            spend_time += 'Лоссы '+str(duration)+'  \n'

            if cfg.eval:
                # Метрики по эпохам
                if self.cfg.enable_sr or self.cfg.train_after_sr or (not self.cfg.enable_rec and self.cfg.recognizer == 'transformer'): 
                    self.writer.add_scalar(f'other_avg/psnr_avg', psnr_avg_sum / cnt, epoch)
                    self.writer.add_scalar(f'other_avg/ssim_avg', ssim_avg_sum / cnt, epoch)
                    
                    if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        self.writer.add_scalar(f'accuracy_avg/aster_sr_accuracy', aster_sr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/aster_sr_lev_dis_relation_avg', aster_sr_lev_dis_relation_avg_sum / cnt, epoch)
                        self.writer.add_scalar(f'accuracy_avg/aster_lr_accuracy', aster_lr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/aster_lr_lev_dis_relation_avg', aster_lr_lev_dis_relation_avg_sum / cnt, epoch)
                        self.writer.add_scalar(f'accuracy_avg/aster_hr_accuracy', aster_hr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/aster_hr_lev_dis_relation_avg', aster_hr_lev_dis_relation_avg_sum / cnt, epoch)

                    if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        self.writer.add_scalar(f'accuracy_avg/crnn_sr_accuracy', crnn_sr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/crnn_sr_lev_dis_relation_avg', crnn_sr_lev_dis_relation_avg_sum / cnt, epoch)
                        self.writer.add_scalar(f'accuracy_avg/crnn_lr_accuracy', crnn_lr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/crnn_lr_lev_dis_relation_avg', crnn_lr_lev_dis_relation_avg_sum / cnt, epoch)
                        self.writer.add_scalar(f'accuracy_avg/crnn_hr_accuracy', crnn_hr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/crnn_hr_lev_dis_relation_avg', crnn_hr_lev_dis_relation_avg_sum / cnt, epoch)
                    
                    if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        self.writer.add_scalar(f'accuracy_avg/moran_sr_accuracy', moran_sr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/moran_sr_lev_dis_relation_avg', moran_sr_lev_dis_relation_avg_sum / cnt, epoch)
                        self.writer.add_scalar(f'accuracy_avg/moran_lr_accuracy', moran_lr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/moran_lr_lev_dis_relation_avg', moran_lr_lev_dis_relation_avg_sum / cnt, epoch)
                        self.writer.add_scalar(f'accuracy_avg/moran_hr_accuracy', moran_hr_accuracy_sum / cnt * 100, epoch)
                        self.writer.add_scalar(f'other_avg/moran_hr_lev_dis_relation_avg', moran_hr_lev_dis_relation_avg_sum / cnt, epoch)
    
                if self.cfg.enable_rec:
                    self.writer.add_scalar(f'accuracy_avg/ctc_sr_accuracy', ctc_sr_accuracy_sum / cnt * 100, epoch)
                    self.writer.add_scalar(f'other_avg/ctc_sr_lev_dis_relation_avg', ctc_sr_lev_dis_relation_avg_sum / cnt, epoch)
            
            
            if self.cfg.test_only:
                quit()

    @staticmethod
    def get_crnn_pred(outputs):
        alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
        predict_result = []
        for output in outputs:
            max_index = torch.max(output, 1)[1]
            out_str = ""
            last = ""
            for i in max_index:
                if alphabet[i] != last:
                    if i != 0:
                        out_str += alphabet[i]
                        last = alphabet[i]
                    else:
                        last = ""
            predict_result.append(out_str)
        return predict_result

    
    def ctc_decode(self, tag_scores):
        labels = self.FullVocabList
        num_processes = 1
        decoder = CTCBeamDecoder(
            labels,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=num_processes,
            blank_id=0,
            log_probs_input=True
        )
        tag_scores = rearrange(tag_scores, 't b l -> b t l') # N_TIMESTEPS x BATCHSIZE x N_LABELS -> BATCHSIZE x N_TIMESTEPS x N_LABELS
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(tag_scores)
        strings = []
        batch_size = tag_scores.shape[0]
        for i in range(batch_size):
            dig_string = beam_results[i][0][:out_lens[i][0]]
            string = ''
            for dig in dig_string:
                string += list(self.FullVocab.keys())[list(self.FullVocab.values()).index(str(int(dig)))]
            
            strings.append(string)
        
        return strings

    @staticmethod
    def levenshtein_distance(a, b): # https://ru.wikibooks.org/wiki/Реализации_алгоритмов/Расстояние_Левенштейна#Python
        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n, m)) space
            a, b = b, a
            n, m = m, n

        current_row = range(n + 1)  # Keep current and previous row, not entire matrix
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[n]

    @staticmethod
    def calculate_acc_lev_dis(cfg, pred_str_sr, label_strs, lev_dis_list, lev_dis_relation_list):
        cnt = 0
        n_correct = 0
        pred_text = 'target -> pred  \n'
        for pred, target in zip(pred_str_sr, label_strs):                
            pred_text += target+' -> '+pred+"  \n"
            lev_dis = TextSR.levenshtein_distance(target, pred)
            lev_dis_list.append(lev_dis)
            lev_dis_relation_list.append(lev_dis / len(target) if len(target) > 0 else lev_dis)
            if pred == target:
                n_correct += 1
            cnt += 1
        return n_correct, cnt, lev_dis_list, lev_dis_relation_list, pred_text

    @staticmethod
    def calculate_aster_pred(images, aster_lev_dis_list, aster_lev_dis_relation_list, cfg, device, label_strs, aster, aster_info):
        dict_sr = TextSR.parse_aster_data(cfg, device, images[:, :3, :, :])
        output_sr = aster(dict_sr)
        pred_rec_sr = output_sr['output']['pred_rec']
        pred_str_sr, _ = get_str_list(pred_rec_sr, dict_sr['rec_targets'], dataset=aster_info)
        
        return TextSR.calculate_acc_lev_dis(cfg, pred_str_sr, label_strs, aster_lev_dis_list, aster_lev_dis_relation_list), pred_str_sr
        
    @staticmethod
    def calculate_crnn_pred(images, crnn_lev_dis_list, crnn_lev_dis_relation_list, cfg, label_strs, crnn):
        crnn_dict = TextSR.parse_crnn_data(images[:, :3, :, :])
        crnn_output = crnn(crnn_dict)
        crnn_outputs = crnn_output.permute(1, 0, 2).contiguous()
        crnn_predict_result = TextSR.get_crnn_pred(crnn_outputs)
        
        return TextSR.calculate_acc_lev_dis(cfg, crnn_predict_result, label_strs, crnn_lev_dis_list, crnn_lev_dis_relation_list), crnn_predict_result
    
    @staticmethod
    def calculate_moran_pred(images, moran_lev_dis_list, moran_lev_dis_relation_list, cfg, label_strs, moran, converter_moran):
        moran_input = TextSR.parse_moran_data(images[:, :3, :, :], converter_moran)
        moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                debug=True)
        preds, preds_reverse = moran_output[0]
        _, preds = preds.max(1)
        sim_preds = converter_moran.decode(preds.data, moran_input[1].data)
        pred_str_sr = [pred.split('$')[0] for pred in sim_preds]  
        
        return TextSR.calculate_acc_lev_dis(cfg, pred_str_sr, label_strs, moran_lev_dis_list, moran_lev_dis_relation_list), pred_str_sr


    def calculate_ctc_pred(self, tag_scores, label_strs, ctc_lev_dis_list, ctc_lev_dis_relation_list):
        ctc_cnt = 0
        ctc_n_correct = 0
        ctc_pred_text = 'target -> ctc_decode_string  \n'
        ctc_decode_strings = self.ctc_decode(tag_scores)
        for ctc_decode_string, target in zip(ctc_decode_strings, label_strs):
            ctc_pred_text += target+' -> '+ctc_decode_string+"  \n"
            lev_dis = self.levenshtein_distance(target, ctc_decode_string)
            ctc_lev_dis_list.append(lev_dis)
            ctc_lev_dis_relation_list.append(lev_dis / len(target) if len(target) > 0 else lev_dis)
            if ctc_decode_string == target:
                ctc_n_correct += 1
            ctc_cnt += 1

        return ctc_n_correct, ctc_cnt, ctc_lev_dis_list, ctc_lev_dis_relation_list, ctc_pred_text, ctc_decode_strings
    
    
    def update_best_metric(self, metrics_dict, best_history_metric_values, dataset_name, epoch):
        
        metric_name, metric_print, direction = util.get_follow_metric_name(self.cfg)
        # acc_metric_name, metric_print, direction
        
        current_metric = metrics_dict[metric_name]
        best_metric = best_history_metric_values[dataset_name]['value']

        if best_metric is None:
            best_metric = current_metric
            update = True
        else:
            update = current_metric > best_metric if direction=='max' else current_metric < best_metric

        if update:
            best_history_metric_values[dataset_name]['value'] = float(current_metric)
            best_history_metric_values[dataset_name]['epoch'] = epoch
            if self.cfg.rec_best_model_save == 'acc' and self.cfg.enable_rec == True:
                current_metric = f'{float(current_metric * 100):.2f} %'
            else:
                current_metric = f'{float(current_metric):.5f}'

            print(f'╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟupdate best_{metric_print}_{dataset_name}. {best_metric} -> {float(current_metric):.4f}')
        else:
            print(f'not update best_{metric_print}_{dataset_name} = {best_metric}')

        return best_history_metric_values

    
    def save_best_model(self,
                        follow_metric_name,
                        current_metric_dict,
                        best_history_metric_values, 
                        best_sum_metric_value, 
                        metrics_dict_datasets, 
                        model, 
                        optimizer, 
                        scheduler, 
                        scheduler_warmup, 
                        iters, 
                        epoch):
        follow_metric_name, metric_print, direction = util.get_follow_metric_name(self.cfg)

        current_metric_sum = sum(current_metric_dict.values())
        if best_sum_metric_value is None:
            best_sum_metric_value = current_metric_sum
            update = True
        else:
            update = current_metric_sum > best_sum_metric_value if direction=='max' else current_metric_sum < best_sum_metric_value
        
        if  update:

            best_sum_metric_value = sum(current_metric_dict.values())

            best_model_info = metrics_dict_datasets

            print('saving best model')
            self.save_checkpoint(model, optimizer, epoch, iters, follow_metric_name, best_history_metric_values, best_model_info, True,
                                self.log_dir_name, scheduler, scheduler_warmup)
        
        return best_sum_metric_value
    
    
    def calc_print_external_recognizer(self,
                                        name,
                                        n_correct_sr_sum,
                                        sr_lev_dis_relation_list,
                                        cnt_images,
                                        metric_dict,
                                        dataset_name,
                                        epoch,
                                        calc_lr=False,
                                        n_correct_lr_sum=None,
                                        lr_lev_dis_relation_list=None,
                                        calc_hr=False,
                                        n_correct_hr_sum=None,
                                        hr_lev_dis_relation_list=None):
        # ACC SR

        sr_accuracy = round(n_correct_sr_sum / cnt_images, 4)
        metric_dict[name+'_sr_accuracy'] = sr_accuracy
        self.writer.add_scalar(f'{name.upper()}/{dataset_name}/sr_accuracy', sr_accuracy * 100, epoch)
        # print(name+'_sr_accuracy: %.2f%%' % (sr_accuracy * 100))

        # LEV DIS SR
        
        sr_lev_dis_relation_avg = round(sum(sr_lev_dis_relation_list) / cnt_images, 4)
        metric_dict[name+'_sr_lev_dis_relation_avg'] = sr_lev_dis_relation_avg
        self.writer.add_scalar(f'{name.upper()}/{dataset_name}/sr_lev_dis_relation_avg', sr_lev_dis_relation_avg, epoch)
        
        print(f'{name}_sr_accuracy {float(sr_accuracy):.2f} | {name}_sr_lev_dis_relation_avg {float(sr_lev_dis_relation_avg):.4f}\t')

        if calc_lr:
            # ACC LR

            lr_accuracy = round(n_correct_lr_sum / cnt_images, 4)
            metric_dict[name+'_lr_accuracy'] = lr_accuracy
            self.writer.add_scalar(f'{name.upper()}/{dataset_name}/lr_accuracy', lr_accuracy * 100, epoch)
            # print(name+'_lr_accuracy: %.2f%%' % (lr_accuracy * 100))

            # LEV DIS LR
            
            lr_lev_dis_relation_avg = round(sum(lr_lev_dis_relation_list) / cnt_images, 4)
            metric_dict[name+'_lr_lev_dis_relation_avg'] = lr_lev_dis_relation_avg
            self.writer.add_scalar(f'{name.upper()}/{dataset_name}/lr_lev_dis_relation_avg', lr_lev_dis_relation_avg, epoch)
            
            print(f'{name}_lr_accuracy {float(lr_accuracy):.2f} | {name}_lr_lev_dis_relation_avg {float(lr_lev_dis_relation_avg):.4f}\t')
        
        if calc_hr:
            # ACC HR

            hr_accuracy = round(n_correct_hr_sum / cnt_images, 4)
            metric_dict[name+'_hr_accuracy'] = hr_accuracy
            self.writer.add_scalar(f'{name.upper()}/{dataset_name}/hr_accuracy', hr_accuracy * 100, epoch)
            # print(name+'_hr_accuracy: %.2f%%' % (hr_accuracy * 100))

            # LEV DIS HR
            
            hr_lev_dis_relation_avg = round(sum(hr_lev_dis_relation_list) / cnt_images, 4)
            metric_dict[name+'_hr_lev_dis_relation_avg'] = hr_lev_dis_relation_avg
            self.writer.add_scalar(f'{name.upper()}/{dataset_name}/hr_lev_dis_relation_avg', hr_lev_dis_relation_avg, epoch)
            
            print(f'{name}_hr_accuracy {float(hr_accuracy):.2f} | {name}_hr_lev_dis_relation_avg {float(hr_lev_dis_relation_avg):.4f}\t')

        return metric_dict
    
    # валидация одного датасета
    def eval(self, model, val_loader, image_crit, index, aster, crnn, moran, aster_info, dataset_name, iters, epoch, min_val_loss):
        with torch.no_grad():

            spend_time = ''
            global easy_test_times
            global medium_test_times
            global hard_test_times

            start_time = time.time() * 1000

            model.eval()

            end_time = time.time() * 1000
            duration = end_time - start_time
            spend_time += '\t model.eval() '+str(duration)+'  \n'
            
            # LOSS
            loss_sum = 0
            mse_loss_sum = 0
            ctc_loss_sum = 0

            if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                # ASTER
                
                aster_n_correct_sr_sum = 0
                aster_sr_lev_dis_list = []
                aster_sr_lev_dis_relation_list = []
                aster_n_correct_lr_sum = 0
                aster_lr_lev_dis_list = []
                aster_lr_lev_dis_relation_list = []
                aster_n_correct_hr_sum = 0
                aster_hr_lev_dis_list = []
                aster_hr_lev_dis_relation_list = []
            
            if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                # CRNN

                crnn_n_correct_sr_sum = 0
                crnn_sr_lev_dis_list = []
                crnn_sr_lev_dis_relation_list = []
                crnn_n_correct_lr_sum = 0
                crnn_lr_lev_dis_list = []
                crnn_lr_lev_dis_relation_list = []
                crnn_n_correct_hr_sum = 0
                crnn_hr_lev_dis_list = []
                crnn_hr_lev_dis_relation_list = []

            if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                # MORAN
                moran_n_correct_sr_sum = 0
                moran_sr_lev_dis_list = []
                moran_sr_lev_dis_relation_list = []
                moran_n_correct_lr_sum = 0
                moran_lr_lev_dis_list = []
                moran_lr_lev_dis_relation_list = []
                moran_n_correct_hr_sum = 0
                moran_hr_lev_dis_list = []
                moran_hr_lev_dis_relation_list = []

            # CTC
            
            ctc_sr_n_correct_sum = 0
            ctc_sr_lev_dis_list = []
            ctc_sr_lev_dis_relation_list = []
            # ctc_lr_n_correct_sum = 0
            # ctc_lr_lev_dis_list = []

            psnr_batches = []
            ssim_batches = []
            cnt_images = 0

            metric_dict = {
                'aster_sr_accuracy': 0.0, 
                'aster_sr_lev_dis_relation_avg': 0.0, 
                'aster_lr_accuracy': 0.0, 
                'aster_lr_lev_dis_relation_avg': 0.0, 
                'aster_hr_accuracy': 0.0, 
                'aster_hr_lev_dis_relation_avg': 0.0, 
                'crnn_sr_accuracy': 0.0, 
                'crnn_sr_lev_dis_relation_avg': 0.0, 
                'crnn_lr_accuracy': 0.0, 
                'crnn_lr_lev_dis_relation_avg': 0.0, 
                'crnn_hr_accuracy': 0.0, 
                'crnn_hr_lev_dis_relation_avg': 0.0, 
                'moran_sr_accuracy': 0.0, 
                'moran_sr_lev_dis_relation_avg': 0.0, 
                'moran_lr_accuracy': 0.0, 
                'moran_lr_lev_dis_relation_avg': 0.0,
                'moran_hr_accuracy': 0.0, 
                'moran_hr_lev_dis_relation_avg': 0.0, 
                'ctc_sr_accuracy': 0.0, 
                'ctc_sr_lev_dis_relation_avg': 0.0, 
                'psnr_avg': 0.0, 
                'ssim_avg': 0.0, 
                'images_and_labels': []}

            pbar = tqdm.tqdm((enumerate(val_loader)), leave = False, desc='evaluation', total=len(val_loader))
            for i, data in pbar:
                info_string = ''
                start_time = time.time() * 1000
                # Получение данных из датасетов

                images_hr, images_lr, _, label_strs, _ = data

                val_batch_size = images_lr.shape[0]

                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += '\t Получение данных из датасетов '+str(duration)+'  \n'
                
                # Прогон модели

                start_time = time.time() * 1000

                length = ''
                text = ''
                t, l = self.converter_moran.encode(label_strs, scanned=True)
                text = torch.LongTensor(self.cfg.batch_size * 5)
                length = torch.IntTensor(self.cfg.batch_size)
                utils_moran.loadData(text, t)
                utils_moran.loadData(length, l)

                images_sr, tag_scores = model(images_lr, length, text)

                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += '\t Прогон модели '+str(duration)+'  \n'

                # Лоссы

                start_time = time.time() * 1000

                if self.cfg.recognizer == 'transformer':
                    loss, mse_loss, attention_loss, recognition_loss, word_decoder_result = image_crit(images_sr, images_hr, label_strs)
                elif self.cfg.recognizer == 'lstm':
                    loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)
                    loss_sum += loss
                    mse_loss_sum += mse_loss
                    ctc_loss_sum += ctc_loss
                else:
                    raise exceptions.WrongRecognizer

                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += '\t Лоссы '+str(duration)+'  \n'

                # Если включена ветка СР или модель предобучена на СР, считаем степень зашумлённости и похожести изображений
                if self.cfg.enable_sr or self.cfg.train_after_sr:

                    start_time = time.time() * 1000

                    # Вычисление PSNR и SSIM (Средние по батчу)
                                      
                    psnr_batches.append(self.cal_psnr(images_sr, images_hr))
                    ssim_batches.append(self.cal_ssim(images_sr, images_hr))

                    end_time = time.time() * 1000
                    duration = end_time - start_time
                    spend_time += '\t Вычисление PSNR и SSIM (Средние по батчу) '+str(duration)+'  \n'

                # Точность и дистанция Левенштейна
                if self.cfg.enable_sr or self.cfg.train_after_sr or (not self.cfg.enable_rec and self.cfg.recognizer == 'transformer'):

                    start_time1 = time.time() * 1000

                    aster_pred_lr = 'NONE'
                    aster_pred_sr = 'NONE'
                    aster_pred_hr = 'NONE'
                    
                    crnn_pred_lr = 'NONE'
                    crnn_pred_sr = 'NONE'
                    crnn_pred_hr = 'NONE'

                    moran_pred_lr = 'NONE'
                    moran_pred_sr = 'NONE'
                    moran_pred_hr = 'NONE'
                    
                    torch.cuda.empty_cache()
                    if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        start_time = time.time() * 1000
                        
                        pbar.set_description("calc ASTER")
                        # ASTER

                        aster, aster_info = self.Aster_init()
                        aster.eval()

                        # with Pool(3) as p:

                        #     f_calculate = partial(TextSR.calculate_aster_pred, cfg=self.cfg, device=self.device, label_strs=label_strs, aster=aster, aster_info=aster_info)
                        #     input = ((images_sr, aster_sr_lev_dis_list, aster_sr_lev_dis_relation_list),
                        #             (images_lr, aster_lr_lev_dis_list, aster_lr_lev_dis_relation_list),
                        #             (images_hr, aster_hr_lev_dis_list, aster_hr_lev_dis_relation_list))
                        #     result = list(p.starmap(f_calculate, input))
                        
                        # (aster_n_correct_sr, aster_cnt_sr, aster_sr_lev_dis_list, aster_sr_lev_dis_relation_list, aster_sr_pred_text), aster_predict_result_sr = result[0]
                        # (aster_n_correct_lr, aster_cnt_lr, aster_lr_lev_dis_list, aster_lr_lev_dis_relation_list, aster_lr_pred_text), aster_predict_result_lr = result[1]
                        # (aster_n_correct_hr, aster_cnt_hr, aster_hr_lev_dis_list, aster_hr_lev_dis_relation_list, aster_hr_pred_text), aster_predict_result_hr = result[2]
                        
                        # aster_n_correct_sr_sum += aster_n_correct_sr
                        # aster_n_correct_lr_sum += aster_n_correct_lr
                        # aster_n_correct_hr_sum += aster_n_correct_hr
                        
                        # SR
                        result = self.calculate_aster_pred(images_sr, aster_sr_lev_dis_list, aster_sr_lev_dis_relation_list, self.cfg, self.device, label_strs, aster, aster_info)
                        (aster_n_correct_sr, aster_cnt_sr, aster_sr_lev_dis_list, aster_sr_lev_dis_relation_list, aster_sr_pred_text), aster_predict_result_sr = result
                        aster_n_correct_sr_sum += aster_n_correct_sr
                        
                        # LR
                        result = self.calculate_aster_pred(images_lr, aster_lr_lev_dis_list, aster_lr_lev_dis_relation_list, self.cfg, self.device, label_strs, aster, aster_info)
                        (aster_n_correct_lr, aster_cnt_lr, aster_lr_lev_dis_list, aster_lr_lev_dis_relation_list, aster_lr_pred_text), aster_predict_result_lr = result
                        aster_n_correct_lr_sum += aster_n_correct_lr
                        
                        # HR
                        result = self.calculate_aster_pred(images_hr, aster_hr_lev_dis_list, aster_hr_lev_dis_relation_list, self.cfg, self.device, label_strs, aster, aster_info)
                        (aster_n_correct_hr, aster_cnt_hr, aster_hr_lev_dis_list, aster_hr_lev_dis_relation_list, aster_hr_pred_text), aster_predict_result_hr = result
                        aster_n_correct_hr_sum += aster_n_correct_hr

                        del aster
                        del aster_info

                        torch.cuda.empty_cache()

                        end_time = time.time() * 1000
                        duration = end_time - start_time
                        spend_time += '\t \t ASTER '+str(duration)+'  \n'

                    if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        start_time = time.time() * 1000
                        pbar.set_description("calc CRNN")
                        # CRNN

                        crnn, _ = self.CRNN_init()
                        crnn.eval()
                        
                        # with Pool(3) as p:

                        #     f_calculate = partial(TextSR.calculate_crnn_pred, cfg=self.cfg, label_strs=label_strs, crnn=crnn)
                        #     input = ((images_sr, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list),
                        #             (images_lr, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list),
                        #             (images_hr, crnn_hr_lev_dis_list, crnn_hr_lev_dis_relation_list))
                        #     result = list(p.starmap(f_calculate, input))
                        
                        # (crnn_n_correct_sr, crnn_cnt_sr, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list, crnn_sr_pred_text), crnn_predict_result_sr = result[0]
                        # (crnn_n_correct_lr, crnn_cnt_lr, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list, crnn_lr_pred_text), crnn_predict_result_lr = result[1]
                        # (crnn_n_correct_hr, crnn_cnt_hr, crnn_hr_lev_dis_list, crnn_hr_lev_dis_relation_list, crnn_hr_pred_text), crnn_predict_result_hr = result[2]

                        # crnn_n_correct_sr_sum += crnn_n_correct_sr
                        # crnn_n_correct_lr_sum += crnn_n_correct_lr
                        # crnn_n_correct_hr_sum += crnn_n_correct_hr
                        
                        # SR
                        result = self.calculate_crnn_pred(images_sr, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list, self.cfg, label_strs, crnn)
                        (crnn_n_correct_sr, crnn_cnt_sr, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list, crnn_sr_pred_text), crnn_predict_result_sr = result
                        crnn_n_correct_sr_sum += crnn_n_correct_sr
                        
                        # LR
                        result = self.calculate_crnn_pred(images_lr, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list, self.cfg, label_strs, crnn)
                        (crnn_n_correct_lr, crnn_cnt_lr, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list, crnn_lr_pred_text), crnn_predict_result_lr = result
                        crnn_n_correct_lr_sum += crnn_n_correct_lr
                        
                        # HR
                        result = self.calculate_crnn_pred(images_hr, crnn_hr_lev_dis_list, crnn_hr_lev_dis_relation_list, self.cfg, label_strs, crnn)
                        (crnn_n_correct_hr, crnn_cnt_hr, crnn_hr_lev_dis_list, crnn_hr_lev_dis_relation_list, crnn_hr_pred_text), crnn_predict_result_hr = result
                        crnn_n_correct_hr_sum += crnn_n_correct_hr

                        del crnn

                        torch.cuda.empty_cache()

                        end_time = time.time() * 1000
                        duration = end_time - start_time
                        spend_time += '\t \t CRNN '+str(duration)+'  \n'

                    if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                        start_time = time.time() * 1000
                        pbar.set_description("calc MORAN")
                        # MORAN

                        moran = self.MORAN_init()
                        moran.eval()

                        
                        # with Pool(3) as p:

                        #     f_calculate = partial(TextSR.calculate_moran_pred, cfg=self.cfg, label_strs=label_strs, moran=moran, converter_moran=self.converter_moran)
                        #     input = ((images_sr, moran_sr_lev_dis_list, moran_sr_lev_dis_relation_list),
                        #             (images_lr, moran_lr_lev_dis_list, moran_lr_lev_dis_relation_list),
                        #             (images_hr, moran_hr_lev_dis_list, moran_hr_lev_dis_relation_list))
                        #     result = list(p.starmap(f_calculate, input))
                        
                        # (moran_n_correct_sr, moran_cnt_sr, moran_sr_lev_dis_list, moran_sr_lev_dis_relation_list, moran_sr_pred_text), moran_predict_result_sr = result[0]
                        # (moran_n_correct_lr, moran_cnt_lr, moran_lr_lev_dis_list, moran_lr_lev_dis_relation_list, moran_lr_pred_text), moran_predict_result_lr = result[1]
                        # (moran_n_correct_hr, moran_cnt_hr, moran_hr_lev_dis_list, moran_hr_lev_dis_relation_list, moran_hr_pred_text), moran_predict_result_hr = result[2]

                        # moran_n_correct_sr_sum += moran_n_correct_sr
                        # moran_n_correct_lr_sum += moran_n_correct_lr
                        # moran_n_correct_hr_sum += moran_n_correct_hr
                        
                        # SR
                        result = self.calculate_moran_pred(images_sr, moran_sr_lev_dis_list, moran_sr_lev_dis_relation_list, self.cfg, label_strs, moran, self.converter_moran)
                        (moran_n_correct_sr, moran_cnt_sr, moran_sr_lev_dis_list, moran_sr_lev_dis_relation_list, moran_sr_pred_text), moran_predict_result_sr = result
                        moran_n_correct_sr_sum += moran_n_correct_sr
                        
                        # LR
                        result = self.calculate_moran_pred(images_lr, moran_lr_lev_dis_list, moran_lr_lev_dis_relation_list, self.cfg, label_strs, moran, self.converter_moran)
                        (moran_n_correct_lr, moran_cnt_lr, moran_lr_lev_dis_list, moran_lr_lev_dis_relation_list, moran_lr_pred_text), moran_predict_result_lr = result
                        moran_n_correct_lr_sum += moran_n_correct_lr
                        
                        # HR
                        result = self.calculate_moran_pred(images_hr, moran_hr_lev_dis_list, moran_hr_lev_dis_relation_list, self.cfg, label_strs, moran, self.converter_moran)
                        (moran_n_correct_hr, moran_cnt_hr, moran_hr_lev_dis_list, moran_hr_lev_dis_relation_list, moran_hr_pred_text), moran_predict_result_hr = result
                        moran_n_correct_hr_sum += moran_n_correct_hr

                        del moran

                        torch.cuda.empty_cache()

                        end_time = time.time() * 1000
                        duration = end_time - start_time
                        spend_time += '\t \t MORAN '+str(duration)+'  \n'

                else:
                    aster_predict_result_lr = None
                    aster_pred_lr = 'NONE'
                    aster_predict_result_sr = None
                    aster_pred_sr = 'NONE'
                    aster_predict_result_hr = None
                    aster_pred_hr = 'NONE'
                    
                    crnn_predict_result_lr = None
                    crnn_pred_lr = 'NONE'
                    crnn_predict_result_sr = None
                    crnn_pred_sr = 'NONE'
                    crnn_predict_result_hr = None
                    crnn_pred_hr = 'NONE'
                    
                    moran_predict_result_lr = None
                    moran_pred_lr = 'NONE'
                    moran_predict_result_sr = None
                    moran_pred_sr = 'NONE'
                    moran_predict_result_hr = None
                    moran_pred_hr = 'NONE'

                end_time = time.time() * 1000
                duration = end_time - start_time1
                spend_time += '\t Точность и дистанция Левенштейна '+str(duration)+'  \n'

                # Если включена ветка распознавания, считаем точность распознавания СТС
                start_time = time.time() * 1000
                if self.cfg.enable_rec:
                    pbar.set_description("calc CTC")
                    # CTC
                    ctc_sr_n_correct, ctc_cnt, ctc_sr_lev_dis_list, ctc_sr_lev_dis_relation_list, ctc_pred_text, ctc_decode_strings = self.calculate_ctc_pred(tag_scores, label_strs, ctc_sr_lev_dis_list, ctc_sr_lev_dis_relation_list)                    
                    ctc_sr_n_correct_sum += ctc_sr_n_correct
                else:
                    ctc_pred = 'NONE'
                    ctc_decode_strings = None

                end_time = time.time() * 1000
                duration = end_time - start_time1
                spend_time += '\t Точность распознавания СТС '+str(duration)+'  \n'
                

                # Вывод изображений и предсказаний в TensorBoard
                if i == len(val_loader) - 1:
                    start_time = time.time() * 1000
                    index = random.randint(0, images_lr.shape[0]-1)

                    # Если включена ветка распознавания
                    if self.cfg.enable_rec:
                        ground_truth = label_strs[index].replace('"', '<quot>')
                        ctc_pred = ctc_decode_strings[index].replace('"', '<quot>')

                        self.writer.add_text(f'CTC_pred/{epoch}_epoch/{dataset_name}', ctc_pred_text)
                    

                    # Если включена ветка СР или модель предобучена на СР
                    if self.cfg.enable_sr or self.cfg.train_after_sr or (not self.cfg.enable_rec and self.cfg.recognizer == 'transformer'):

                        ground_truth = label_strs[index].replace('"', '<quot>')

                        if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                            # ASTER

                            aster_pred_sr = aster_predict_result_sr[index].replace('"', '<quot>')
                            aster_pred_lr = aster_predict_result_lr[index].replace('"', '<quot>')
                            aster_pred_hr = aster_predict_result_hr[index].replace('"', '<quot>')

                            self.writer.add_text(f'ASTER_SR_pred/{epoch}_epoch/{dataset_name}', aster_sr_pred_text)
                            self.writer.add_text(f'ASTER_LR_pred/{epoch}_epoch/{dataset_name}', aster_lr_pred_text)
                            self.writer.add_text(f'ASTER_HR_pred/{epoch}_epoch/{dataset_name}', aster_hr_pred_text)

                        if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                            # CRNN

                            crnn_pred_sr = crnn_predict_result_sr[index].replace('"', '<quot>')
                            crnn_pred_lr = crnn_predict_result_lr[index].replace('"', '<quot>')
                            crnn_pred_hr = crnn_predict_result_hr[index].replace('"', '<quot>')

                            self.writer.add_text(f'CRNN_SR_pred/{epoch}_epoch/{dataset_name}', crnn_sr_pred_text)
                            self.writer.add_text(f'CRNN_LR_pred/{epoch}_epoch/{dataset_name}', crnn_lr_pred_text)
                            self.writer.add_text(f'CRNN_HR_pred/{epoch}_epoch/{dataset_name}', crnn_hr_pred_text)

                        if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                            # MORAN

                            moran_pred_sr = moran_predict_result_sr[index].replace('"', '<quot>')
                            moran_pred_lr = moran_predict_result_lr[index].replace('"', '<quot>')
                            moran_pred_hr = moran_predict_result_hr[index].replace('"', '<quot>')

                            self.writer.add_text(f'MORAN_SR_pred/{epoch}_epoch/{dataset_name}', moran_sr_pred_text)
                            self.writer.add_text(f'MORAN_LR_pred/{epoch}_epoch/{dataset_name}', moran_lr_pred_text)
                            self.writer.add_text(f'MORAN_HR_pred/{epoch}_epoch/{dataset_name}', moran_hr_pred_text)
                        
                        metric_dict['images_and_labels'].append((images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, ctc_decode_strings))

                        show_lr = images_lr[index, ...].clone().detach()
                        show_hr = images_hr[index, ...].clone().detach()
                        show_sr = images_sr[index, ...].clone().detach()                    

                        # un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        print('save display images')
                        pred_lr = aster_pred_lr if aster_pred_lr else (crnn_pred_lr if crnn_pred_lr else (moran_pred_lr if moran_pred_lr else 'NONE'))
                        pred_sr = aster_pred_sr if aster_pred_sr else (crnn_pred_sr if crnn_pred_sr else (moran_pred_sr if moran_pred_sr else 'NONE'))
                        self.writer.add_image(f'{dataset_name}/{epoch}_epoch_{index}_index_lr_image_eval_pred:{pred_lr}', torch.clamp(show_lr, min=0, max=1), iters)
                        self.writer.add_image(f'{dataset_name}/{epoch}_epoch_{index}_index_sr_image_eval_pred:{pred_sr}_ctc_pred:{ctc_pred}', torch.clamp(show_sr, min=0, max=1), iters)
                        self.writer.add_image(f'{dataset_name}/{epoch}_epoch_{index}_index_hr_image_eval_groundTruth:{ground_truth}', torch.clamp(show_hr, min=0, max=1), iters)

                    end_time = time.time() * 1000
                    duration = end_time - start_time1
                    spend_time += '\t Вывод изображений и предсказаний в TensorBoard '+str(duration)+'  \n'
                cnt_images += val_batch_size

                torch.cuda.empty_cache()

            # Конец валидации

            start_time = time.time() * 1000
            
            loss_avg = loss_sum / len(val_loader)
            self.multi_writer.add_scalar(f'validation/loss/{dataset_name}', loss_avg, iters)
            mse_loss_avg = mse_loss_sum / len(val_loader)
            self.multi_writer.add_scalar(f'validation/mse_loss/{dataset_name}', mse_loss_avg, iters)
            ctc_loss_avg = ctc_loss_sum / len(val_loader)
            self.multi_writer.add_scalar(f'validation/ctc_loss/{dataset_name}', ctc_loss_avg, iters)

            # if loss_avg > min_val_loss:
            #     print('saving best val loss model')
            #     self.save_checkpoint(model, optimizer, epoch, iters, follow_metric_name, best_history_metric_values, best_model_info, True,
            #                     self.log_dir_name, scheduler, scheduler_warmup)

            print('[{}]\t'
                  'loss {:.3f} | mse_loss {:.3f} | ctc_loss {:.3f}\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          loss, mse_loss, ctc_loss))
            
            # Если включена ветка СР  или модель предобучена на СР
            if self.cfg.enable_sr or self.cfg.train_after_sr or (not self.cfg.enable_rec and self.cfg.recognizer == 'transformer'): 
                # PSNR

                psnr_avg = sum(psnr_batches) / len(psnr_batches)
                psnr_avg = round(psnr_avg.item(), 6)
                metric_dict['psnr_avg'] = psnr_avg
                self.writer.add_scalar(f'other/{dataset_name}/psnr_avg', psnr_avg, epoch)

                # SSIM

                ssim_avg = sum(ssim_batches) / len(ssim_batches)
                ssim_avg = round(ssim_avg.item(), 6)
                metric_dict['ssim_avg'] = ssim_avg
                self.writer.add_scalar(f'other/{dataset_name}/ssim_avg', ssim_avg, epoch)

                print(f'PSNR {float(psnr_avg):.2f} | SSIM {float(ssim_avg):.4f}\t')

                if (epoch % self.cfg.AsterValInterval == 0 and epoch != 0) or self.cfg.test_only:
                    # ASTER

                    metric_dict = self.calc_print_external_recognizer(name='aster',
                        n_correct_sr_sum=aster_n_correct_sr_sum,
                        sr_lev_dis_relation_list=aster_sr_lev_dis_relation_list,
                        cnt_images=cnt_images,
                        metric_dict=metric_dict,
                        dataset_name=dataset_name,
                        epoch=epoch,
                        calc_lr=True,
                        n_correct_lr_sum=aster_n_correct_lr_sum,
                        lr_lev_dis_relation_list=aster_lr_lev_dis_relation_list,
                        calc_hr=True,
                        n_correct_hr_sum=aster_n_correct_hr_sum,
                        hr_lev_dis_relation_list=aster_hr_lev_dis_relation_list)

                if (epoch % self.cfg.CrnnValInterval == 0 and epoch != 0) or self.cfg.test_only:
                    # CRNN
                
                    metric_dict = self.calc_print_external_recognizer(name='crnn',
                        n_correct_sr_sum=crnn_n_correct_sr_sum,
                        sr_lev_dis_relation_list=crnn_sr_lev_dis_relation_list,
                        cnt_images=cnt_images,
                        metric_dict=metric_dict,
                        dataset_name=dataset_name,
                        epoch=epoch,
                        calc_lr=True,
                        n_correct_lr_sum=crnn_n_correct_lr_sum,
                        lr_lev_dis_relation_list=crnn_lr_lev_dis_relation_list,
                        calc_hr=True,
                        n_correct_hr_sum=crnn_n_correct_hr_sum,
                        hr_lev_dis_relation_list=crnn_hr_lev_dis_relation_list)

                if (epoch % self.cfg.MoranValInterval == 0 and epoch != 0) or self.cfg.test_only:
                    # MORAN
                
                    metric_dict = self.calc_print_external_recognizer(name='moran',
                        n_correct_sr_sum=moran_n_correct_sr_sum,
                        sr_lev_dis_relation_list=moran_sr_lev_dis_relation_list,
                        cnt_images=cnt_images,
                        metric_dict=metric_dict,
                        dataset_name=dataset_name,
                        epoch=epoch,
                        calc_lr=True,
                        n_correct_lr_sum=moran_n_correct_lr_sum,
                        lr_lev_dis_relation_list=moran_lr_lev_dis_relation_list,
                        calc_hr=True,
                        n_correct_hr_sum=moran_n_correct_hr_sum,
                        hr_lev_dis_relation_list=moran_hr_lev_dis_relation_list)
 
            # Если включена ветка распознавания
            if self.cfg.enable_rec:
            
                metric_dict = self.calc_print_external_recognizer('ctc',
                    n_correct_sr_sum=ctc_sr_n_correct_sum,
                    sr_lev_dis_relation_list=ctc_sr_lev_dis_relation_list,
                    cnt_images=cnt_images,
                    metric_dict=metric_dict,
                    dataset_name=dataset_name,
                    epoch=epoch)

            end_time = time.time() * 1000
            duration = end_time - start_time1
            spend_time += '\t Хвост валидации '+str(duration)+'  \n'
            
            return metric_dict, spend_time


    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        items = os.listdir(self.test_data_dir)
        for test_dir in items:
            test_data, test_loader = self.get_test_data(os.path.join(self.test_data_dir, test_dir))
            data_name = self.args.test_data_dir.split('/')[-1]
            print('evaling %s' % data_name)
            if self.args.rec == 'moran':
                moran = self.MORAN_init()
                moran.eval()
            elif self.args.rec == 'aster':
                aster, aster_info = self.Aster_init()
                aster.eval()
            elif self.args.rec == 'crnn':
                crnn, _ = self.CRNN_init()
                crnn.eval()
            if self.args.arch != 'bicubic':
                # for p in model.parameters():
                #     p.requires_grad = False
                model.eval()
            n_correct = 0
            cnt_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
            current_acc_dict = {test_dir: 0}
            time_begin = time.time()
            sr_time = 0
            for i, data in (enumerate(test_loader)):
                images_hr, images_lr, _, label_strs, _ = data
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                sr_beigin = time.time()
                images_sr = model(images_lr)

                # print('srshape',images_sr.shape)
                # print('hrshape',images_hr.shape)

                # images_sr = images_lr
                sr_end = time.time()
                sr_time += sr_end - sr_beigin
                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                if self.args.rec == 'moran':
                    moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                         debug=True)
                    preds, preds_reverse = moran_output[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                    pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                elif self.args.rec == 'aster':
                    aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                    aster_output_sr = aster(aster_dict_sr["images"])
                    pred_rec_sr = aster_output_sr['output']['pred_rec']
                    pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                    aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                    aster_output_lr = aster(aster_dict_lr)
                    pred_rec_lr = aster_output_lr['output']['pred_rec']
                    pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                elif self.args.rec == 'crnn':
                    crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                    crnn_output = crnn(crnn_input)
                    _, preds = crnn_output.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                    pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                for pred, target in zip(pred_str_sr, label_strs):
                    if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                        n_correct += 1
                cnt_images += val_batch_size
                torch.cuda.empty_cache()
                if i % 10 == 0:
                    print('Evaluation: [{}][{}/{}]\t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  i + 1, len(test_loader), ))
                # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
            time_end = time.time()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            acc = round(n_correct / cnt_images, 4)
            fps = cnt_images / (time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[test_dir] = float(acc)
            result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
            print(result)