from this import d
from turtle import update
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.cuda.amp import autocast, GradScaler

from interfaces import base
import copy
import logging
from datetime import datetime
import utils
from utils import util, ssim_psnr, UnNormalize, exceptions
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
                           \n
                         resume: {cfg.resume}  \n
                           \n
                         LOSS:  \n
                         lambda_mse: {cfg.lambda_mse}  \n
                         lambda_ctc: {cfg.lambda_ctc}  \n
                           \n
                         MODEL STRUCTURE:  \n
                         convNext_type: {cfg.convNext_type}  \n
                         acc_best_model: {cfg.acc_best_model}  \n
                         rec_best_model_save: {cfg.rec_best_model_save}  \n
                         recognizer: {cfg.recognizer}  \n
                         recognizer_input: {cfg.recognizer_input}  \n
                         recognizer_input_convnext: {cfg.recognizer_input_convnext}  \n
                           \n
                         BRANCHES:  \n
                         enable_sr: {cfg.enable_sr}  \n
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
                         test_val_textzoom_data_dir: {cfg.test_val_textzoom_data_dir}"""
        self.writer.add_text('config', config_txt)

        train_dataset, train_loader = self.get_train_data()
        test_val_dataset_list, test_val_loader_list = self.get_test_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        
        del model_dict['model']
        del model_dict['crit']

        # optimizer

        optimizer = model_dict['optimizer']
        del model_dict['optimizer']

        if optimizer is None:
            optimizer = self.optimizer_init(model)

        aster, aster_info = self.CRNN_init()
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
                start_batch_time = time.time() * 1000
                spend_time = ''
                start_time = time.time() * 1000
                model.train()
                # for p in model.parameters():
                #     assert p.requires_grad == True
                iters = len(train_loader) * epoch + j
                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += 'first block '+str(duration)+'  \n'

                # Получение данных из датасетов

                start_time = time.time() * 1000
                images_hr, images_lr, label_strs, dataset_name = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += 'Получение данных из датасетов '+str(duration)+'  \n'


                # Прогон модели
                
                start_time = time.time() * 1000
                if self.cfg.fp16:
                    with autocast():
                        images_sr, tag_scores = model(images_lr)
                else:
                    images_sr, tag_scores = model(images_lr)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                if self.cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

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

                    un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                    self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_lr_image_first_batch', torch.clamp(un(show_lr), min=0, max=1), iters)
                    self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_sr_image_first_batch', torch.clamp(un(show_sr), min=0, max=1), iters)
                    self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_hr_image_first_batch', torch.clamp(un(show_hr), min=0, max=1), iters)
                
                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += 'Вывод изображений в TensorBoard на первом батче в эпохе '+str(duration)+'  \n'


                # ВАЛИДАЦИЯ

                start_time = time.time() * 1000
                if j == len(train_loader)-1 and self.cfg.eval:
                    print('\n')
                    print('\n')
                    print('===================================================VALIDATION===================================================')

                    current_follow_metric_dict = {}

                    metrics_dict_datasets = {}
                    
                    for k, val_loader in enumerate(test_val_loader_list):
                        dataset_name = val_loader.dataset.dataset_name

                        print('evaluation %s' % dataset_name)

                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info, dataset_name, iters, epoch)
                        metrics_dict_datasets[dataset_name] = metrics_dict

                        # Сохраняем значения отслеживаемой метки из каждого датасета
                        follow_metric_name, _, _ = self.get_follow_metric_name()
                        current_follow_metric_dict[dataset_name] = metrics_dict[follow_metric_name]
                        
                        # Обновляем словарь с лучими значениями отслеживаемой метрики в каждом датасете
                        best_history_follow_metric_values = self.update_best_metric(metrics_dict, best_history_follow_metric_values, dataset_name, epoch)
                    
                    # Сохранение модели
                    if self.cfg.saveBestModel:
                        best_sum_follow_metric_value = self.save_best_model(follow_metric_name, current_follow_metric_dict, best_history_follow_metric_values, 
                                                                            best_sum_follow_metric_value, metrics_dict_datasets, model, optimizer, scheduler_plateau, scheduler_warmup, iters, epoch)
                  
                end_time = time.time() * 1000
                duration = end_time - start_time
                spend_time += 'ВАЛИДАЦИЯ '+str(duration)+'  \n'
     
                # Сохранение чек-поинта
                start_time = time.time() * 1000 
                if iters % len(train_loader)-1 == 0 and iters != 0:
                    follow_metric_name, _, _ = self.get_follow_metric_name()
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
            

    def get_crnn_pred(self, outputs):
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


    def calculate_crnn_pred(self, images, label_strs, recognizer, crnn_lev_dis_list, crnn_lev_dis_relation_list):
        crnn_recognizer_dict = self.parse_crnn_data(images[:, :3, :, :])
        crnn_recognizer_output = recognizer(crnn_recognizer_dict)
        crnn_outputs = crnn_recognizer_output.permute(1, 0, 2).contiguous()
        crnn_predict_result = self.get_crnn_pred(crnn_outputs)

        crnn_cnt = 0
        crnn_n_correct = 0
        crnn_pred_text = 'target -> crnn_pred  \n'
        for pred, target in zip(crnn_predict_result, label_strs):
            if self.cfg.letters == 'lower':
                target = target.lower()
            elif self.cfg.letters == 'upper':
                target = target.upper()
                
            crnn_pred_text += target+' -> '+pred+"  \n"
            lev_dis = self.levenshtein_distance(target, pred)
            crnn_lev_dis_list.append(lev_dis)
            crnn_lev_dis_relation_list.append(lev_dis / len(target) if len(target) > 0 else lev_dis)
            if pred == target:
                crnn_n_correct += 1
            crnn_cnt += 1
        
        return crnn_n_correct, crnn_cnt, crnn_lev_dis_list, crnn_lev_dis_relation_list, crnn_pred_text, crnn_predict_result
    

    def calculate_ctc_pred(self, tag_scores, label_strs, ctc_lev_dis_list, ctc_lev_dis_relation_list):
        ctc_cnt = 0
        ctc_n_correct = 0
        ctc_pred_text = 'target -> ctc_decode_string  \n'
        ctc_decode_strings = self.ctc_decode(tag_scores)
        if self.cfg.enable_rec:
            for ctc_decode_string, target in zip(ctc_decode_strings, label_strs):
                if self.cfg.letters == 'lower':
                    target = target.lower()
                elif self.cfg.letters == 'upper':
                    target = target.upper()

                ctc_pred_text += target+' -> '+ctc_decode_string+"  \n"
                lev_dis = self.levenshtein_distance(target, ctc_decode_string)
                ctc_lev_dis_list.append(lev_dis)
                ctc_lev_dis_relation_list.append(lev_dis / len(target) if len(target) > 0 else lev_dis)
                if ctc_decode_string == target:
                    ctc_n_correct += 1
            ctc_cnt += 1

        return ctc_n_correct, ctc_cnt, ctc_lev_dis_list, ctc_lev_dis_relation_list, ctc_pred_text, ctc_decode_strings
    
    
    def get_follow_metric_name(self):
        if self.cfg.enable_rec == True or self.cfg.recognizer == 'transformer':
            if self.cfg.acc_best_model == 'crnn':
                metric_name = 'crnn_sr'
            elif self.cfg.acc_best_model == 'ctc':
                metric_name = 'ctc_sr'
            else:
                raise exceptions.WrongModelForSaveBestRec
            if self.cfg.rec_best_model_save == 'acc':
                metric_name += '_accuracy'
                metric_print = 'accuracy'
                direction = 'max'
            elif self.cfg.rec_best_model_save == 'lev_dis':
                metric_name += '_lev_dis_relation_avg'
                metric_print = 'levenshtein_distance'
                direction = 'min'
            else:
                raise exceptions.WrongMetrucForSaveBestRec
        elif self.cfg.enable_sr == True:
            metric_name = 'ssim_avg'
            metric_print = 'SSIM'
            direction = 'max'
        else:
            raise exceptions.WrongEnabledBranches

        return metric_name, metric_print, direction
    
    
    def update_best_metric(self, metrics_dict, best_history_metric_values, dataset_name, epoch):
        
        metric_name, metric_print, direction = self.get_follow_metric_name()
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

    
    def save_best_model(self, follow_metric_name, current_metric_dict, best_history_metric_values, best_sum_metric_value, metrics_dict_datasets, model, optimizer, scheduler, scheduler_warmup, iters, epoch):
        
        follow_metric_name, metric_print, direction = self.get_follow_metric_name()

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
    
    
    # валидация одного датасета
    def eval(self, model, val_loader, image_crit, index, recognizer, aster_info, dataset_name, iters, epoch):
        with torch.no_grad():
            global easy_test_times
            global medium_test_times
            global hard_test_times

            model.eval()
            recognizer.eval()

            crnn_n_correct_sr_sum = 0
            crnn_sr_lev_dis_list = []
            crnn_sr_lev_dis_relation_list = []
            crnn_n_correct_lr_sum = 0
            crnn_lr_lev_dis_list = []
            crnn_lr_lev_dis_relation_list = []
            ctc_sr_n_correct_sum = 0
            ctc_sr_lev_dis_list = []
            ctc_sr_lev_dis_relation_list = []
            # ctc_lr_n_correct_sum = 0
            # ctc_lr_lev_dis_list = []
            psnr_batches = []
            ssim_batches = []
            cnt_images = 0

            # metric_dict = {'ICDAR 2013': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'ICDAR 2015': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'The Street View Text': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'The IIIT 5K-word': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'TextZoom_easy': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'TextZoom_medium': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'TextZoom_hard': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'Total': {'psnr': [], 'ssim': [], 'crnn_sr_accuracy': 0.0, 'ctc_sr_accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []}}

            metric_dict = {'crnn_sr_accuracy': 0.0, 'crnn_sr_lev_dis_relation_avg': 0.0, 'crnn_lr_accuracy': 0.0, 'crnn_lr_lev_dis_relation_avg': 0.0, 'ctc_sr_accuracy': 0.0, 'ctc_sr_lev_dis_relation_avg': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []}

            image_start_index = 0

            for i, data in (enumerate(val_loader)):
                # Сбор данных из датасетов и получение результатов работы модели
                images_hr, images_lr, label_strs, _ = data

                val_batch_size = images_lr.shape[0]

                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                images_sr, tag_scores = model(images_lr)

                if self.cfg.recognizer == 'transformer':
                    loss, mse_loss, attention_loss, recognition_loss, word_decoder_result = image_crit(images_sr, images_hr, label_strs)
                    
                elif self.cfg.recognizer == 'lstm':
                    loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)
                else:
                    raise exceptions.WrongRecognizer

                # Если включена ветка СР или модель предобучена на СР, считаем степень зашумлённости и похожести изображений
                if self.cfg.enable_sr or self.cfg.train_after_sr: 
                    # Вычисление PSNR и SSIM (Средние по батчу)                    
                    psnr_batches.append(self.cal_psnr(images_sr, images_hr))
                    ssim_batches.append(self.cal_ssim(images_sr, images_hr))

                    # CRNN Test
                    
                    # SR
                    crnn_n_correct_sr, crnn_cnt_sr, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list, crnn_sr_pred_text, crnn_predict_result_sr = self.calculate_crnn_pred(images_sr, label_strs, recognizer, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list)
                    crnn_n_correct_sr_sum += crnn_n_correct_sr
                    
                    # LR
                    crnn_n_correct_lr, crnn_cnt_lr, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list, crnn_lr_pred_text, crnn_predict_result_lr = self.calculate_crnn_pred(images_lr, label_strs, recognizer, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list)
                    crnn_n_correct_lr_sum += crnn_n_correct_lr

                else:
                    crnn_predict_result_lr = None
                    crnn_pred_lr = 'NONE'
                    crnn_predict_result_sr = None
                    crnn_pred_sr = 'NONE'

                # Если включена ветка распознавания, считаем точность распознавания
                if self.cfg.enable_rec:

                    # CTC Test
                    ctc_sr_n_correct, ctc_cnt, ctc_sr_lev_dis_list, ctc_sr_lev_dis_relation_list, ctc_pred_text, ctc_decode_strings = self.calculate_ctc_pred(tag_scores, label_strs, ctc_sr_lev_dis_list, ctc_sr_lev_dis_relation_list)                    
                    ctc_sr_n_correct_sum += ctc_sr_n_correct
                    
                elif self.cfg.recognizer == 'transformer':
                    # CRNN Test
                    
                    # SR
                    crnn_n_correct_sr, crnn_cnt_sr, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list, crnn_sr_pred_text, crnn_predict_result_sr = self.calculate_crnn_pred(images_sr, label_strs, recognizer, crnn_sr_lev_dis_list, crnn_sr_lev_dis_relation_list)
                    crnn_n_correct_sr_sum += crnn_n_correct_sr
                    
                    # LR
                    crnn_n_correct_lr, crnn_cnt_lr, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list, crnn_lr_pred_text, crnn_predict_result_lr = self.calculate_crnn_pred(images_lr, label_strs, recognizer, crnn_lr_lev_dis_list, crnn_lr_lev_dis_relation_list)
                    crnn_n_correct_lr_sum += crnn_n_correct_lr

                    ctc_pred = 'NONE'
                    ctc_decode_strings = None
                else:
                    ctc_pred = 'NONE'
                    ctc_decode_strings = None
                

                # Вывод изображений и предсказаний в TensorBoard
                if i == len(val_loader) - 1:
                    index = random.randint(0, images_lr.shape[0]-1)

                    # Если включена ветка распознавания
                    if self.cfg.enable_rec:
                        ground_truth = label_strs[index].replace('"', '<quot>')
                        if self.cfg.letters == 'lower':
                            ground_truth = ground_truth.lower()
                        elif self.cfg.letters == 'upper':
                            ground_truth = ground_truth.upper()
                        ctc_pred = ctc_decode_strings[index].replace('"', '<quot>')

                        self.writer.add_text(f'CTC_pred/{epoch}_epoch/{dataset_name}', ctc_pred_text)
                    elif self.cfg.recognizer == 'transformer':
                        ground_truth = label_strs[index].replace('"', '<quot>')
                        crnn_pred_sr = crnn_predict_result_sr[index].replace('"', '<quot>')
                        crnn_pred_lr = crnn_predict_result_lr[index].replace('"', '<quot>')

                        self.writer.add_text(f'CRNN_SR_pred/{epoch}_epoch/{dataset_name}', crnn_sr_pred_text)
                        self.writer.add_text(f'CRNN_LR_pred/{epoch}_epoch/{dataset_name}', crnn_lr_pred_text)
                    

                    # Если включена ветка СР или модель предобучена на СР
                    if self.cfg.enable_sr or self.cfg.train_after_sr or self.cfg.recognizer == 'transformer':

                        crnn_pred_sr = crnn_predict_result_sr[index].replace('"', '<quot>')
                        crnn_pred_lr = crnn_predict_result_lr[index].replace('"', '<quot>')

                        self.writer.add_text(f'CRNN_SR_pred/{epoch}_epoch/{dataset_name}', crnn_sr_pred_text)
                        self.writer.add_text(f'CRNN_LR_pred/{epoch}_epoch/{dataset_name}', crnn_lr_pred_text)
                        
                        metric_dict['images_and_labels'].append((images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, crnn_predict_result_sr, crnn_predict_result_lr, ctc_decode_strings))

                        show_lr = images_lr[index, ...].clone().detach()
                        show_hr = images_hr[index, ...].clone().detach()
                        show_sr = images_sr[index, ...].clone().detach()                    

                        un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        print('save display images')
                        self.writer.add_image(f'{dataset_name}/{epoch}_epoch_{index}_index_lr_image_eval_crnn_pred:{crnn_pred_lr}', torch.clamp(un(show_lr), min=0, max=1), iters)
                        self.writer.add_image(f'{dataset_name}/{epoch}_epoch_{index}_index_sr_image_eval_crnn_pred:{crnn_pred_sr}_ctc_pred:{ctc_pred}', torch.clamp(un(show_sr), min=0, max=1), iters)
                        self.writer.add_image(f'{dataset_name}/{epoch}_epoch_{index}_index_hr_image_eval_groundTruth:{ground_truth}', torch.clamp(un(show_hr), min=0, max=1), iters)
                    
                cnt_images += val_batch_size

                torch.cuda.empty_cache()

            # Если включена ветка СР  или модель предобучена на СР
            if self.cfg.enable_sr or self.cfg.train_after_sr: 
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

                # CRNN ACC SR

                crnn_sr_accuracy = round(crnn_n_correct_sr_sum / cnt_images, 4)
                metric_dict['crnn_sr_accuracy'] = crnn_sr_accuracy
                self.writer.add_scalar(f'accuracy/{dataset_name}/crnn_sr_accuracy', crnn_sr_accuracy * 100, epoch)
                print('crnn_sr_accuracy: %.2f%%' % (crnn_sr_accuracy * 100))

                # CRNN LEV DIS SR
                
                crnn_sr_lev_dis_relation_avg = round(sum(crnn_sr_lev_dis_relation_list) / cnt_images, 4)
                metric_dict['crnn_sr_lev_dis_relation_avg'] = crnn_sr_lev_dis_relation_avg
                self.writer.add_scalar(f'other/{dataset_name}/crnn_sr_lev_dis_relation_avg', crnn_sr_lev_dis_relation_avg, epoch)

                # CRNN ACC LR

                crnn_lr_accuracy = round(crnn_n_correct_lr_sum / cnt_images, 4)
                metric_dict['crnn_lr_accuracy'] = crnn_lr_accuracy
                self.writer.add_scalar(f'accuracy/{dataset_name}/crnn_lr_accuracy', crnn_lr_accuracy * 100, epoch)
                print('crnn_lr_accuracy: %.2f%%' % (crnn_lr_accuracy * 100))

                # CRNN LEV DIS LR
                
                crnn_lr_lev_dis_relation_avg = round(sum(crnn_lr_lev_dis_relation_list) / cnt_images, 4)
                metric_dict['crnn_lr_lev_dis_relation_avg'] = crnn_lr_lev_dis_relation_avg
                self.writer.add_scalar(f'other/{dataset_name}/crnn_lr_lev_dis_relation_avg', crnn_lr_lev_dis_relation_avg, epoch)
            else:
                psnr_avg = 0
                ssim_avg = 0
                crnn_sr_accuracy = 0
                crnn_lr_accuracy = 0
                crnn_sr_lev_dis_relation_avg = 0
                crnn_lr_lev_dis_relation_avg = 0
                print('Ветка сверхразрешения ВЫКЛЮЧЕНА. CRNN, PSNR и SSIM не соответствуют действительности\n')
 
            # Если включена ветка распознавания
            if self.cfg.enable_rec:

                # CTC ACC SR

                ctc_sr_accuracy = round(ctc_sr_n_correct_sum / cnt_images, 4)
                metric_dict['ctc_sr_accuracy'] = ctc_sr_accuracy
                self.writer.add_scalar(f'accuracy/{dataset_name}/ctc_sr_accuracy', ctc_sr_accuracy * 100, epoch)
                print('ctc_sr_accuracy: %.2f%%' % (ctc_sr_accuracy * 100))

                # CTC LEV DIS SR
                
                ctc_sr_lev_dis_relation_avg = round(sum(ctc_sr_lev_dis_relation_list) / cnt_images, 4)
                metric_dict['ctc_sr_lev_dis_relation_avg'] = ctc_sr_lev_dis_relation_avg
                self.writer.add_scalar(f'other/{dataset_name}/ctc_sr_lev_dis_relation_avg', ctc_sr_lev_dis_relation_avg, epoch)
            elif self.cfg.recognizer == 'transformer':
                # CRNN ACC SR

                crnn_sr_accuracy = round(crnn_n_correct_sr_sum / cnt_images, 4)
                metric_dict['crnn_sr_accuracy'] = crnn_sr_accuracy
                self.writer.add_scalar(f'accuracy/{dataset_name}/crnn_sr_accuracy', crnn_sr_accuracy * 100, epoch)
                print('crnn_sr_accuracy: %.2f%%' % (crnn_sr_accuracy * 100))

                # CRNN LEV DIS SR
                
                crnn_sr_lev_dis_relation_avg = round(sum(crnn_sr_lev_dis_relation_list) / cnt_images, 4)
                metric_dict['crnn_sr_lev_dis_relation_avg'] = crnn_sr_lev_dis_relation_avg
                self.writer.add_scalar(f'other/{dataset_name}/crnn_sr_lev_dis_relation_avg', crnn_sr_lev_dis_relation_avg, epoch)

                # CRNN ACC LR

                crnn_lr_accuracy = round(crnn_n_correct_lr_sum / cnt_images, 4)
                metric_dict['crnn_lr_accuracy'] = crnn_lr_accuracy
                self.writer.add_scalar(f'accuracy/{dataset_name}/crnn_lr_accuracy', crnn_lr_accuracy * 100, epoch)
                print('crnn_lr_accuracy: %.2f%%' % (crnn_lr_accuracy * 100))

                # CRNN LEV DIS LR
                
                crnn_lr_lev_dis_relation_avg = round(sum(crnn_lr_lev_dis_relation_list) / cnt_images, 4)
                metric_dict['crnn_lr_lev_dis_relation_avg'] = crnn_lr_lev_dis_relation_avg
                self.writer.add_scalar(f'other/{dataset_name}/crnn_lr_lev_dis_relation_avg', crnn_lr_lev_dis_relation_avg, epoch)

                ctc_loss = 0
                ctc_sr_accuracy = 0
                ctc_sr_lev_dis_relation_avg = 0

                print('Ветка распознавания ВЫКЛЮЧЕНА. CTC не соответствуют действительности\n')
            else:
                ctc_sr_accuracy = 0
                ctc_sr_lev_dis_relation_avg = 0
                print('Ветка распознавания ВЫКЛЮЧЕНА. CTC не соответствуют действительности\n')

            print('[{}]\t'
                  'loss {:.3f} | mse_loss {:.3f} | ctc_loss {:.3f}\t'
                  'PSNR {:.2f} | SSIM {:.4f}\t'
                  'crnn_sr_accuracy {:.2f} | crnn_lr_accuracy {:.2f}\t'
                  'crnn_sr_lev_dis_relation_avg {:.4f} | crnn_lr_lev_dis_relation_avg {:.4f}\t'
                  'ctc_sr_accuracy {:.2f} | ctc_sr_lev_dis_relation_avg {:.4f}'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          loss, mse_loss, ctc_loss,
                          float(psnr_avg), float(ssim_avg),
                          float(crnn_sr_accuracy), float(crnn_lr_accuracy),
                          float(crnn_sr_lev_dis_relation_avg), float(crnn_lr_lev_dis_relation_avg),
                          float(ctc_sr_accuracy), float(ctc_sr_lev_dis_relation_avg), ))

            return metric_dict


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
                images_hr, images_lr, label_strs = data
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