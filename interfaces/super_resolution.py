from this import d
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

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

step = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0


class TextSR(base.TextBase):
    def train(self):
        print('train started')
        cfg = self.cfg

        scheduler_plateau_patience = 2
        scheduler_plateau_cooldown = 4
        scheduler_warmup_epoch = scheduler_plateau_cooldown
        scheduler_plateau_factor = 0.8

        config_txt = f"""Возможно номальное обучение  \n
                         scale_factor: {cfg.scale_factor}  \n
                         width: {cfg.width}  \n
                         height: {cfg.height}  \n
                         workers: {cfg.workers}  \n
                         batch_size: {cfg.batch_size}  \n
                         epochs: {cfg.epochs}  \n
                         lr: {cfg.lr}  \n
                         convNext_type: {cfg.convNext_type}  \n
                           \n
                         enable_sr: {cfg.enable_sr}  \n
                         enable_rec: {cfg.enable_rec}  \n
                           \n
                         freeze_sr: {cfg.freeze_sr}  \n
                         freeze_rec: {cfg.freeze_rec}  \n
                         freeze_convnext: {cfg.freeze_convnext}  \n
                           \n
                         train_after_sr: {cfg.train_after_sr}  \n
                           \n
                         lambda_mse: {cfg.lambda_mse}  \n
                         lambda_ctc: {cfg.lambda_ctc}  \n
                           \n
                         acc_best_model: {cfg.acc_best_model}  \n
                         rec_best_model_save: {cfg.rec_best_model_save}  \n
                           \n
                         scheduler_plateau_patience: {scheduler_plateau_patience}  \n
                         scheduler_plateau_cooldown: {scheduler_plateau_cooldown}  \n
                         scheduler_warmup_epoch: {scheduler_warmup_epoch}  \n
                         scheduler_plateau_factor: {scheduler_plateau_factor}"""
        self.writer.add_text('config', config_txt)

        train_dataset, train_loader = self.get_train_data()
        test_val_dataset_list, test_val_loader_list = self.get_test_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        aster, aster_info = self.CRNN_init()
        optimizer = self.optimizer_init(model)

        # scheduler

        scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', patience=scheduler_plateau_patience, cooldown=scheduler_plateau_cooldown, min_lr=1e-7, verbose=True, factor=scheduler_plateau_factor)
        scheduler_warmup = WarmupScheduler(optimizer, warmup_epoch=scheduler_warmup_epoch)

        best_history_metric_values = dict()
        for val_loader in test_val_loader_list:
            data_name = val_loader.dataset.dataset_name
            best_history_metric_values[data_name] = 0
        
        # Если включена только ветка распознавания
        if self.cfg.enable_rec == True:
            if self.cfg.rec_best_model_save == 'lev_dis':
                best_history_lev_dis = dict()
                for val_loader in test_val_loader_list:
                    data_name = val_loader.dataset.dataset_name
                    best_history_lev_dis[data_name] = 0
            elif self.cfg.rec_best_model_save == 'acc':
                best_history_acc = dict()
                for val_loader in test_val_loader_list:
                    data_name = val_loader.dataset.dataset_name
                    best_history_acc[data_name] = 0
            else:
                raise exceptions.WrongMetrucForSaveBestRec
        # Если включена только ср ветка
        elif self.cfg.enable_sr == True:
            best_history_ssim = dict()
            for val_loader in test_val_loader_list:
                data_name = val_loader.dataset.dataset_name
                best_history_ssim[data_name] = 0
        else:
            raise exceptions.WrongEnabledBranches

        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        best_ssim = 0

        converge_list = []

        print(len(train_loader))

        global step

        loss = 99999

        for epoch in tqdm.tqdm(range(cfg.epochs), desc='training'):
            scheduler_plateau.step(loss)
            scheduler_warmup.step()
            
            pbar = tqdm.tqdm((enumerate(train_loader)), leave = False, desc='batch', total=len(train_loader))
            for j, data in pbar:
                model.train()
                # for p in model.parameters():
                #     assert p.requires_grad == True
                iters = len(train_loader) * epoch + j

                # Получение данных из датасетов
                images_hr, images_lr, label_strs, dataset_name = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                # Прогон модели
                
                images_sr, tag_scores = model(images_lr)

                # Лоссы
                
                loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)
                
                # Запись лоссов в TensorBoard
                
                # self.writer.add_scalar('loss/total_loss', total_loss.data, step)
                self.writer.add_scalar('loss/loss', loss, step)
                self.writer.add_scalar('loss/mse_loss', mse_loss, step)
                self.writer.add_scalar('loss/ctc_loss', ctc_loss, step)
                # self.writer.add_scalar('loss/content_loss', recognition_loss, step)
                
                # Loss и Optimizer

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                # lr = []
                for ind, param_group in enumerate(optimizer.param_groups):
                    # lr = param_group['lr']
                    lr = scheduler_warmup.get_last_lr()[0]
                    self.writer.add_scalar('loss/learning rate', lr, epoch)
                    break
                
                # Вывод лоссов и ЛР в консоль

                if iters % cfg.displayInterval == 0:
                    info_string = f"loss={float(loss.data):03.3f} | " \
                                  f"mse_loss={float(mse_loss):03.3f} | " \
                                  f"ctc_loss={float(ctc_loss):03.3f} | " \
                                  f"learning rate={lr:.10f}"

                    pbar.set_description(info_string)
                
                # Вывод изображений в TensorBoard на первом батче в эпохе
                if j == 1:
                    index = random.randint(0, images_lr.shape[0]-1)
                    dataset = dataset_name[index]
                    show_lr = images_lr[index, ...].clone().detach()
                    show_hr = images_hr[index, ...].clone().detach()
                    show_sr = images_sr[index, ...].clone().detach()

                    un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                    self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_lr_image_first_batch', torch.clamp(un(show_lr), min=0, max=1), step)
                    self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_sr_image_first_batch', torch.clamp(un(show_sr), min=0, max=1), step)
                    self.writer.add_image(f'first_batch/{dataset}/{epoch}_epoch_{index}_index_hr_image_first_batch', torch.clamp(un(show_hr), min=0, max=1), step)


                # Валидация
                if j == len(train_loader)-1:
                    print('\n')
                    print('======================================================')

                    current_acc_dict = {}
                    current_metric_dict = {}
                    current_ssim_dict = {}
                    
                    for k, val_loader in enumerate(test_val_loader_list):
                        data_name = val_loader.dataset.dataset_name

                        print('\n')
                        print('evaling %s' % data_name)

                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info, data_name, step, epoch)

                        converge_list.append({'iterator': iters,
                                              'psnr': metrics_dict['psnr'],
                                              'ssim': metrics_dict['ssim'],
                                              'crnn_sr_accuray': metrics_dict['crnn_sr_accuray'],
                                              'crnn_lr_accuray': metrics_dict['crnn_lr_accuray'],
                                              'ctc_sr_accuray': metrics_dict['ctc_sr_accuray'],
                                              'psnr_avg': metrics_dict['psnr_avg'],
                                              'ssim_avg': metrics_dict['ssim_avg']})

                        # Если включена только ветка распознавания, то обновляем точность распознавания
                        if self.cfg.enable_rec == True:
                            if self.cfg.acc_best_model == 'crnn':
                                accuray = metrics_dict['crnn_sr_accuray']
                            elif self.cfg.acc_best_model == 'ctc':
                                accuray = metrics_dict['ctc_sr_accuray']
                            else:
                                accuray = metrics_dict['ctc_sr_accuray']
                            
                            current_acc_dict[data_name] = float(accuray)

                            if accuray > best_history_acc[data_name]:
                                best_history_acc[data_name] = float(accuray)
                                best_history_acc['epoch'] = epoch
                                print('╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟupdate best_acc_%s = %.4f%%*' % (data_name, best_history_acc[data_name] * 100))
                            else:
                                print('not update best_acc_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                        # Если включена только ср ветка, то обновляем степень схожести изображений (SSIM)
                        elif self.cfg.enable_sr == True:
                            ssim = metrics_dict['ssim_avg']
                            current_ssim_dict[data_name] = float(ssim)

                            if ssim > best_history_ssim[data_name]:
                                best_history_ssim[data_name] = float(ssim)
                                best_history_ssim['epoch'] = epoch
                                print('╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟ*:・ﾟupdate best_ssim_%s = %.10f' % (data_name, best_history_ssim[data_name]))
                            else:
                                print('not update best_ssim_%s = %.4f%%' % (data_name, best_history_ssim[data_name]))
                        else:
                            raise exceptions.WrongEnabledBranches
                    
                    # Если включена только ветка распознавания, то модель сохраняем по точности распознавания
                    if self.cfg.enable_rec == True:
                        if sum(current_acc_dict.values()) > best_acc:
                            best_acc = sum(current_acc_dict.values())
                            best_model_acc = current_acc_dict
                            best_model_acc['epoch'] = epoch
                            best_model_psnr[data_name] = metrics_dict['psnr_avg']
                            best_model_ssim[data_name] = metrics_dict['ssim_avg']
                            best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                            print('saving best model')
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True,
                                                converge_list, self.log_dir_name)
                    # Если включена только ср ветка, то модель сохраняем по степени схожести изображений (SSIM)
                    elif self.cfg.enable_sr == True:
                        if sum(current_ssim_dict.values()) > best_ssim:
                            best_ssim = sum(current_ssim_dict.values())
                            best_model_ssim[data_name] = current_ssim_dict

                            if self.cfg.acc_best_model == 'crnn':
                                accuray = metrics_dict['crnn_sr_accuray']
                            elif self.cfg.acc_best_model == 'ctc':
                                accuray = metrics_dict['ctc_sr_accuray']
                            else:
                                accuray = metrics_dict['ctc_sr_accuray']
                            best_model_acc = accuray

                            best_model_ssim['epoch'] = epoch
                            best_model_psnr[data_name] = metrics_dict['psnr_avg']

                            best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                            print('saving best model')
                            self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True,
                                                converge_list, self.log_dir_name)
                        
                # Сохранение чек-поинта
                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list,
                                         self.log_dir_name)

            
                step += 1
            

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
        labels = self.FULL_VOCAB_LIST
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
        for i in range(self.cfg.batch_size):
            dig_string = beam_results[i][0][:out_lens[i][0]]
            string = ''
            for dig in dig_string:
                string += list(self.FULL_VOCAB.keys())[list(self.FULL_VOCAB.values()).index(str(int(dig)))]
            
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

    
    def eval(self, model, val_loader, image_crit, index, recognizer, aster_info, mode, step, epoch):
        with torch.no_grad():
            global easy_test_times
            global medium_test_times
            global hard_test_times

            model.eval()
            recognizer.eval()

            crnn_n_correct_sr = 0
            crnn_sr_lev_dis_list = []
            crnn_n_correct_lr = 0
            crnn_lr_lev_dis_list = []
            ctc_sr_n_correct = 0
            ctc_sr_lev_dis_list = []
            # ctc_lr_n_correct = 0
            # ctc_lr_lev_dis_list = []
            sum_images = 0

            # metric_dict = {'ICDAR 2013': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'ICDAR 2015': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'The Street View Text': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'The IIIT 5K-word': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'TextZoom_easy': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'TextZoom_medium': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'TextZoom_hard': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []},
            #                'Total': {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'ctc_sr_accuray': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []}}

            metric_dict = {'psnr': [], 'ssim': [], 'crnn_sr_accuray': 0.0, 'crnn_sr_lev_dis_avg': 0.0, 'crnn_lr_accuray': 0.0, 'crnn_lr_lev_dis_avg': 0.0, 'ctc_sr_accuray': 0.0, 'ctc_sr_lev_dis_avg': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels': []}

            image_start_index = 0

            for i, data in (enumerate(val_loader)):
                # Сбор данных из датасетов и получение результатов работы модели
                images_hr, images_lr, label_strs, dataset_name = data

                val_batch_size = images_lr.shape[0]

                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                images_sr, tag_scores = model(images_lr)

                loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)

                # Если включена ветка СР или модель предобучена на СР, считаем степень зашумлённости и похожести изображений
                if self.cfg.enable_sr or self.cfg.train_after_sr: 
                    # Вычисление PSNR и SSIM
                    
                    metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                    metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                # Если включена ветка распознавания, считаем точность распознавания
                if self.cfg.enable_rec: 
                    # CRNN Test
                    
                    crnn_recognizer_dict_sr = self.parse_crnn_data(images_sr[:, :3, :, :])
                    crnn_recognizer_output_sr = recognizer(crnn_recognizer_dict_sr)
                    crnn_outputs_sr = crnn_recognizer_output_sr.permute(1, 0, 2).contiguous()
                    crnn_predict_result_sr = self.get_crnn_pred(crnn_outputs_sr)
                    metric_dict['images_and_labels'].append((images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, crnn_predict_result_sr))

                    crnn_cnt_sr = 0
                    crnn_sr_pred_text = 'target -> crnn_sr_pred  \n'
                    for pred, target in zip(crnn_predict_result_sr, label_strs):

                        crnn_sr_pred_text += target+' -> '+pred+"  \n"
                        crnn_sr_lev_dis_list.append(self.levenshtein_distance(target, pred))
                        if pred == target:
                            crnn_n_correct_sr += 1
                        crnn_cnt_sr += 1
                    
                    crnn_recognizer_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                    crnn_recognizer_output_lr = recognizer(crnn_recognizer_dict_lr)
                    crnn_outputs_lr = crnn_recognizer_output_lr.permute(1, 0, 2).contiguous()
                    crnn_predict_result_lr = self.get_crnn_pred(crnn_outputs_lr)

                    crnn_cnt_lr = 0
                    crnn_lr_pred_text = 'target -> crnn_lr_pred  \n'
                    for pred, target in zip(crnn_predict_result_lr, label_strs):
                        crnn_lr_pred_text += target+' -> '+pred+"  \n"
                        crnn_lr_lev_dis_list.append(self.levenshtein_distance(target, pred))
                        if pred == target:
                            crnn_n_correct_lr += 1
                        crnn_cnt_lr += 1

                    # CTC Test

                    ctc_cnt = 0
                    ctc_pred_text = 'target -> ctc_decode_string  \n'
                    ctc_decode_strings = self.ctc_decode(tag_scores)
                    if self.cfg.enable_rec:
                        for ctc_decode_string, target in zip(ctc_decode_strings, label_strs):
                            ctc_pred_text += target+' -> '+ctc_decode_string+"  \n"
                            ctc_sr_lev_dis_list.append(self.levenshtein_distance(target, ctc_decode_string))
                            if ctc_decode_string == target:
                                ctc_sr_n_correct += 1
                        ctc_cnt += 1
                else:
                    crnn_pred_lr = 'NONE'
                    crnn_pred_sr = 'NONE'
                    ctc_pred = 'NONE'
                

                # Вывод изображений и предсказаний в TensorBoard
                if i == len(val_loader) - 1:
                    index = random.randint(0, images_lr.shape[0]-1)

                    # Если включена ветка распознавания
                    if self.cfg.enable_rec:
                        ground_truth = label_strs[index].replace('"', '<quot>')
                        crnn_pred_sr = crnn_predict_result_sr[index].replace('"', '<quot>')
                        crnn_pred_lr = crnn_predict_result_lr[index].replace('"', '<quot>')
                        ctc_pred = ctc_decode_strings[index].replace('"', '<quot>')

                        self.writer.add_text(f'{mode}/CTC_pred/{epoch}_epoch', ctc_pred_text)
                        self.writer.add_text(f'{mode}/CRNN_SR_pred/{epoch}_epoch', crnn_sr_pred_text)
                        self.writer.add_text(f'{mode}/CRNN_LR_pred/{epoch}_epoch', crnn_lr_pred_text)

                    # Если включена ветка СР или модель предобучена на СР
                    if self.cfg.enable_sr or self.cfg.train_after_sr:
                        show_lr = images_lr[index, ...].clone().detach()
                        show_hr = images_hr[index, ...].clone().detach()
                        show_sr = images_sr[index, ...].clone().detach()                    

                        un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        print('save display images')
                        self.writer.add_image(f'{mode}/{epoch}_epoch_{index}_index_lr_image_eval_crnn_pred:{crnn_pred_lr}', torch.clamp(un(show_lr), min=0, max=1), step)
                        self.writer.add_image(f'{mode}/{epoch}_epoch_{index}_index_sr_image_eval_crnn_pred:{crnn_pred_sr}_ctc_pred:{ctc_pred}', torch.clamp(un(show_sr), min=0, max=1), step)
                        self.writer.add_image(f'{mode}/{epoch}_epoch_{index}_index_hr_image_eval_groundTruth:{ground_truth}', torch.clamp(un(show_hr), min=0, max=1), step)
                    
                sum_images += val_batch_size

                torch.cuda.empty_cache()

            # Если включена ветка СР  или модель предобучена на СР
            if self.cfg.enable_sr or self.cfg.train_after_sr: 
                # PSNR

                psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
                psnr_avg = round(psnr_avg.item(), 6)
                metric_dict['psnr_avg'] = psnr_avg
                self.writer.add_scalar(f'other/{mode}/psnr_avg', psnr_avg, step)

                # SSIM

                ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
                ssim_avg = round(ssim_avg.item(), 6)
                metric_dict['ssim_avg'] = ssim_avg
                self.writer.add_scalar(f'other/{mode}/ssim_avg', ssim_avg, step)
            else:
                psnr_avg = 0
                ssim_avg = 0
                print('Ветка сверхразрешения ВЫКЛЮЧЕНА. PSNR и SSIM не соответствуют действительности\n')
 
            # Если включена ветка распознавания
            if self.cfg.enable_rec:
                # CRNN ACC SR

                crnn_sr_accuray = round(crnn_n_correct_sr / sum_images, 4)
                metric_dict['crnn_sr_accuray'] = crnn_sr_accuray
                self.writer.add_scalar(f'accuray/{mode}/crnn_sr_accuray', crnn_sr_accuray * 100, step)
                print('crnn_sr_accuray: %.2f%%' % (crnn_sr_accuray * 100))

                # CRNN LEV DIS SR
                
                crnn_sr_lev_dis_avg = round(sum(crnn_sr_lev_dis_list) / sum_images, 4)
                metric_dict['crnn_sr_lev_dis_avg'] = crnn_sr_lev_dis_avg
                self.writer.add_scalar(f'other/{mode}/crnn_sr_lev_dis_avg', crnn_sr_lev_dis_avg, step)

                # CRNN ACC LR

                crnn_lr_accuray = round(crnn_n_correct_lr / sum_images, 4)
                metric_dict['crnn_lr_accuray'] = crnn_lr_accuray
                self.writer.add_scalar(f'accuray/{mode}/crnn_lr_accuray', crnn_lr_accuray * 100, step)
                print('crnn_lr_accuray: %.2f%%' % (crnn_lr_accuray * 100))

                # CRNN LEV DIS LR
                
                crnn_lr_lev_dis_avg = round(sum(crnn_lr_lev_dis_list) / sum_images, 4)
                metric_dict['crnn_lr_lev_dis_avg'] = crnn_lr_lev_dis_avg
                self.writer.add_scalar(f'other/{mode}/crnn_lr_lev_dis_avg', crnn_lr_lev_dis_avg, step)

                # CTC ACC SR

                ctc_sr_accuray = round(ctc_sr_n_correct / sum_images, 4)
                metric_dict['ctc_sr_accuray'] = ctc_sr_accuray
                self.writer.add_scalar(f'accuray/{mode}/ctc_sr_accuray', ctc_sr_accuray * 100, step)
                print('ctc_sr_accuray: %.2f%%' % (ctc_sr_accuray * 100))

                # CTC LEV DIS SR
                
                ctc_sr_lev_dis_avg = round(sum(ctc_sr_lev_dis_list) / sum_images, 4)
                metric_dict['ctc_sr_lev_dis_avg'] = ctc_sr_lev_dis_avg
                self.writer.add_scalar(f'other/{mode}/ctc_sr_lev_dis_avg', ctc_sr_lev_dis_avg, step)
            else:
                crnn_sr_accuray = 0
                crnn_lr_accuray = 0
                crnn_sr_lev_dis_avg = 0
                crnn_lr_lev_dis_avg = 0
                ctc_sr_accuray = 0
                ctc_sr_lev_dis_avg = 0
                print('Ветка распознавания ВЫКЛЮЧЕНА. CRNN и CTC не соответствуют действительности\n')

            print('[{}]\t'
                  'loss {:.3f} | mse_loss {:.3f} | ctc_loss {:.3f}\t'
                  'PSNR {:.2f} | SSIM {:.4f}\t'
                  'crnn_sr_accuray {:.2f} | crnn_lr_accuray {:.2f}\t'
                  'crnn_sr_lev_dis_avg {:.2f} | crnn_lr_lev_dis_avg {:.2f}\t'
                  'ctc_sr_accuray {:.2f} | ctc_sr_lev_dis_avg {:.2f}'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          loss, mse_loss, ctc_loss,
                          float(psnr_avg), float(ssim_avg),
                          float(crnn_sr_accuray), float(crnn_lr_accuray),
                          float(crnn_sr_lev_dis_avg), float(crnn_lr_lev_dis_avg),
                          float(ctc_sr_accuray), float(ctc_sr_lev_dis_avg), ))

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
            sum_images = 0
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
                sum_images += val_batch_size
                torch.cuda.empty_cache()
                if i % 10 == 0:
                    print('Evaluation: [{}][{}/{}]\t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  i + 1, len(test_loader), ))
                # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
            time_end = time.time()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            acc = round(n_correct / sum_images, 4)
            fps = sum_images / (time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[test_dir] = float(acc)
            result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
            print(result)