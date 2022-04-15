import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from interfaces import base
import copy
import logging
from datetime import datetime
from utils import util, ssim_psnr, UnNormalize
import random
import os
import time
import tqdm

from utils.metrics import get_str_list, Accuracy
from utils.util import str_filt

import matplotlib.pyplot as plt

times = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0


class TextSR(base.TextBase):
    def train(self):
        print('train started')
        cfg = self.cfg
        train_dataset, train_loader = self.get_train_data()
        test_val_dataset_list, test_val_loader_list = self.get_test_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        aster, aster_info = self.CRNN_init()
        optimizer = self.optimizer_init(model)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # best_history_acc = dict(
        #     zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.cfg.test_val_data_dir],
        #         [0] * len(test_val_loader_list)))

        best_history_acc = dict()
        for val_loader in test_val_loader_list:
            data_name = val_loader.dataset.dataset_name
            best_history_acc[data_name] = 0

        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []
        print(len(train_loader))
        for epoch in tqdm.tqdm(range(cfg.epochs), desc='training'):
            pbar = tqdm.tqdm((enumerate(train_loader)), leave = False, desc='batch', total=len(train_loader))
            for j, data in pbar:
            # for j, data in tqdm.tqdm((enumerate(train_loader)), leave = False, desc='batch'):
                a = j + 1
                model.train()
                for p in model.parameters():
                    assert p.requires_grad == True
                iters = len(train_loader) * epoch + j

                images_hr, images_lr, label_strs, dataset_name = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                # un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # temp_hr = images_hr[0, ...].clone().detach()
                # img_un = un(temp_hr)
                # plt.imshow(torch.moveaxis(img_un.cpu(), 0, 2))
                # plt.show()

                # if torch.sum(torch.isnan(images_hr))>0 or torch.sum(torch.isnan(images_lr))>0:
                #     print('j:',j)
                #     raise Exception('j:',j)

                images_sr, tag_scores = model(images_lr)

                loss, mse_loss, ctc_loss = image_crit(images_sr, tag_scores, images_hr, label_strs)

                # if j>=499:
                #     # index = random.randint(0, images_lr.shape[0] - 1)
                #     dataset_name = train_loader
                #     for index in range(images_lr.shape[0]):
                #         self.writer.add_image(f'debug/{dataset_name[index]}/lr_image/j {j}/index {index}',
                #                               torch.clamp(((images_lr[index, ...] + 1) / 2 * 255).long(), min=0, max=255),
                #                               easy_test_times)
                #         self.writer.add_image(f'debug/{dataset_name[index]}/sr_image/j {j}/index {index}',
                #                               torch.clamp(((images_sr[index, ...] + 1) / 2 * 255).long(), min=0, max=255),
                #                               easy_test_times)
                #         self.writer.add_image(f'debug/{dataset_name[index]}/hr_image/j {j}/index {index}',
                #                               torch.clamp(((images_hr[index, ...] + 1) / 2 * 255).long(), min=0, max=255),
                #                               easy_test_times)

                loss_im = loss * 100
                # loss_im = loss

                global times
                self.writer.add_scalar('loss/total_loss', loss_im.data, times)
                self.writer.add_scalar('loss/loss', loss, times)
                self.writer.add_scalar('loss/mse_loss', mse_loss, times)
                self.writer.add_scalar('loss/ctc_loss', ctc_loss, times)
                # self.writer.add_scalar('loss/content_loss', recognition_loss, times)
                times += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step(loss)

                if iters % cfg.displayInterval == 0:
                    info_string = f"total_loss={float(loss_im.data):03.3f} | " \
                                  f"loss={loss:03.3f} | " \
                                  f"mse_loss={mse_loss:03.3f} | " \
                                  f"ctc_loss={ctc_loss:03.3f} | "

                    pbar.set_description(info_string)
                # if iters % cfg.displayInterval == 0:
                #     print('[{}]\t'
                #           'Epoch: [{}]\t'
                #           'train_loader: [{}/{}]\t'
                #           # 'vis_dir={:s}\t'
                #           'total_loss {:.3f} \t'
                #           'loss {:.3f} \t'
                #           'mse_loss {:.3f} \t'
                #           'ctc_loss {:.3f} \t'
                #           .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                #                   epoch,
                #                   j + 1, len(train_loader),
                #                   # self.vis_dir,
                #                   float(loss_im.data),
                #                   loss,
                #                   mse_loss,
                #                   ctc_loss,
                #                   ))

                if j == len(train_loader)-1:
                # if iters % cfg.valInterval == 0:
                    print('======================================================')
                    current_acc_dict = {}
                    # data_name = ''
                    # metrics_dict = {}
                    for k, val_loader in enumerate(test_val_loader_list):
                        data_name = val_loader.dataset.dataset_name
                        # self.cfg.test_val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info, data_name)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:

                            data_for_evaluation = metrics_dict['images_and_labels']

                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
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

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list,
                                         self.log_dir_name)

            # сохранять sr_img несколько изоб в послений батч эпохи


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


    def eval(self, model, val_loader, image_crit, index, recognizer, aster_info, mode):
        with torch.no_grad():
            global easy_test_times
            global medium_test_times
            global hard_test_times

            # for p in model.parameters():
            #     p.requires_grad = False
            # for p in recognizer.parameters():
            #     p.requires_grad = False
            model.eval()
            recognizer.eval()
            n_correct = 0
            n_correct_lr = 0
            sum_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                           'images_and_labels': []}
            image_start_index = 0
            for i, data in (enumerate(val_loader)):
                images_hr, images_lr, label_strs, dataset_name = data
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                images_sr, _ = model(images_lr)

                if i == len(val_loader) - 1:
                    index = random.randint(0, images_lr.shape[0]-1)
                    show_lr = images_lr[index, ...].clone().detach()
                    show_hr = images_hr[index, ...].clone().detach()
                    show_sr = images_sr[index, ...].clone().detach()

                    un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                    self.writer.add_image(f'vis/{mode}/lr_image', torch.clamp(un(show_lr), min=0, max=1), easy_test_times)
                    self.writer.add_image(f'vis/{mode}/sr_image', torch.clamp(un(show_sr), min=0, max=1), easy_test_times)
                    self.writer.add_image(f'vis/{mode}/hr_image', torch.clamp(un(show_hr), min=0, max=1), easy_test_times)

                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                recognizer_dict_sr = self.parse_crnn_data(images_sr[:, :3, :, :])
                recognizer_output_sr = recognizer(recognizer_dict_sr)
                outputs_sr = recognizer_output_sr.permute(1, 0, 2).contiguous()
                predict_result_sr = self.get_crnn_pred(outputs_sr)
                metric_dict['images_and_labels'].append(
                    (images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, predict_result_sr))

                cnt = 0
                for pred, target in zip(predict_result_sr, label_strs):
                    # self.print('{} | {} | {} | {}\n'.format(write_line, pred, str_filt(target, 'lower'),
                    #                                      pred == str_filt(target, 'lower')))
                    # write_line += 1
                    if pred == str_filt(target, 'lower'):
                        n_correct += 1
                    cnt += 1

                sum_images += val_batch_size
                torch.cuda.empty_cache()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            print('[{}]\t'
                  'loss_rec {:.3f}| loss_im {:.3f}\t'
                  'PSNR {:.2f} | SSIM {:.4f}\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          0, 0,
                          float(psnr_avg), float(ssim_avg), ))
            print('save display images')
            accuracy = round(n_correct / sum_images, 4)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            print('sr_accuray: %.2f%%' % (accuracy * 100))
            metric_dict['accuracy'] = accuracy
            metric_dict['psnr_avg'] = psnr_avg
            metric_dict['ssim_avg'] = ssim_avg
            self.writer.add_scalar('other/sr_accuray', accuracy * 100, times)
            self.writer.add_scalar('other/psnr_avg', psnr_avg, times)
            self.writer.add_scalar('other/ssim_avg', ssim_avg, times)

            if mode == 'easy':
                self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, easy_test_times)
                easy_test_times += 1
            if mode == 'medium':
                self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
                medium_test_times += 1
            if mode == 'hard':
                self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, hard_test_times)
                hard_test_times += 1

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