from tokenize import blank_re
import torch
import torch.nn as nn
from string import ascii_uppercase, ascii_lowercase
import numpy as np
from einops import rearrange

from ctcdecode import CTCBeamDecoder
from utils import util

class WangjiLoss(nn.Module):
    def __init__(self, args, cfg):
        super(WangjiLoss, self).__init__()
        self.cfg = cfg
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)

        self.Letters, self.Symbols, self.FullVocab, self.FullVocabList = util.get_vocab(self.cfg)

        # self.ce_loss = nn.CrossEntropyLoss()
        # self.l1_loss = nn.L1Loss()
        # self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # self.english_dict = {}
        # for index in range(len(self.english_alphabet)):
        #     self.english_dict[self.english_alphabet[index]] = index
        #
        # self.build_up_transformer()


    def forward(self, sr_img, tag_scores, hr_img, labels):

        batch_size = len(labels)
        ctc_loss = 0
        weight = int(self.cfg.width / self.cfg.scale_factor / 2)
        if self.cfg.enable_rec:
            targets = torch.zeros(batch_size, weight) + int(self.FullVocab['PAD'])
            # label = label.upper()

            for count, label in enumerate(labels):
                target = []

                for character in label:

                    # print(character)
                    # if character == ' ':
                    #     target.append(int(self.Symbols['"']) + 1)
                    if character.isdigit():
                        if int(character) == 0:
                            target.append(10)
                        else:
                            target.append(int(character))
                    elif character in self.Letters:
                        target.append(int(self.Letters[character]))
                    elif character in self.Symbols:
                        target.append(int(self.Symbols[character]))

                padding = torch.zeros(weight - len(target)) + int(self.FullVocab['PAD'])
                targets[count] = torch.hstack((torch.tensor(target), padding))

            input_lengths = torch.full(size=(batch_size,), fill_value=weight, dtype=torch.long)

            target_lengths = torch.zeros(batch_size, dtype=torch.long)

            for i in range(len(labels)):
                target_lengths[i] = len(labels[i])

            # target_lengths = torch.zeros(batch_size)+weight

            ctc_loss = self.ctc_loss(tag_scores, targets, input_lengths, target_lengths)

        mse_loss = 0
        if self.cfg.enable_sr:
            mse_loss = self.mse_loss(sr_img, hr_img)

        loss = mse_loss * self.cfg.lambda_mse + ctc_loss * self.cfg.lambda_ctc

        return loss, mse_loss, ctc_loss

    



