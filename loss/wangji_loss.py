from tokenize import blank_re
import torch
import torch.nn as nn
from string import ascii_uppercase, ascii_lowercase
import numpy as np
from einops import rearrange

from ctcdecode import CTCBeamDecoder

class WangjiLoss(nn.Module):
    def __init__(self, args, cfg):
        super(WangjiLoss, self).__init__()
        self.cfg = cfg
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)

        self.LETTERS = {letter: str(index) for index, letter in enumerate(ascii_uppercase + ascii_lowercase, start=11)}
        SYMBOLS = ['+', '|', '[', '%', ':', ',', '>', '@', '$', ')', '©', '-', ';', '!', '&', '?', '.',
                   ']', '/', '#', '=', '‰', '(', '\'', '\"', ' ']

        self.SYMBOLS = {symbols: str(index) for index, symbols in enumerate(SYMBOLS, start=int(self.LETTERS["z"]) + 1)}

        blank_digits_list = []
        blank_digits_dict = {'BLANK': '0'}
        blank_digits_list.append("_")
        for i in range (1,10):
            blank_digits_list.append(str(i))
            blank_digits_dict[str(i)] = str(i)
        blank_digits_list.append(str(0))
        blank_digits_dict[str(0)] = str(10)
        
        self.FULL_VOCAB = blank_digits_dict | self.LETTERS | self.SYMBOLS | {"BOS": str(self.cfg.BOS), "EOS": str(self.cfg.EOS), "PAD": str(self.cfg.PAD)}
        
        self.FULL_VOCAB_LIST = blank_digits_list + list(self.LETTERS.keys()) + list(self.SYMBOLS.keys()) + ['bos', 'eos', 'pad']

        # self.ce_loss = nn.CrossEntropyLoss()
        # self.l1_loss = nn.L1Loss()
        # self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # self.english_dict = {}
        # for index in range(len(self.english_alphabet)):
        #     self.english_dict[self.english_alphabet[index]] = index
        #
        # self.build_up_transformer()


    def forward(self, sr_img, tag_scores, hr_img, labels):

        ctc_loss = 0
        if self.cfg.enable_rec:
            targets = torch.zeros(self.cfg.batch_size, 32) + self.cfg.PAD
            # label = label.upper()

            for count, label in enumerate(labels):
                target = []
                for character in label:
                    # print(character)
                    # if character == ' ':
                    #     target.append(int(self.SYMBOLS['"']) + 1)
                    if character.isdigit():
                        if int(character) == 0:
                            target.append(10)
                        else:
                            target.append(int(character))
                    elif character in self.LETTERS:
                        target.append(int(self.LETTERS[character]))
                    elif character in self.SYMBOLS:
                        target.append(int(self.SYMBOLS[character]))

                padding = torch.zeros(32 - len(target))+self.cfg.PAD
                targets[count] = torch.hstack((torch.tensor(target), padding))

            input_lengths = torch.full(size=(self.cfg.batch_size,), fill_value=32, dtype=torch.long)

            target_lengths = torch.zeros(self.cfg.batch_size, dtype=torch.long)

            for i in range(len(labels)):
                target_lengths[i] = len(labels[i])

            # target_lengths = torch.zeros(self.cfg.batch_size)+32

            ctc_loss = self.ctc_loss(tag_scores, targets, input_lengths, target_lengths)

        mse_loss = 0
        if self.cfg.enable_sr:
            mse_loss = self.mse_loss(sr_img, hr_img)

        loss = mse_loss * self.cfg.lambda_mse + ctc_loss * self.cfg.lambda_ctc

        return loss, mse_loss, ctc_loss

    



