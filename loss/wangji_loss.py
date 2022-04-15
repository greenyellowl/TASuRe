import torch
import torch.nn as nn
from string import ascii_uppercase, ascii_lowercase
import numpy as np

class WangjiLoss(nn.Module):
    def __init__(self, args, cfg):
        super(WangjiLoss, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)

        self.LETTERS = {letter: str(index) for index, letter in enumerate(ascii_uppercase + ascii_lowercase, start=11)}
        SYMBOLS = ['+', '|', '[', '%', ':', ',', '>', '@', '$', ')', '©', '-', ';', '!', '&', '?', '.',
                   ']', '/', '#', '=', '‰', '(', '\'', '\"', ' ']

        self.SYMBOLS = {symbols: str(index) for index, symbols in enumerate(SYMBOLS, start=int(self.LETTERS["z"]) + 1)}

        self.cfg = cfg
        # self.ce_loss = nn.CrossEntropyLoss()
        # self.l1_loss = nn.L1Loss()
        # self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # self.english_dict = {}
        # for index in range(len(self.english_alphabet)):
        #     self.english_dict[self.english_alphabet[index]] = index
        #
        # self.build_up_transformer()


    def forward(self, sr_img, tag_scores, hr_img, labels):

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

        mse_loss = self.mse_loss(sr_img, hr_img)
        ctc_loss = self.ctc_loss(tag_scores, targets, input_lengths, target_lengths)

        loss = mse_loss + ctc_loss * self.cfg.lambda_ctc
        return loss, mse_loss, ctc_loss