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
        
        self.FULL_VOCAB = blank_digits_dict | self.LETTERS | self.SYMBOLS | {"BOS": self.cfg.BOS, "EOS": self.cfg.EOS, "PAD": self.cfg.PAD}
        
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


        a =self.ctc_decode(tag_scores)


        return loss, mse_loss, ctc_loss

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
        string = ''
        dig_string = beam_results[0][0][:out_lens[0][0]]
        for dig in dig_string:
            dig2 = int(dig)
            string += list(self.FULL_VOCAB.keys())[list(self.FULL_VOCAB.values()).index(str(dig2))]
            # string += self.FULL_VOCAB.keys()[self.FULL_VOCAB.values().index(str(dig2))]
        aaaa = 'a'
        return "aaa"