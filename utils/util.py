#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import string
from IPython import embed

import matplotlib.pyplot as plt
from string import ascii_uppercase, ascii_lowercase


def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    return str_

def check_enabled_grad(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, 'ГРАД НЕТ')
        else:
            print(name, 'ГРАД ЕСТЬ')

def show_image(images, ind):
    un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    temp_image = images[ind, ...].clone().detach()
    img_un = un(temp_image)
    plt.imshow(torch.moveaxis(img_un.cpu(), 0, 2))
    plt.show()

def get_vocab(cfg):
    blank_digits_list = []
    blank_digits_dict = {'BLANK': '0'}
    blank_digits_list.append("_")
    for i in range (1,10):
        blank_digits_list.append(str(i))
        blank_digits_dict[str(i)] = str(i)
    blank_digits_list.append(str(0))
    blank_digits_dict[str(0)] = str(10)

    if cfg.letters == 'all':
        Letters = {letter: str(index) for index, letter in enumerate(ascii_uppercase + ascii_lowercase, start=11)}
    elif cfg.letters == 'lower':
        Letters = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=11)}
    elif cfg.letters == 'upper':
        Letters = {letter: str(index) for index, letter in enumerate(ascii_uppercase, start=11)}
    Symbols = ['+', '|', '[', '%', ':', ',', '>', '@', '$', ')', '©', '-', ';', '!', '&', '?', '.',
                ']', '/', '#', '=', '‰', '(', '\'', '\"', ' ']

    Symbols = {symbols: str(index) for index, symbols in enumerate(Symbols, start=int(Letters[list(Letters)[-1]]) + 1)}
    
    BOS_ind = str(int(Symbols[list(Symbols)[-1]])+1)
    EOS_ind = str(int(Symbols[list(Symbols)[-1]])+2)
    PAD_ind = str(int(Symbols[list(Symbols)[-1]])+3)
    FullVocab = blank_digits_dict | Letters | Symbols | {"BOS": BOS_ind, "EOS": EOS_ind, "PAD": PAD_ind}
    
    FullVocabList = blank_digits_list + list(Letters.keys()) + list(Symbols.keys()) + ['BOS', 'EOS', 'PAD']

    return Letters, Symbols, FullVocab, FullVocabList

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            from IPython import embed
            # embed()
            text = [
                self.dict[char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    # v.data.resize_(data.size()).copy_(data)
    v.resize_(data.size()).copy_(data)

def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

if __name__=='__main__':
    converter = strLabelConverter(string.digits+string.ascii_lowercase)
    embed()