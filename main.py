import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import yaml
from easydict import EasyDict
from interfaces.super_resolution import TextSR

torch.manual_seed(1234)
from multiprocessing import set_start_method


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def main(config, args):
    TextSuperResolution = TextSR(config, args)

    if args.test:
        TextSuperResolution.test()
    elif args.demo:
        TextSuperResolution.demo()
    else:
        TextSuperResolution.train()


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='wangji', choices=['wangji', 'tbsrn', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn'])
    parser.add_argument('--exp_name', required=False, help='Type your experiment name', default='aa')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../dataset/lmdb/str/TextZoom/test/medium/', help='')
    # parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--arch', default='tbsrn', choices=['tbsrn', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
#                                                            'edsr', 'lapsrn'])
#     parser.add_argument('--text_focus', action='store_true')
#     parser.add_argument('--exp_name', required=True, help='Type your experiment name')
#     parser.add_argument('--test', action='store_true', default=False)
#     parser.add_argument('--test_data_dir', type=str, default='./dataset/mydata/test')
#     parser.add_argument('--batch_size', type=int, default=None, help='')
#     parser.add_argument('--resume', type=str, default='', help='')
#     parser.add_argument('--rec', default='crnn', choices=['crnn'])
#     parser.add_argument('--STN', action='store_true', default=False, help='')
#     parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
#     parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
#     parser.add_argument('--mask', action='store_true', default=False, help='')
#     parser.add_argument('--hd_u', type=int, default=32, help='')
#     parser.add_argument('--srb', type=int, default=5, help='')
#     parser.add_argument('--demo', action='store_true', default=False)
#     parser.add_argument('--demo_dir', type=str, default='./demo')
#     args = parser.parse_args()
#     config_path = os.path.join('config', 'super_resolution.yaml')
#     config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
#     config = EasyDict(config)
#     main(config, args)