import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

import numpy as np
import json
import argparse
import random

from torch import optim, nn

from transformers import BertTokenizer

from utils.config import config, setSeed, print_args
from utils.word_encoder import BERTWordEncoder
from utils.data_loader import get_loader
from utils.framework import FewShotNERFramework
from utils.framework_mtnet import FewShotNERFramework_MTNet
from model.proto import Proto, Proto_multiOclass, ProtoMAML
from model.nnshot import NNShot
from model.mtnet import MTNet


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


def main():
    opt = config()
    print_args(opt)

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length

    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    print('mode: {}'.format(opt.mode))

    setSeed(opt.seed)

    print('loading pre-trained language model and tokenizer...')
    if opt.dataset == 'fewcomm':
        UNCASED = './transformer_model/bert-base-chinese'
        VOCAB = 'vocab.txt'
        pretrain_ckpt = opt.pretrain_ckpt or './transformer_model/bert-base-chinese'
        word_encoder = BERTWordEncoder(pretrain_ckpt)
        tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    else:
        UNCASED = './transformer_model/bert-base-uncased'
        VOCAB = 'vocab.txt'
        pretrain_ckpt = opt.pretrain_ckpt or './transformer_model/bert-base-uncased'
        word_encoder = BERTWordEncoder(pretrain_ckpt)
        tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))

    print('loading data...')
    if opt.dataset == 'fewcomm':
        opt.train = f'E:\\task\\test\\data\\FewCOMM\\train.txt'
        opt.test = f'E:\\task\\test\\data\\FewCOMM\\test.txt'
        opt.dev = f'E:\\task\\test\\data\\FewCOMM\\dev.txt'
    else:
        raise NotImplementedError

    if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
        raise RuntimeError('data file is not exist!')

    train_data_loader = get_loader(opt.train, tokenizer,
                                   N=trainN, K=K, Q=Q, batch_size=batch_size, max_length=max_length,
                                   ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data)
    val_data_loader = get_loader(opt.dev, tokenizer,
                                 N=N, K=K, Q=Q, batch_size=1, max_length=max_length, ignore_index=opt.ignore_index,
                                 use_sampled_data=opt.use_sampled_data)
    test_data_loader = get_loader(opt.test, tokenizer,
                                  N=N, K=K, Q=Q, batch_size=1, max_length=max_length, ignore_index=opt.ignore_index,
                                  use_sampled_data=opt.use_sampled_data)

    prefix = '-'.join([model_name, opt.dataset, opt.mode, opt.dataset_mode, str(N), str(K), 'seed' + str(opt.seed),
                       str(int(round(time.time() * 1000)))])
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    print('Loading model...')
    # model_name = 'nnshot'
    if model_name == 'proto':
        print('use proto')
        model = Proto(opt, word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(opt, tokenizer, train_data_loader, val_data_loader, test_data_loader,
                                        use_sampled_data=opt.use_sampled_data)
    elif model_name == 'nnshot':
        print('use nnshot')
        model = NNShot(opt, word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(opt, tokenizer, train_data_loader, val_data_loader, test_data_loader,
                                        use_sampled_data=opt.use_sampled_data)
    elif model_name == 'structshot':
        print('use structshot')
        model = NNShot(opt, word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(opt, tokenizer, train_data_loader, val_data_loader, test_data_loader, N=opt.N,
                                        tau=opt.tau, train_fname=opt.train, viterbi=True,
                                        use_sampled_data=opt.use_sampled_data)
    elif model_name == 'MTNet':
        print('use MTNet')
        model = MTNet(word_encoder, dot=opt.dot, args=opt, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework_MTNet(train_data_loader, val_data_loader, test_data_loader, args=opt,
                                              tokenizer=tokenizer, use_sampled_data=opt.use_sampled_data)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    test = 0

    if test == 0:
        if opt.lr == -1:
            opt.lr = 2e-5


        framework.train(model, prefix,
                        load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        val_step=opt.val_step, fp16=opt.fp16,
                        train_iter=opt.train_iter, warmup_step=int(opt.train_iter * 0.1), val_iter=opt.val_iter,
                        learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    else:
        ckpt = opt.load_ckpt


        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    print('testing...')
    precision, recall, f1, fp, fn, within, outer = framework.eval(model, opt.test_iter, ckpt=ckpt)
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision, recall, f1))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f' % (fp, fn, within, outer))


if __name__ == '__main__':

    default_collate_func = dataloader.default_collate

    setattr(dataloader, 'default_collate', default_collate_override)

    for t in torch._storage_classes:  # 根据Python版本不同进行不同的处理
        if sys.version_info[0] == 2:
            if t in ForkingPickler.dispatch:
                del ForkingPickler.dispatch[t]
        else:
            if t in ForkingPickler._extra_reducers:
                del ForkingPickler._extra_reducers[t]

    main()
