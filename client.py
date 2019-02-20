import torch

import dataset
import time
import pickle
import argparse
import rn
from torch import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument('-datasetname', type=str)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-input_dim', type=int, default=128)
parser.add_argument('-epochs', type=int, default=1000)
parser.add_argument('-device_num', type=int, default=0)
parser.add_argument('-multi_gpu', type=int, nargs='+')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('-z_dim', type=int)
parser.add_argument('-enc_layer', type=int, nargs='+')
parser.add_argument('-classifier_layer', type=int, nargs='+')
parser.add_argument('-embedding_dim', type=int)
parser.add_argument('-rnn_dim', type=int)
parser.add_argument('-rnn_layer_size', type=int)
parser.add_argument('-vocab_size', type=int)
parser.add_argument('-answer_vocab_size', type=int)
parser.add_argument('-beta', type=float, default=1.0)
parser.add_argument('-restore', action='store_true', default=False)
parser.add_argument('-option', type=str, default='')

args = parser.parse_args()

device = torch.device("cuda:{}".format(args.device_num))

if args.multi_gpu:
    multiplier = len(args.multi_gpu)
else:
    multiplier = 1

train_loader, test_loader, input_dim = dataset.load_data(args.datasetname,
                                                         args.batch_size * multiplier,
                                                         args.input_dim,
                                                         multiplier)

model = VAE(input_dim, args.enc_layer, args.embedding_dim, args.rnn_dim,
                 args.vocab_size, args.answer_vocab_size)
if args.multi_gpu:
    model = nn.DataParallel(model, device_ids=args.multi_gpu)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)