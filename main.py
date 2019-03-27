import time
import pickle
import os

import argparse
import torch

from torch import nn, optim
from tensorboardX import SummaryWriter

from train import train, test
import dataset

import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-datasetname', type=str)
parser.add_argument('-model', type=str)
parser.add_argument('-batch_size', type=int, default=10)
parser.add_argument('-input_dim', type=int, default=128)
parser.add_argument('-epochs', type=int, default=200)
parser.add_argument('-device_num', type=int, default=0)
parser.add_argument('-multi_gpu', type=int, nargs='+')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('-enc_layer', type=int, nargs='+')
parser.add_argument('-g_theta_layer', type=int, nargs='+')
parser.add_argument('-f_phi_layer', type=int, nargs='+')
parser.add_argument('-classifier_layer', type=int, nargs='+')
parser.add_argument('-embedding_dim', type=int)
parser.add_argument('-q_att_layer', type=int, nargs='+')
parser.add_argument('-img_att_layer', type=int, nargs='+')
parser.add_argument('-fixed_embed', action='store_true', default=False)
parser.add_argument('-rnn_dim', type=int)
parser.add_argument('-rnn_layer_size', type=int)
parser.add_argument('-vocab_size', type=int, default=82) # 81 for CLEVR + 1(0 index for '_' pad)
parser.add_argument('-answer_vocab_size', type=int, default=28)
parser.add_argument('-beta', type=float, default=1.0)
parser.add_argument('-restore', type=str, default='')
parser.add_argument('-option', type=str, default='')

args = parser.parse_args()


torch.cuda.set_device(args.device_num)
device = torch.device("cuda:{}".format(args.device_num))
# device = torch.device('cpu')



if args.multi_gpu:
    multiplier = len(args.multi_gpu)
else:
    multiplier = 1


qa_dir =  '/home/jinwon/Relational_Network/data/CLEVR_v1.0' \
                  '/processed_data'

if 'CLEVR' in args.datasetname:
    with open(os.path.join(qa_dir, 'idx_word_dict.pkl'), 'rb') as f:
        idx_to_word_dict = pickle.load(f)
        idx_to_question = idx_to_word_dict['idx_to_question']
        idx_to_question_type = idx_to_word_dict['idx_to_question_type']
        idx_to_answer = idx_to_word_dict['idx_to_answer']
elif 'GQA' in args.datasetname:
    q_type_to_idx = {'choose': 0,
                        'compare': 1,
                        'logical': 2,
                        'query': 3,
                        'verify': 4}
    idx_to_question_type = {v:k for k, v in q_type_to_idx.items()}

if args.restore:

    log_dir = args.restore
    model_dir = os.path.join(log_dir, 'model')

    import sys

    sys.path[0] = f'result/{args.datasetname}/{args.restore}/model'

    if 'CLEVR' in args.datasetname:
        if args.model == 'base':
            from base import Model
        elif args.model == 'topdown':
            from topdown_model import Model
        elif args.model == 'base_bert':
            from base_bert import Model
        elif args.model == 'base_q_att':
            from base_q_att import Model
    elif 'GQA' in args.datasetname:
        if args.model == 'base':
            from base_gqa_spatial import Model

    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'optimizer.pkl'), 'rb') as f:
        optimizer = pickle.load(f)

    saved_model_list = [x for x in os.listdir(model_dir) if x.endswith('.pt')]
    saved_model_list = sorted(saved_model_list,
                              key=lambda x: int(x.split('_')[-1].split('.pt')[0]))

    latest_model = saved_model_list[-1]



    checkpoint = torch.load(os.path.join(model_dir, latest_model))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

else:

    if 'CLEVR' in args.datasetname:
        if args.model == 'topdown':
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            img_att_layer = '-'.join([str(x) for x in args.img_att_layer])
            log_dir = f'result/{args.datasetname}/' \
                f'{args.model}_i_{args.input_dim}_f_{f_phi_layer}_img_att{img_att_layer}' \
                f'e_{args.embedding_dim}_r_{args.rnn_dim}_' \
                f'b_{args.batch_size}_fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
        elif args.model == 'base':
            g_theta_layer = '-'.join([str(x) for x in args.g_theta_layer])
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            log_dir = f'result/{args.datasetname}/' \
                f'{args.model}_i_{args.input_dim}_g_{g_theta_layer}_f_{f_phi_layer}_' \
                f'e_{args.embedding_dim}_r_{args.rnn_dim}_' \
                f'b_{args.batch_size}_fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
        elif args.model == 'base_bert':
            g_theta_layer = '-'.join([str(x) for x in args.g_theta_layer])
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            log_dir = f'result/{args.datasetname}/{args.model}_i_{args.input_dim}_' \
                f'g_{g_theta_layer}_f_{f_phi_layer}_r_{args.rnn_dim}_b_{args.batch_size}_' \
                f'fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
        elif args.model == 'base_q_att':
            g_theta_layer = '-'.join([str(x) for x in args.g_theta_layer])
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            q_att_layer = '-'.join([str(x) for x in args.q_att_layer])
            log_dir = f'result/{args.datasetname}/' \
                f'{args.model}_i_{args.input_dim}_g_{g_theta_layer}_f_{f_phi_layer}_' \
                f'e_{args.embedding_dim}_r_{args.rnn_dim}_at_{q_att_layer}' \
                f'b_{args.batch_size}_fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
    elif 'GQA' in args.datasetname:
        if args.model == 'topdown':
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            img_att_layer = '-'.join([str(x) for x in args.img_att_layer])
            log_dir = f'result/{args.datasetname}/' \
                f'{args.model}_f_{f_phi_layer}_img_att{img_att_layer}' \
                f'e_{args.embedding_dim}_r_{args.rnn_dim}_' \
                f'b_{args.batch_size}_fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
        elif args.model == 'base':
            g_theta_layer = '-'.join([str(x) for x in args.g_theta_layer])
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            log_dir = f'result/{args.datasetname}/' \
                f'{args.model}_g_{g_theta_layer}_f_{f_phi_layer}_' \
                f'e_{args.embedding_dim}_r_{args.rnn_dim}_' \
                f'b_{args.batch_size}_fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
        elif args.model == 'base_bert':
            g_theta_layer = '-'.join([str(x) for x in args.g_theta_layer])
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            log_dir = f'result/{args.datasetname}/{args.model}_i_{args.input_dim}_' \
                f'g_{g_theta_layer}_f_{f_phi_layer}_r_{args.rnn_dim}_b_{args.batch_size}_' \
                f'fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'
        elif args.model == 'base_q_att':
            g_theta_layer = '-'.join([str(x) for x in args.g_theta_layer])
            f_phi_layer = '-'.join([str(x) for x in args.f_phi_layer])
            q_att_layer = '-'.join([str(x) for x in args.q_att_layer])
            log_dir = f'result/{args.datasetname}/' \
                f'{args.model}_g_{g_theta_layer}_f_{f_phi_layer}_' \
                f'e_{args.embedding_dim}_r_{args.rnn_dim}_at_{q_att_layer}' \
                f'b_{args.batch_size}_fixed_embed_{args.fixed_embed}_gpu_{multiplier}_{args.option}'

    model_dir = os.path.join(log_dir, 'model')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if 'CLEVR' in args.datasetname:




        if args.model == 'base':
            from base import Model
            model_script = 'base.py'
        elif args.model == 'topdown':
            from topdown_model import Model
            model_script = 'topdown_model.py'
        elif args.model == 'base_bert':
            from base_bert import Model
            model_script = 'base_bert.py'
        elif args.model == 'base_q_att':
            from base_q_att import Model
            model_script = 'base_q_att.py'
    elif 'GQASpatial' == args.datasetname:
        if args.model == 'base':
            from base_gqa_spatial import Model
            model_script = 'base_gqa_spatial.py'
    elif 'GQAObjects' == args.datasetname:
        if args.model == 'base':
            from base_gqa_object import Model
            model_script = 'base_gqa_object.py'

    shutil.copyfile(model_script, os.path.join(model_dir, model_script))
    shutil.copyfile('ops.py', os.path.join(model_dir, 'ops.py'))

    model = Model(**vars(args))
    # model = Model(args.g_theta_layer,
    #               args.f_phi_layer,
    #               args.embedding_dim,
    #               args.rnn_dim,
    #               args.answer_vocab_size, args.fixed_embed)

    optimizer = optim.Adam(model.parameters(), lr=2.5 * 1e-4)

    start_epoch = 0

    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, 'optimizer.pkl'), 'wb') as f:
        pickle.dump(optimizer, f)

if args.multi_gpu:
    model = nn.DataParallel(model, device_ids=args.multi_gpu)
model.to(device)



is_bert = args.model == 'base_bert'
train_loader, test_loader, input_dim = dataset.load_data(args.datasetname,
                                                         args.batch_size * multiplier,
                                                         args.input_dim,
                                                         is_bert,
                                                         multiplier)



writer = SummaryWriter(log_dir)


for epoch in range(1 + start_epoch, args.epochs + 1 + start_epoch):

    # if epoch < 5:
    #     optimizer = warmup_lr_scheduler(optimizer, epoch, init_lr=2.5 * 1e-4,
    #                                     max_lr=args.batch_size / 64 * multiplier *2.5*1e-4)
    # elif epoch == 5:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    print('epoch', epoch, 'lr', optimizer.state_dict()['param_groups'][0]['lr'])

    start_time = time.time()
    train_loss, train_acc, train_acc_list = train(model, train_loader, optimizer,
                                                  idx_to_question_type, epoch, device)
    print("train {}--- {} seconds ---".format(epoch, (time.time() - start_time)))

    start_time = time.time()

    if epoch % 1 == 0:
        test_loss, test_acc, test_acc_list = test(model, test_loader, idx_to_question_type,
                                                  epoch, device)
        print("test {}--- {} seconds ---".format(epoch, (time.time() - start_time)))

        for idx, q_type in idx_to_question_type.items():
            writer.add_scalar('test_acc_sub_{}'.format(q_type), test_acc_list[idx],
                              epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        # if epoch >= 5:
        #     scheduler.step(test_loss)

        print(optimizer)
        print(log_dir)

    for idx, q_type in idx_to_question_type.items():
        writer.add_scalar('train_acc_sub_{}'.format(q_type), train_acc_list[idx],
                          epoch)

    writer.add_scalar('train_loss', train_loss, epoch)

    writer.add_scalar('train_acc', train_acc, epoch)

    saved_model_list = [x for x in os.listdir(model_dir) if x.endswith('.pt')]
    saved_model_list = sorted(saved_model_list,
                              key=lambda x: int(x.split('_')[-1].split('.pt')[0]))

    if len(saved_model_list) >= 5:
        os.remove(os.path.join(model_dir, saved_model_list[0]))

    if epoch % 5 == 0:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(model_dir, 'model_{}.pt'.format(epoch)))