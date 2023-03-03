'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')
import time

from tensorboardX import SummaryWriter
from os.path import join

BOARD_PATH = "/home/ismp/sda1/kaiwei/NGCF-PyTorch/NGCF/board"

if __name__ == '__main__':
    #print("============gpu_id==============")
    #print(args.gpu_id)
    args.device = torch.device('cuda:' + str(args.gpu_id))
    #print(args.device)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    print("=========================================")

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    print("========create model ====================")
    print("=========================================")


    # init tensorboard
    if args.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.comment)
                                        )
        print("enable tensorflowboard")
    else:
        w = None
        print("not enable tensorflowboard")

    t0 = time.time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    print("=========================================")
    print("========start training===================")
    print("=========================================")
    for epoch in range(args.epoch):
        print("===============epoch : %d==========================" % epoch)
        t1 = time.time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if args.tensorboard:
            w.add_scalar(f'BPRLoss/loss', loss, epoch+1)
            w.add_scalar(f'BPRLoss/mf_loss', mf_loss, epoch+1)
            w.add_scalar(f'BPRLoss/emb_loss', emb_loss, epoch+1)

        # *********************************************************
        # 如果 epoch 不是 10 的倍數，則下面都不做 
        if (epoch + 1) % args.round_verbose!= 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch+1, time.time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        # *********************************************************
        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, w, epoch, drop_flag=False)

        t3 = time.time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs = %.1fs + %.1fs]: train == [%.5f = %.5f + %.5f], recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch+1, t3 - t1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
#        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
#            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
#            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)


    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter = [%d]@[%.1f]\nrecall = [%s],\nprecision = [%s],\nhit = [%s],\nndcg = [%s]" % \
                 (idx, 
                  time.time() - t0, 
                  ' '.join(['%.5f' % r for r in recs[idx]]),
                  ' '.join(['%.5f' % r for r in pres[idx]]),
                  ' '.join(['%.5f' % r for r in hit[idx]]),
                  ' '.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    if args.tensorboard:
        w.close()