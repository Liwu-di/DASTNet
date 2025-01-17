# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 15:17
# @Author  : 银尘
# @FileName: main5.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : 3 source city DAST
import argparse
import random

import torch
import copy
import time
import os
import numpy as np
import torch.optim as optim
from utils.funcs import load_data, load_all_adj
from utils.funcs import masked_loss
from utils.vec import generate_vector
from model import DASTNet, Domain_classifier_DG
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.research_util import *
import ast

basic_config(logs_style=LOG_STYLE_ALL)


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='4', help='dataset')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--division_seed', type=int, default=0, help='division_seed')
    parser.add_argument('--model', type=str, default='DASTNet', help='model')
    parser.add_argument('--labelrate', type=float, default=23, help='percent')
    parser.add_argument('--patience', type=int, default=200, help='patience')
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--vec_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=64)
    parser.add_argument("--walk_length", "--wl", type=int, default=8)
    parser.add_argument("--num_walks", type=int, default=200)
    parser.add_argument("--theta", type=float, default=1)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument("--q", type=float, default=1)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pre_len", type=int, default=3)
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument('--val', action='store_true', default=False, help='eval')
    parser.add_argument('--test', action='store_true', default=False, help='test')
    parser.add_argument('--train', action='store_true', default=False, help='train')
    parser.add_argument('--etype', type=str, default="gin", choices=["gin"], help='feature type')
    parser.add_argument('--dataname', type=str, default='Taxi', help='Within [Bike, Taxi]')
    parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
    parser.add_argument('--data_amount', type=int, default=3, help='0: full data, 30/7/3 correspond to days of data')
    parser.add_argument('--need_third', type=int, default=0)
    parser.add_argument("--c", type=str, default="default", help="research record")
    parser.add_argument("--machine_code", type=str, default="my-1060", help="code of machine")
    parser.add_argument("--need_remark", type=int, default=0)
    parser.add_argument("--fine_epoch", type=int, default=80)
    parser.add_argument("--need_road", type=bool, default=True)
    parser.add_argument("--cut_data", type=int, default=3312)
    parser.add_argument("--normal", type=int, default=2)
    return parser.parse_args()


def select_mask(a):
    if a == 420:
        return dcmask
    elif a == 476:
        return chimask
    elif a == 460:
        return nymask
    elif a == 1024:
        return bj_mask


def train(dur, model, optimizer, total_step, start_step):
    t0 = time.time()
    train_mae, val_mae, train_rmse, val_rmse, train_acc = list(), list(), list(), list(), list()
    train_correct = 0

    model.train()
    if type == 'pretrain':
        domain_classifier.train()

    for i, (feat, label) in enumerate(train_dataloader.get_iterator()):
        mask = select_mask(feat.shape[2])
        Reverse = False
        if i > 0:
            if train_acc[-1] > 0.333333:
                Reverse = True
        p = float(i + start_step) / total_step
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue

        optimizer.zero_grad()
        if args.model not in ['DCRNN', 'STGCN', 'HA']:
            if type == 'pretrain':
                pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat, shared_pems09_feat = model(vec_pems04,
                                                                                                             vec_pems07,
                                                                                                             vec_pems08,
                                                                                                             vec_pems09,
                                                                                                             feat,
                                                                                                             False,
                                                                                                             args.need_road)
            elif type == 'fine-tune':
                pred = model(vec_pems04, vec_pems07, vec_pems08, vec_pems09, feat, False, args.need_road)

            pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
            label = label.reshape((-1, label.size(2)))

            if type == 'pretrain':
                pems04_pred = domain_classifier(shared_pems04_feat, constant, Reverse)
                pems07_pred = domain_classifier(shared_pems07_feat, constant, Reverse)
                pems08_pred = domain_classifier(shared_pems08_feat, constant, Reverse)
                pems09_pred = domain_classifier(shared_pems09_feat, constant, Reverse)

                pems04_label = 0 * torch.ones(pems04_pred.shape[0]).long().to(device)
                pems07_label = 1 * torch.ones(pems07_pred.shape[0]).long().to(device)
                pems08_label = 2 * torch.ones(pems08_pred.shape[0]).long().to(device)
                pems09_label = 3 * torch.ones(pems09_pred.shape[0]).long().to(device)

                pems04_pred_label = pems04_pred.max(1, keepdim=True)[1]
                pems04_correct = pems04_pred_label.eq(pems04_label.view_as(pems04_pred_label)).sum()
                pems07_pred_label = pems07_pred.max(1, keepdim=True)[1]
                pems07_correct = pems07_pred_label.eq(pems07_label.view_as(pems07_pred_label)).sum()
                pems08_pred_label = pems08_pred.max(1, keepdim=True)[1]
                pems08_correct = pems08_pred_label.eq(pems08_label.view_as(pems08_pred_label)).sum()
                pems09_pred_label = pems09_pred.max(1, keepdim=True)[1]
                pems09_correct = pems09_pred_label.eq(pems09_label.view_as(pems09_pred_label)).sum()

                pems04_loss = domain_criterion(pems04_pred, pems04_label)
                pems07_loss = domain_criterion(pems07_pred, pems07_label)
                pems08_loss = domain_criterion(pems08_pred, pems08_label)
                pems09_loss = domain_criterion(pems09_pred, pems09_label)

                domain_loss = pems04_loss + pems07_loss + pems08_loss + pems09_loss

        if type == 'pretrain':
            train_correct = pems04_correct + pems08_correct + pems07_correct + pems09_correct

        mae_train, rmse_train, mape_train = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                        maskp=mask)

        if type == 'pretrain':
            if i == 1:
                log(mae_train, domain_loss)
            loss = mae_train + args.beta * (args.theta * domain_loss)
        elif type == 'fine-tune':
            loss = mae_train

        loss.backward()
        optimizer.step()

        train_mae.append(mae_train.item())
        train_rmse.append(rmse_train.item())

        if type == 'pretrain':
            train_acc.append(train_correct.item() / 855)
        elif type == 'fine-tune':
            train_acc.append(0)

    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    for i, (feat, label) in enumerate(val_dataloader.get_iterator()):
        mask = select_mask(feat.shape[2])
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue
        pred = model(vec_pems04, vec_pems07, vec_pems08, vec_pems09, feat, True, args.need_road)
        pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
        label = label.reshape((-1, label.size(2)))
        mae_val, rmse_val, mape_val = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                  maskp=mask)
        val_mae.append(mae_val.item())
        val_rmse.append(rmse_val.item())

    test_mae, test_rmse, test_mape = test()
    dur.append(time.time() - t0)
    return np.mean(train_mae), np.mean(train_rmse), np.mean(val_mae), np.mean(
        val_rmse), test_mae, test_rmse, test_mape, np.mean(train_acc)


def test():
    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    test_mape, test_rmse, test_mae = list(), list(), list()

    for i, (feat, label) in enumerate(test_dataloader.get_iterator()):
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        mask = select_mask(feat.shape[2])
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue

        pred = model(vec_pems04, vec_pems07, vec_pems08, vec_pems09, feat, True, args.need_road)
        pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
        label = label.reshape((-1, label.size(2)))

        mae_test, rmse_test, mape_test = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                     maskp=mask)

        test_mae.append(mae_test.item())
        test_rmse.append(rmse_test.item())
        test_mape.append(mape_test.item())

    test_rmse = np.mean(test_rmse)
    test_mae = np.mean(test_mae)
    test_mape = np.mean(test_mape)

    return test_mae, test_rmse, test_mape


def model_train(args, model, optimizer):
    dur = []
    epoch = 1
    best = 999999999999999
    acc = list()

    step_per_epoch = train_dataloader.get_num_batch()
    total_step = 200 * step_per_epoch
    cnt = 0
    while epoch <= args.epoch:
        start_step = epoch * step_per_epoch
        if type == 'fine-tune' and epoch > 1000:
            args.val = True
        mae_train, rmse_train, mae_val, rmse_val, mae_test, rmse_test, mape_test, train_acc = train(dur, model,
                                                                                                    optimizer,
                                                                                                    total_step,
                                                                                                    start_step)
        log(f'Epoch {epoch} | acc_train: {train_acc: .4f} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | mae_test: {mae_test: .4f} | rmse_test: {rmse_test: .4f} | mape_test: {mape_test: .4f} | Time(s) {dur[-1]: .4f}')
        epoch += 1
        acc.append(train_acc)
        if mae_val <= best:
            if type == 'fine-tune' and mae_val > 0.001:
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
            elif type == 'pretrain':
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
        else:
            cnt += 1
        if cnt == args.patience or epoch > args.epoch:
            print(f'Stop!!')
            print(f'Avg acc: {np.mean(acc)}')
            break
    print("Optimization Finished!")
    return state


args = arg_parse(argparse.ArgumentParser())
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
if args.c != "default":
    c = ast.literal_eval(args.c)
    record = ResearchRecord(**c)
    record_id = record.insert(__file__, get_timestamp(), args.__str__())
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.labelrate > 100:
    args.labelrate = 100
from funcs import *


def load_all_adj(device):
    dirs = "./data/{}/{}_roads.npy"
    ny, chi, dc, bj = None, None, None, None
    for i in ["NY", "CHI", "DC", "BJ"]:
        t = dirs.format(i, i)
        t = np.load(t)
        if t.shape[0] < 1000:
            t = t.reshape((t.shape[0] * t.shape[1], t.shape[0] * t.shape[1]))
        t = np.where(t >= 1, 1, t)
        t = add_self_loop(t)
        for m in range(t.shape[0]):
            for n in range(t.shape[1]):
                a, b = idx_1d22d(m, t.shape)
                c, d = idx_1d22d(n, t.shape)
                dis = abs(a - c) + abs(b - d)
                if t[m][n] - 0 > 1e-6 and dis != 0:
                    t[m][n] = t[m][n] / dis
        if t.shape[0] == 460:
            ny = t
        elif t.shape[0] == 476:
            chi = t
        elif t.shape[0] == 420:
            dc = t
        elif t.shape[0] == 1024:
            bj = t

    return torch.tensor(ny).to(device), torch.tensor(chi).to(device), torch.tensor(dc).to(device), torch.tensor(bj).to(
        device)


adj_pems04, adj_pems07, adj_pems08, adj_pems09 = load_all_adj(device)
vec_pems04 = vec_pems07 = vec_pems08 = vec_pems09 = None, None, None, None
dc = np.load("./data/DC/{}DC_{}.npy".format(args.dataname, args.datatype))
dcmask = dc.sum(0) > 0

chi = np.load("./data/CHI/{}CHI_{}.npy".format(args.dataname, args.datatype))
chimask = chi.sum(0) > 0

ny = np.load("./data/NY/{}NY_{}.npy".format(args.dataname, args.datatype))
nymask = ny.sum(0) > 0

bj = np.load("./data/BJ/{}BJ_{}.npy".format(args.dataname, args.datatype))
bj_mask = bj.sum(0) > 0

cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]

pems04_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems04',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems07_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems07',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems08_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems08',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems09_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems09',
                               '{}_vecdim.pkl'.format(args.vec_dim))

for i in [pems04_emb_path, pems07_emb_path, pems08_emb_path, pems09_emb_path]:
    a = i.split(os.path.sep)
    b = []
    for i in a:
        if "pkl" in i:
            continue
        else:
            b.append(i)
    local_path_generate(folder_name=os.path.sep.join(b), create_folder_only=True)

if os.path.exists(pems04_emb_path):
    print(f'Loading pems04 embedding...')
    vec_pems04 = torch.load(pems04_emb_path, map_location='cpu')
    vec_pems04 = vec_pems04.to(device)
else:
    print(f'Generating pems04 embedding...')
    args.dataset = '4'
    vec_pems04, _ = generate_vector(adj_pems04.cpu().numpy(), args)
    vec_pems04 = vec_pems04.to(device)
    print(f'Saving pems04 embedding...')
    torch.save(vec_pems04.cpu(), pems04_emb_path)

if os.path.exists(pems07_emb_path):
    print(f'Loading pems07 embedding...')
    vec_pems07 = torch.load(pems07_emb_path, map_location='cpu')
    vec_pems07 = vec_pems07.to(device)
else:
    print(f'Generating pems07 embedding...')
    args.dataset = '7'
    vec_pems07, _ = generate_vector(adj_pems07.cpu().numpy(), args)
    vec_pems07 = vec_pems07.to(device)
    print(f'Saving pems07 embedding...')
    torch.save(vec_pems07.cpu(), pems07_emb_path)

if os.path.exists(pems08_emb_path):
    print(f'Loading pems08 embedding...')
    vec_pems08 = torch.load(pems08_emb_path, map_location='cpu')
    vec_pems08 = vec_pems08.to(device)
else:
    print(f'Generating pems08 embedding...')
    args.dataset = '8'
    vec_pems08, _ = generate_vector(adj_pems08.cpu().numpy(), args)
    vec_pems08 = vec_pems08.to(device)
    print(f'Saving pems08 embedding...')
    torch.save(vec_pems08.cpu(), pems08_emb_path)

if os.path.exists(pems09_emb_path):
    print(f'Loading pems09 embedding...')
    vec_pems09 = torch.load(pems09_emb_path, map_location='cpu')
    vec_pems09 = vec_pems09.to(device)
else:
    print(f'Generating pems09 embedding...')
    vec_pems09, _ = generate_vector(adj_pems09.cpu().numpy(), args)
    vec_pems09 = vec_pems09.to(device)
    print(f'Saving pems08 embedding...')
    torch.save(vec_pems09.cpu(), pems09_emb_path)

print(
    f'Successfully load embeddings, 4: {vec_pems04.shape}, 7: {vec_pems07.shape}, 8: {vec_pems08.shape}, 9:{vec_pems09.shape}')

domain_criterion = torch.nn.NLLLoss()
domain_classifier = Domain_classifier_DG(num_class=4, encode_dim=args.enc_dim)

domain_classifier = domain_classifier.to(device)
state = g = None, None

batch_seen = 0
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]
assert args.model in ["DASTNet"]

bak_epoch = args.epoch
bak_val = args.val
bak_test = args.test
type = 'pretrain'
pretrain_model_path = os.path.join('{}'.format(cur_dir), 'pretrained', 'transfer_models',
                                   '{}'.format(args.dataset), '{}_prelen'.format(args.pre_len),
                                   'flow_model4_{}_epoch_{}{}{}{}{}{}{}{}{}.pkl'.format(args.model, args.epoch,
                                                                                        args.dataname,
                                                                                        args.datatype,
                                                                                        str(args.learning_rate),
                                                                                        str(args.batch_size),
                                                                                        str(args.split_ratio),
                                                                                        args.seq_len,
                                                                                        args.pre_len,
                                                                                        str(random.random() * 1000000)))

a = pretrain_model_path.split(os.path.sep)
b = []
for i in a:
    if "pkl" not in i:
        b.append(i)
local_path_generate(os.path.sep.join(b), create_folder_only=True)


class DASTNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, device, batch_size, etype, pre_len, dataset, ft_dataset,
                 adj_pems04, adj_pems07, adj_pems08, adj_pems09):
        super(DASTNet, self).__init__()
        self.dataset = dataset
        self.finetune_dataset = ft_dataset
        self.pems04_adj = adj_pems04
        self.pems07_adj = adj_pems07
        self.pems08_adj = adj_pems08
        self.pems09_adj = adj_pems09
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.device = device

        self.pems04_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.pems07_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.pems08_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.pems09_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.shared_pems04_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)
        self.shared_pems07_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)
        self.shared_pems08_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)
        self.shared_pems09_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)

        self.speed_predictor = VGRU_FEAT(hidden_dim=hidden_dim, output_dim=pre_len, encode_dim=encode_dim).to(device)
        self.pems04_linear = nn.Linear(hidden_dim, pre_len, )
        self.pems07_linear = nn.Linear(hidden_dim, pre_len, )
        self.pems08_linear = nn.Linear(hidden_dim, pre_len, )
        self.pems09_linear = nn.Linear(hidden_dim, pre_len, )

        self.weight_feat_private = nn.Parameter(torch.tensor([1.0]).to(self.device))
        self.weight_feat_shared = nn.Parameter(torch.tensor([0.0]).to(self.device))
        self.private_pems04_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.private_pems07_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.private_pems08_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.private_pems09_linear = nn.Linear(hidden_dim, hidden_dim, )

        self.shared_pems04_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.shared_pems07_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.shared_pems08_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.shared_pems09_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems04_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems07_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems08_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems09_linear = nn.Linear(hidden_dim, hidden_dim, )

    def forward(self, vec_pems04, vec_pems07, vec_pems08, vec_pems09, feat, eval, need_road=True, flag=True):
        if self.dataset != self.finetune_dataset or flag == False:
            if not eval:
                shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
                shared_pems09_feat = self.shared_pems09_featExtractor(vec_pems09, self.pems09_adj).to(self.device)
            else:
                if self.dataset == '4' or self.dataset == 'ny':
                    shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                elif self.dataset == '7' or self.dataset == 'chi':
                    shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                elif self.dataset == '8' or self.dataset == 'dc':
                    shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
                elif self.dataset == "9" or self.dataset == "bj":
                    shared_pems09_feat = self.shared_pems09_featExtractor(vec_pems09, self.pems09_adj).to(self.device)
            if self.dataset == '4' or self.dataset == 'ny':
                h_pems04 = shared_pems04_feat.expand(self.batch_size, self.pems04_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems04, need_road)
                pred = self.pems04_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems04_adj.shape[0], -1))
            elif self.dataset == '7' or self.dataset == 'chi':
                h_pems07 = shared_pems07_feat.expand(self.batch_size, self.pems07_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems07, need_road)
                pred = self.pems07_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems07_adj.shape[0], -1))
            elif self.dataset == '8' or self.dataset == 'dc':
                h_pems08 = shared_pems08_feat.expand(self.batch_size, self.pems08_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems08, need_road)
                pred = self.pems08_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems08_adj.shape[0], -1))
            elif self.dataset == '9' or self.dataset == 'bj':
                h_pems09 = shared_pems09_feat.expand(self.batch_size, self.pems09_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems09, need_road)
                pred = self.pems08_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems09_adj.shape[0], -1))

            if not eval:
                return pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat, shared_pems09_feat
            else:
                return pred
        else:
            if self.dataset == '4' or self.dataset == 'ny':
                shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                pems04_feat = self.pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                pems04_feat = self.combine_pems04_linear(
                    self.private_pems04_linear(pems04_feat) + self.shared_pems04_linear(shared_pems04_feat))
                h_pems04 = pems04_feat.expand(self.batch_size, self.pems04_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems04, need_road)
                pred = self.pems04_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems04_adj.shape[0], -1))
            elif self.dataset == '7' or self.dataset == 'chi':
                shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                pems07_feat = self.pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                pems07_feat = self.combine_pems07_linear(
                    self.private_pems07_linear(pems07_feat) + self.shared_pems07_linear(shared_pems07_feat))
                h_pems07 = pems07_feat.expand(self.batch_size, self.pems07_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems07, need_road)
                pred = self.pems07_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems07_adj.shape[0], -1))
            elif self.dataset == '8' or self.dataset == 'dc':
                shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
                pems08_feat = self.pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
                pems08_feat = self.combine_pems08_linear(
                    self.private_pems08_linear(pems08_feat) + self.shared_pems08_linear(shared_pems08_feat))
                h_pems08 = pems08_feat.expand(self.batch_size, self.pems08_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems08, need_road)
                pred = self.pems08_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems08_adj.shape[0], -1))

            elif self.dataset == '9' or self.dataset == 'bj':
                shared_pems09_feat = self.shared_pems09_featExtractor(vec_pems09, self.pems09_adj).to(self.device)
                pems09_feat = self.pems09_featExtractor(vec_pems09, self.pems09_adj).to(self.device)
                pems09_feat = self.combine_pems09_linear(
                    self.private_pems09_linear(pems09_feat) + self.shared_pems09_linear(shared_pems09_feat))
                h_pems09 = pems09_feat.expand(self.batch_size, self.pems09_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems09, need_road)
                pred = self.pems09_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems09_adj.shape[0], -1))

            return pred


args.dataset = "8"
if os.path.exists(pretrain_model_path):
    print(f'Loading pretrained model at {pretrain_model_path}')
    state = torch.load(pretrain_model_path, map_location='cpu')
else:
    print(f'No existing pretrained model at {pretrain_model_path}')
    args.val = args.test = False
    datasets = ["9", "4", "7", "8"]
    dataset_bak = args.dataset
    labelrate_bak = args.labelrate
    args.labelrate = 100
    dataset_count = 0

    for dataset in [item for item in datasets if item not in [dataset_bak]]:
        dataset_count = dataset_count + 1

        print(
            f'\n\n****************************************************************************************************************')
        print(f'dataset: {dataset}, model: {args.model}, pre_len: {args.pre_len}, labelrate: {args.labelrate}')
        print(
            f'****************************************************************************************************************\n\n')

        if dataset == '4':
            g = vec_pems04
        elif dataset == '7':
            g = vec_pems07
        elif dataset == '8':
            g = vec_pems08
        elif dataset == "9":
            g = vec_pems09

        args.dataset = dataset
        if args.cut_data == 3312:
            if args.dataset != "9":
                args.cut_data = 8784
            else:
                args.cut_data = 3312
        train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data(args)
        model = DASTNet(input_dim=args.vec_dim, hidden_dim=args.hidden_dim, encode_dim=args.enc_dim,
                        device=device, batch_size=args.batch_size, etype=args.etype, pre_len=args.pre_len,
                        dataset=args.dataset, ft_dataset=dataset_bak,
                        adj_pems04=adj_pems04, adj_pems07=adj_pems07, adj_pems08=adj_pems08, adj_pems09=adj_pems09).to(
            device)
        optimizer = optim.SGD([{'params': model.parameters()},
                               {'params': domain_classifier.parameters()}], lr=args.learning_rate, momentum=0.8)

        if dataset_count != 1:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optim'])

        state = model_train(args, model, optimizer)

    print(f'Saving model to {pretrain_model_path} ...')
    torch.save(state, pretrain_model_path)
    args.dataset = dataset_bak
    args.labelrate = labelrate_bak
    args.val = bak_val
    args.test = bak_test

type = 'fine-tune'
args.epoch = args.fine_epoch

print(f'\n\n*******************************************************************************************')
print(
    f'dataset: {args.dataset}, model: {args.model}, pre_len: {args.pre_len}, labelrate: {args.labelrate}, seed: {args.division_seed}')
print(f'*******************************************************************************************\n\n')

if args.dataset == '4':
    g = vec_pems04
elif args.dataset == '7':
    g = vec_pems07
elif args.dataset == '8':
    g = vec_pems08
elif args.dataset == "9":
    g = vec_pems09
args.split_ratio = 0.8
train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data(args, cut=True)
model = DASTNet(input_dim=args.vec_dim, hidden_dim=args.hidden_dim, encode_dim=args.enc_dim,
                device=device, batch_size=args.batch_size, etype=args.etype, pre_len=args.pre_len,
                dataset=args.dataset, ft_dataset=args.dataset,
                adj_pems04=adj_pems04, adj_pems07=adj_pems07, adj_pems08=adj_pems08, adj_pems09=adj_pems09).to(device)
optimizer = optim.SGD([{'params': model.parameters()},
                       {'params': domain_classifier.parameters()}], lr=args.learning_rate, momentum=0.8)
model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optim'])

if args.labelrate != 0:
    test_state = model_train(args, model, optimizer)
    model.load_state_dict(test_state['model'])
    optimizer.load_state_dict(test_state['optim'])

test_mae, test_rmse, test_mape = test()
print(f'mae: {test_mae: .4f}, rmse: {test_rmse: .4f}, mape: {test_mape * 100: .4f}\n\n')
if args.c != "default":
    if args.need_remark == 1:
        record.update(record_id, get_timestamp(),
                      "%.4f,%.4f,%.4f" %
                      (test_rmse, test_mae, test_mape * 100),
                      remark="{}C {} {} {} {}".format("2" if args.need_third == 0 else "3", str(args.data_amount),
                                                      args.dataname, args.datatype, args.machine_code))
    else:
        record.update(record_id, get_timestamp(),
                      "%.4f,%.4f, %.4f" %
                      (test_rmse, test_mae, test_mape * 100),
                      remark="{}".format(args.machine_code))
