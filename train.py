from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


from utils import load_data, accuracy, macro_f1, micro_f1, macro_precision, macro_recall,f1, precision, recall
from models import HGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhidden', type=int, default=10, help='Number of node level hidden units.')
parser.add_argument('--shidden', type=int, default=8, help='Number of semantic level hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--nd_dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--se_dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=10, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if torch.cuda.is_available():
     device = torch.device("cuda")
     torch.cuda.set_device(0)
else:
     device = torch.device("cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adjs, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
t_feat = features[0].shape[1]
nfeat_list = []
for i in range(1,len(features)):
    nfeat_list.append(features[i].shape[1])    




#adjs = torch.FloatTensor(adjs)
#features = torch.FloatTensor(features)
for a in range(len(adjs)):
    adjs[a] = torch.FloatTensor(adjs[a])
    if args.cuda:
        adjs[a] = adjs[a].cuda()
for f in range(len(features)):
    features[f] = torch.FloatTensor(features[f])  
    if args.cuda:
        features[f] = features[f].cuda()
labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
model = HGAT(tfeat = t_feat,
             nfeat_list=nfeat_list, 
             nhid=args.nhidden, 
             shid=args.shidden,
             nclass=int(labels.max()) + 1, 
             nd_dropout=args.nd_dropout,
             se_dropout=args.se_dropout,
             nheads=args.nb_heads, 
             alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
#    features = features.cuda()
#    adjs = adjs.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#features, adjs, labels = Variable(features), Variable(adjs), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adjs)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adjs)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item(),loss_train.item()


def compute_test():
    model.eval()
    output = model(features, adjs)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
#    mac = macro_f1(output[idx_test], labels[idx_test])  
#    mac_pre = macro_precision(output[idx_test], labels[idx_test])  
#    mac_rec = macro_recall(output[idx_test], labels[idx_test])  
#    print("Test set results:",
#          "loss= {:.4f}".format(loss_test.item()),
#          "accuracy= {:.4f}".format(acc_test.item()),
#          "macro_f1= {:.4f}".format(mac),
#          "macro_precision= {:.4f}".format(mac_pre),
#          "macro_recall= {:.4f}".format(mac_rec))
    mac = f1(output[idx_test], labels[idx_test])  
    mac_pre = precision(output[idx_test], labels[idx_test])  
    mac_rec = recall(output[idx_test], labels[idx_test])   
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "macro_f1= {:.4f}".format(mac),
          "macro_precision= {:.4f}".format(mac_pre),
          "macro_recall= {:.4f}".format(mac_rec))
    
# Train model
print("start training!")
t_total = time.time()
loss_values = []
loss_values_output = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch)[0])
    loss_values_output.append(train(epoch)[1])
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name)

# Testing
compute_test()
#print(len(loss_values_output))
#print(loss_values_output)
#print(len(loss_values))
#print(loss_values)