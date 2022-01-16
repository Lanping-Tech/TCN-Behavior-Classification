import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import cross_entropy
import numpy
import torch.backends.cudnn as cudnn
from sklearn import metrics
import argparse
from tqdm import tqdm
import yaml

from dataset import BehaviorDataset
from tcn_model import TCN
from utils import performance_display

from sklearn.metrics import roc_auc_score, roc_curve

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Image Nonlinear Regression'
    )

    # dataset
    parser.add_argument('--train_data_path', default='data/train.txt', type=str, help='train data path')
    parser.add_argument('--test_data_path', default='data/test.txt', type=str, help='test data path')
    parser.add_argument('--vocab_path', default='data/vocab.txt', type=str, help='vocab path')
    parser.add_argument('--train_batch_size', default=100, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size')

    # model
    parser.add_argument('--model_config', default='model_config.yaml', type=str, help='model config')

    parser.add_argument('--epochs', default=20, type=int, help='number of epochs tp train for')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')

    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def test(model, data_loader, device, test=False):

    model = model.eval()
    loss_func = torch.nn.CrossEntropyLoss().to(args.device)
    total = 0
    correct = 0
    totalloss = 0
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            y_pred = model(data)
            loss = loss_func(y_pred, target)
            totalloss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += 1
            correct += (predicted == target).sum().item()
            y_true_all.append(target.cpu().numpy())
            y_pred_all.append(predicted.cpu().numpy())

    acc = correct / total
    loss = totalloss / total
    y_true_all = numpy.concatenate(y_true_all)
    y_pred_all = numpy.concatenate(y_pred_all)
    auc = roc_auc_score(y_true_all, y_pred_all)
    fpr, tpr, thresholds = roc_curve(y_true_all, y_pred_all)
    ks = (tpr-fpr)
    ks = max(ks)
    print("acc: {:.4f}, loss: {:.4f}, auc: {:.4f}, ks: {:.4f}".format(acc, loss, auc, ks))
    return acc, loss, auc, ks


def train(model, train_loader, test_loader, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss().to(args.device)

    train_accs, train_losses, test_accs, test_losses = [], [], [], []
    train_aucs, test_aucs = [], []
    train_kses, test_kses = [], []

    for epoch in range(args.epochs):
        model = model.train()
        pbar = tqdm(train_loader)
        pbar.set_description("Epoch {}:".format(epoch))
        for data, target in pbar:

            data = data.to(args.device)
            target = target.to(args.device)

            optimizer.zero_grad()
            predict = model(data)
            loss = loss_func(predict, target)

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
        
        # val phase
        train_acc, train_loss, train_auc, train_ks = test(model, train_loader, args.device)
        test_acc, test_loss, test_auc, test_ks = test(model, test_loader, args.device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        train_kses.append(train_ks)
        test_kses.append(test_ks)

        print("Epoch {}: train_acc: {:.4f}, train_loss: {:.4f}, test_acc: {:.4f}, test_loss: {:.4f}".format(epoch, train_acc, train_loss, test_acc, test_loss))
        
        # model save
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.output_path+'/epoch_{0}_train_acc_{1:>0.5}_test_acc_{2:>0.5}.ckpt'.format(epoch,train_acc,test_acc))

    acc_plot = {}
    acc_plot['train_acc'] = train_accs
    acc_plot['test_acc'] = test_accs

    loss_plot = {}
    loss_plot['train_loss'] = train_losses
    loss_plot['test_loss'] = test_losses

    auc_plot = {}
    auc_plot['train_auc'] = train_aucs
    auc_plot['test_auc'] = test_aucs

    ks_plot = {}
    ks_plot['train_ks'] = train_kses
    ks_plot['test_ks'] = test_kses

    performance_display(acc_plot, "ACC", args.output_path)
    performance_display(loss_plot, "LOSS", args.output_path)
    performance_display(auc_plot, "AUC", args.output_path)
    performance_display(ks_plot, "KS", args.output_path)
    print('Training finished')
    print("acc: {:.4f}, loss: {:.4f}, auc: {:.4f}, ks: {:.4f}".format(test_accs[-1], test_losses[-1], test_auc[-1], test_ks[-1]))
    torch.save(model.state_dict(), args.output_path+'/final.ckpt')

if __name__ == '__main__':
    args = parse_arguments()
    seed_torch(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load train and test data
    train_data = BehaviorDataset(args.train_data_path, args.vocab_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True)
    test_data = BehaviorDataset(args.test_data_path, args.vocab_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=True)

    # build model
    with open(args.model_config, 'r') as config:
        model_config = yaml.load(config, Loader=yaml.SafeLoader)
    
    model = TCN(**model_config)
    model = model.to(args.device)

    # train phase
    train(model, train_loader, test_loader, args)