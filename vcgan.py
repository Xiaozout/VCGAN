import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import copy

from scipy import io
from tqdm import tqdm
from collections import defaultdict
from model import RAE, UAE, Generator, Discriminator
from torch.utils.data import DataLoader
from data import loadData, to_Tensor, computeTopNAccuracy


def train_R(epoch):
    R.train()
    r_loss_value = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(args.device)
        r_optimizer.zero_grad()
        rv, recon = R(data)
        r_loss = loss_MSE(recon, data)
        r_loss.backward()
        r_loss_value += r_loss.item()
        r_optimizer.step()
    if (epoch%10 == 0):
        print('\r[RAE Epoch %d/%d] : [loss: %.4f]' %(epoch, args.f_epochs, r_loss.item()))
    return r_loss_value/len(train_loader.dataset)

def train_U(epoch):
    U.train()
    u_loss_value = 0
    for batch_idx, data in enumerate(userinfo_loader):
        data = data.to(args.device)
        u_optimizer.zero_grad()
        uv, recon_u = U(data)
        u_loss = loss_MSE(recon_u, data)
        u_loss.backward()
        u_loss_value += u_loss.item()
        u_optimizer.step()
    if (epoch%10 ==0):
        print('\r[UAE Epoch %d/%d] : [loss: %.4f]' %(epoch, args.f_epochs, u_loss.item()))
    return u_loss_value/len(userinfo_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Variational Collaborative Generative Adversarial Networks')
    parser.add_argument('-e', '--epochs', type= int, default = 200, help = 'number of epochs to train (default: 200)')
    parser.add_argument('-ef', '--f_epochs', type= int, default = 1000, help = 'number of epochs of feature extrator (default: 1000)')
    parser.add_argument('--lr',type = float, default = 1e-3, help = 'learning rate')

    parser.add_argument('--dir', default='/Users/Xiaozout/Desktop/碩士研究相關/Models/Movielen', help = 'dataset directory')
    parser.add_argument('--data', default='data', help = 'specify dataset')
    parser.add_argument('--gpu', action = 'store_true', default = False, help = 'enables CUDA training')
    parser.add_argument('--log', type = int, default = 1, metavar = 'N',
                        help = 'how many batches to wait before logging training status')
    parser.add_argument('-N', type = int, default = 20, help = 'number of recommended items')
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=0.75)
    parser.add_argument('--pretrain', default = False, action='store_true', help = 'Feature Extractor')
    parser.add_argument('--save', help='save model', default = False, action='store_true')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')

    print('dataset directory: ' + args.dir)
    directory = args.dir + '/' + args.data

    path = '{}/u1.base'.format(directory)
    trainSet, train_user, train_item = loadData(path, '\t', 'Train')
   	
    path = '{}/u1.test'.format(directory)
    testSet, test_user, test_item = loadData(path, '\t', 'Test')

    path = '{}/movielen_userInfo_2.csv'.format(directory)
    UserInfo = pd.read_csv(path, encoding = "utf_8_sig")
    UserInfo = UserInfo.values
    UserInfo = torch.tensor(UserInfo.astype(np.float32)).to(args.device)

    userCount = max(train_user, test_user)
    itemCount = max(train_item, test_item)
    trainTesor , testMaskTensor = to_Tensor(trainSet, testSet, userCount, itemCount)

    args.dim = itemCount + 64 + 64
    loss_MSE = nn.MSELoss()
    G_step = 5
    D_step = 2
    topN = 20
    best_precisions = 0

    R = RAE(itemCount).to(args.device)
    U = UAE(UserInfo.shape[1]).to(args.device)
    G = Generator(args.dim, itemCount).to(args.device)
    D = Discriminator(itemCount).to(args.device)

    r_optimizer = torch.optim.Adam(R.parameters(), lr = args.lr)
    u_optimizer =  torch.optim.RMSprop(U.parameters(), lr = 0.01, weight_decay = 0.5)
    g_optimizer = torch.optim.Adam(G.parameters(), lr = args.lr)
    d_optimizer = torch.optim.Adam(D.parameters(), lr = args.lr)

    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones = [30, 200, 400], gamma=0.1)
    d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones = [30, 200, 400], gamma=0.1)

    rating = Variable(copy.deepcopy(trainTesor)).to(args.device)
    profiles = Variable(copy.deepcopy(UserInfo)).to(args.device)
    train_loader = DataLoader(rating, 100, shuffle=True)
    userinfo_loader = DataLoader(profiles, 100, shuffle=True)

    if args.pretrain:
        for epoch in tqdm(range(1, args.f_epochs+1), unit = 'epoch'):
            rloss_value = train_R(epoch)
            if epoch == 1:
                best_rloss = rloss_value
            elif (rloss_value < best_rloss):
                path = args.dir +'/rae'
                torch.save(R.state_dict(), path)
                best_rloss = rloss_value

        for epoch in tqdm(range(1, args.f_epochs+1), unit = 'epoch'):
            uloss_value = train_U(epoch)
            if epoch == 1:
                best_uloss = uloss_value
            elif (uloss_value < best_uloss):
                path = args.dir +'/uae' 
                torch.save(U.state_dict(), path)
                best_uloss = uloss_value
        R.load_state_dict(torch.load(args.dir +'/rae'))
        U.load_state_dict(torch.load(args.dir +'/uae'))
    else:
        R.load_state_dict(torch.load(args.dir +'/rae'))
        U.load_state_dict(torch.load(args.dir +'/uae'))

    rv, _ = R(rating)
    uv, _ = U(profiles)
    G_input = torch.cat([rating, rv], 1)
    G_input = torch.cat([G_input, uv], 1)

    for epoch in tqdm(range(1, args.epochs+1)):
        G.train()
        D.train()
        g_scheduler.step()
        d_scheduler.step()
        for step in range(G_step):
            
            fakeData, mu, logvar = G(G_input)
            fakeData_result = D(fakeData)
            g_loss = args.alpha * torch.mean(torch.log(1. - fakeData_result.detach() + 10e-5))  + args.beta * loss_MSE(fakeData, rating) #+ nn.BCEWithLogitsLoss()(fakeData, rating)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            
        for step in range(D_step):
            
            fakeData, _, _ = G(G_input)
            fakeData_result = D(fakeData) 
            realData_result = D(rating) 
            d_loss = -torch.mean(torch.log(realData_result.detach()+10e-5) + torch.log(1. - fakeData_result.detach()+10e-5)) + 0*loss_MSE(fakeData,rating)  
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
        
        G.eval()
        result_all, _, _ = G(G_input)
        if(epoch%5 == 0):
            
            n_user = len(testSet)
            index = 0
            recalls_5, recalls_10, recalls_20 = 0, 0, 0
            precisions_5, precisions_10, precisions_20 = 0, 0, 0
            
            for testUser in testSet.keys():
                result = result_all[testUser] + Variable(copy.deepcopy(testMaskTensor[index])).to(args.device)
                recall_5, precision_5, recall_10, precision_10, recall_20, precision_20 = computeTopNAccuracy(testSet[testUser], result, topN)
                
                recalls_5 += recall_5
                precisions_5 += precision_5
                recalls_10 += recall_10
                precisions_10 += precision_10
                recalls_20 += recall_20
                precisions_20 += precision_20
                
                index+=1
                
            recalls_5 /= n_user    
            precisions_5 /= n_user
            recalls_10 /= n_user    
            precisions_10 /= n_user
            recalls_20 /= n_user    
            precisions_20 /= n_user
            
            print('\nEpoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f} ==> recall@{}:{:.4f}, precision@{}:{:.4f}, recall@{}:{:.4f}, precision@{}:{:.4f}, recall@{}:{:.4f}, precision@{}:{:.4f}'.format(epoch, args.epochs,
            d_loss.item(), g_loss.item(), 5, recalls_5, 5, precisions_5, 10, recalls_10, 10, precisions_10, 20, recalls_20, 20, precisions_20))


            # if epoch == 5:
            #     best_precisions = precisions_5
            # elif args.save and (precisions_5 > best_precisions):
            #     name = 'vcgan'
            #     path = directory + '/save_model/' + name + '_g'
            #     G.cpu()
            #     torch.save(G.state_dict(), path)