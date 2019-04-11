# Author: J.-H. Jacobsen.
#
# Modified from Pytorch examples code.
# Original license shown below.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from models.model_utils import * #get_all_params, halfswap, lerp

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import torchvision
import torch.nn.functional as F

import os
import sys
import math
import numpy as np

from sklearn.decomposition import PCA


from .iRevNet_digits import netCritic

criterion = nn.CrossEntropyLoss()
criterion_s = nn.BCELoss()
'''mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}
'''
'''
# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def train(model, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, in_shape):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    criticN = netCritic()
    criticN.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)
    optimizerCritic = optim.SGD(criticN.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        torchvision.utils.save_image(inputs,"inputs1.jpg")
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        
        inputs, targets = Variable(inputs), Variable(targets)

        out, out_bij = model(inputs)               # Forward Propagation
        
        '''out_new = out_bij.view(out_bij.shape[0],-1)
                
        out_embedded = TSNE().fit_transform(out_new.detach())
        
        
        
        #####################################################
        label_color_map={0:'red',
                         1:'blue',
                         2:'green',
                         3:'yellow',
                         4:'black',
                         5:'cyan',
                         6:'orange',
                         7:'brown',
                         8:'pink',
                         9:'olive',
                }
        label_color = [label_color_map[l] for l in np.array(targets)]
        #####################################################
        
        plt.scatter(out_embedded[:,0],out_embedded[:,1], c= label_color,label = label_color_map)
        plt.legend()
        plt.savefig("train_scatter.jpg")
        plt.clf()'''
    
        '''alpha = torch.rand(out.shape[0],1).to('cuda')/2
                    
        swap_out_bij = halfswap(out_bij)
        swap_out_targets = halfswap(targets)
        
        label_error = (targets == swap_out_targets).to('cuda',dtype=torch.float32)

        
        interpolate = lerp(out_bij, swap_out_bij, alpha)
        
            
        image1 = model.inverse(out_bij)
        image2 = model.inverse(swap_out_bij)
        image3 = model.inverse(interpolate)'''
   
        '''torchvision.utils.save_image(image1,"train_inverse_1.jpg")
        torchvision.utils.save_image(image2,"train_inverse_2.jpg")
        torchvision.utils.save_image(image3,"train_interpolated_regularized.jpg")'''
        ########################################################################
        out_inv = model.inverse(out_bij)
        #disc = criticN(torch.lerp(inputs, out_inv,0.3))

        alpha = torch.rand(out.shape[0],1).to('cuda')/2

        #print("adsf",out_bij.shape)
        z_mix  = lerp(out_bij,swap_halves(out_bij),alpha)

        out_mix  = model.inverse(z_mix)
        #disc_mix = criticN(out_mix)

        loss_class = criterion(out, targets)
        loss_ae_mse = F.mse_loss(out_inv, inputs)
        #loss_ae_l2 = L2(disc_mix) * 0.3
        loss_ae = loss_ae_mse + loss_class# + loss_ae_l2 

        loss_ae.backward(retain_graph = True)  # Backward Propagation
        optimizer.step()  # Optimizer update

        '''loss_disc_mse = F.mse_loss(disc_mix, alpha.reshape(-1))
        loss_disc_l2 = L2(disc)
        loss_disc = loss_disc_mse + loss_disc_l2

        optimizerCritic.zero_grad()
        loss_disc.backward()
        optimizerCritic.step()'''

        image1 = model.inverse(out_bij)
        image3 = model.inverse(z_mix)
        
        torchvision.utils.save_image(image1,"input_digits.jpg")
        # torchvision.utils.save_image(image2,"interpolated2.jpg")
        torchvision.utils.save_image(image3,"interpolated_mix_digits.jpg")
        
        train_loss += loss_ae.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
            
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss_ae.data[0], 100.0*correct/total))
        sys.stdout.flush()
        

        ########################################################################
        '''critic_class,critic_mix = criticN.forward(image3)
        
        
        loss_recon = F.mse_loss(image1,inputs)
        loss_disc_mse = F.mse_loss(critic_mix, alpha.view(-1))
        loss_class = criterion_s(critic_class, label_error)
        loss_disc_l2 = L2(critic_mix)
        loss_disc = loss_disc_mse +  loss_disc_l2 +loss_recon #+ loss_class
        
        
    
        loss = criterion(out, targets)  # Loss
        loss.backward(retain_graph = True)  # Backward Propagation
        optimizer.step()  # Optimizer update

        optimizerCritic.zero_grad()
        loss_disc.backward(retain_graph=True)
        optimizerCritic.step()



        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
            
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.data[0], 100.0*correct/total))
        sys.stdout.flush()'''


def test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij = model(inputs)
        
        out_new = out_bij.view(out_bij.shape[0],-1)
        
        out_embedded = TSNE().fit_transform(out_new.detach())
        
        label_color_map={0:'red',
                         1:'blue',
                         2:'green',
                         3:'yellow',
                         4:'black',
                         5:'cyan',
                         6:'orange',
                         7:'brown',
                         8:'pink',
                         9:'olive',
                }
        label_color = [label_color_map[l] for l in np.array(targets)]
        
        plt.scatter(out_embedded[:,0],out_embedded[:,1], c= label_color,label = label_color_map)
        plt.legend()
        plt.savefig("test_scatter_digits.jpg")
        plt.clf()
    
        alpha = torch.ones(out.shape[0],1).to('cuda')
        
            
        swap_out_bij = swap_halves(out_bij)
        swap_out_targets = swap_halves(targets)
        
        label_error = (targets == swap_out_targets).to('cuda',dtype=torch.float32)
#        print('error=',label_error)
        image1 = model.inverse(out_bij)
        image2 = model.inverse(swap_out_bij)
        
        torchvision.utils.save_image(image1,"test_inverse_1_digits.jpg")
        torchvision.utils.save_image(image2,"test_inverse_2_digits.jpg")
        
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
            interpolate = lerp(out_bij, swap_out_bij, alpha*i)
            image3 = model.inverse(interpolate)
   
            torchvision.utils.save_image(image3,"test_interpolated_regularized_%f_digits.jpg"%i)
    
    
        loss = criterion(out, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
                'model': model if use_cuda else model,
                'acc': acc,
                'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/office'+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+fname+'.t7')
        best_acc = acc
    return best_acc
