import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SiameseNetwork(nn.Module):
    def __init__(self):

        super(SiameseNetwork, self).__init__()
        
        # Koch et al.
        # Conv2d(input_channels, output_channels, kernel_size) 
        #Input should be 105x105x1
        self.conv1 = nn.Conv2d(1, 64, 7)
        self.conv2 = nn.Conv2d(64, 128, 4)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.conv6 = nn.Conv2d(256, 512, 2)
        self.fc1 = nn.Linear(8192, 4048)
        self.dropout = nn.Dropout(p=0.5)
        self.extraL = nn.Linear(4048, 1)
        #self.conv1 = nn.Conv2d(1, 64, 10)
        #self.conv2 = nn.Conv2d(64, 128, 7)
        #self.conv3 = nn.Conv2d(128, 128, 4)
        #self.conv4 = nn.Conv2d(128, 256, 4)
        #self.conv5 = nn.Conv2d(256, 512, 3)
        #self.conv6 = nn.Conv2d(512, 512, 3)
        #self.fc1 = nn.Linear(9216, 4096)
        #self.dropout = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(4096, 1)
        #self.extraL = nn.Linear(4096, 1)

        # 3 channels 150x150
        #self.conv1 = nn.Conv2d(3, 64, 10)
        #self.conv2 = nn.Conv2d(64, 128, 7)
        #self.conv3 = nn.Conv2d(128, 128, 4)
        #self.conv4 = nn.Conv2d(128, 256, 4)
        #self.fc1 = nn.Linear(30976, 4096)
        #self.fc2 = nn.Linear(4096, 1)

        # 1 channel 200x200
        # self.conv1 = nn.Conv2d(1, 64, 10)
        # self.conv2 = nn.Conv2d(64, 128, 7)
        # self.conv3 = nn.Conv2d(128, 128, 4)
        # self.conv4 = nn.Conv2d(128, 256, 4)
        # self.fc1 = nn.Linear(30976, 4096)

        # using kaiming intialization
        # random source- https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

        # distanceLayer = True  
        # defines if the last layer uses a distance metric or a neuron output
        
    def forward_once(self, x):  

        out = F.relu(F.avg_pool2d(self.conv1(x), 2))
        #print("Output shape of Conv1:", out.shape)
        out = F.relu(F.avg_pool2d(self.conv2(out), 2))
        #print("Output shape of Conv2:", out.shape)
        out = F.relu(F.avg_pool2d(self.conv3(out), 2))
        #print("Output shape of Conv3:", out.shape)
        out = F.relu(F.avg_pool2d(self.conv4(out), 2))
        #print("Output shape of Conv4:", out.shape)
        out = F.relu(F.avg_pool2d(self.conv5(out), 2))
        out = F.relu(self.conv6(out))

        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.extraL(out)
        # out = torch.sigmoid(self.fc1(out)) # #sigmoid as we use BCELoss
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # if self.lastLayer:
        #     # compute l1 distance
        #     diff = torch.abs(output1 - output2)
        #     # score the similarity between the 2 encodings
        #     scores = self.extraL(diff)
        #     return scores
        # else:
        return output1, output2



########### +++++++++++++++ ###########

#create the Siamese Neural Network
# class SiameseNetwork(nn.Module):

#     def __init__(self):
#         super(SiameseNetwork, self).__init__()

#         # Setting up the Sequential of CNN Layers
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=11,stride=4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, stride=2),
            
#             nn.Conv2d(96, 256, kernel_size=5, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(256, 384, kernel_size=3,stride=1),
#             nn.ReLU(inplace=True)
#         )

#         # Setting up the Fully Connected Layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(384, 1024),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(1024, 256),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(256,2)
#         )
        
#     def forward_once(self, x):
#         # This function will be called for both images
#         # Its output is used to determine the similiarity
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

#     def forward(self, input1, input2):
#         # In this function we pass in both images and obtain both vectors
#         # which are returned
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)

#         return output1, output2
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:37:27 2022

@author: OMEN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
class ResNet34(nn.Module):
    def __init__(self, resblock=ResBlock, outputs=2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False),
            resblock(512, 512, downsample=False),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, 1)

    def forward(self, input1, input2):
        input1 = self.layer0(input1)
        input1 = self.layer1(input1)
        input1 = self.layer2(input1)
        input1 = self.layer3(input1)
        input1 = self.layer4(input1)
        #print("layer4 shape", np.shape(input1))
        input1 = self.gap(input1)
        #print("gap shape", np.shape(input1))
        input1 = torch.flatten(input1,start_dim=1)
        #print ("shape", np.shape(input1))
        input1 = self.fc(input1)
        
        input2 = self.layer0(input2)
        input2 = self.layer1(input2)
        input2 = self.layer2(input2)
        input2 = self.layer3(input2)
        input2 = self.layer4(input2)
        input2 = self.gap(input2)
        input2 = torch.flatten(input2,start_dim=1)
        input2 = self.fc(input2)
        #x = F.pairwise_distance(input1, input2)
        #x = torch.abs(input1-input2)
        #x = self.fc(x)
       

        return input1, input2