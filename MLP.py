#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 22:59:20 2021

@author: ccm
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset , DataLoader



class NetWork(nn.Module):
    
    def __init__(self ,input_nodes, hidden_nodes ,output_nodes):
        super().__init__()
        
        self.input_node = input_nodes
        self.hidden_node = hidden_nodes
        self.output_node = output_nodes
        
        self.fc1 = nn.Linear(self.input_node , self.hidden_node[0])
        self.fc2 = nn.Linear(self.hidden_node[0] , self.hidden_node[1])
        self.fc3 = nn.Linear(self.hidden_node[1] , self.hidden_node[2])
        self.fc4 = nn.Linear(self.hidden_node[2] , self.output_node)
        
        self.dropout = nn.Dropout(p =0.2)
        self.ReLu = nn.ReLU()
        self.LogSoftMax = nn.LogSoftmax(dim = 1)
    
    def forward_reg(self , x):
        
        x = x.view(x.shape[0] , -1)
        
        x = self.fc1(x)
        x = self.dropout(self.ReLu(x))
        x = self.fc2(x)
        x = self.dropout(self.ReLu(x))
        x = self.fc3(x)
        x = self.dropout(self.ReLu(x))
        x = self.fc4(x)
        x = self.ReLu(x)
        
        return x
    
    def forward_class(self , x):
        
        x = x.view(x.shape[0] , -1)
        
        x = self.fc1(x)
        x = self.dropout(self.ReLu(x))
        x = self.fc2(x)
        x = self.dropout(self.ReLu(x))
        x = self.fc3(x)
        x = self.dropout(self.ReLu(x))
        x = self.fc4(x)
        x = self.LogSoftMax(x)
        
        return x
        
        
       
       
       
       
       
            
            

        
        