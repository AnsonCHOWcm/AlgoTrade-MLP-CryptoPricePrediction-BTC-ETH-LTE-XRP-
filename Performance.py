#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 01:58:15 2021

@author: ccm
"""
import numpy as np

# Define a class for Performance Measurements
class Measures :
    
    def __init__(self , data):
        self.data = data
        
        
    def Cumulative_Return(self):
        return(self.data[-1]/self.data[0] -1)
    
    def Annualized_GM(self):
        return ((self.data[-1]/self.data[0])**(365/(len(self.data)-1))-1)
    
    def Annualized_Vol(self):
        ret = np.log(self.data[1:]/self.data[:-1])-1
        return(np.std(ret) * 16)
    
    def Annualized_Sharpe(self):
        return((self.Annualized_GM()-(0.016/12))/self.Annualized_Vol())
    
    def Sortino_Ratio(self):
        ret = np.log(self.data[1:]/self.data[:-1])-1
        neg_ret = ret[np.where(ret<0)]
        neg_vol = np.std(neg_ret)
        return ((self.Annualized_GM()-(0.01))/neg_vol)
    
    def MaxDrawDown(self):
        mdd = 0
        peak = self.data[0]
        for x in self.data:
         if x > peak: 
            peak = x
         dd = (peak - x) / peak
         if dd > mdd:
            mdd = dd
        return mdd
    
    def CalmarRatio(self):
        return (self.Annualized_GM() / self.MaxDrawDown())
    
    
    
    
    
    
        
                