#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 22:58:37 2021

@author: ccm
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

crypto_price_df = pd.read_csv("crypto_price(2016-2020).csv",index_col=0)

# Get the return

crypto_ret_df = crypto_price_df.pct_change().dropna()

# Risk Parity

def risk_parity_function(weights , cov):
    sigma = (np.dot(np.dot(weights , cov),weights.T))**(1/2)
    n = len(weights)
    temp = np.dot(cov,weights.T) / sigma
    MC = np.multiply(weights,temp.T)
    
    return sum((sigma/n - MC)**(2))

def risk_parity(data):
    cov = data.cov()
    n = cov.shape[0]
    weights = np.ones(n)/n
    cons = ({'type' : 'eq' , 'fun' :  lambda x : 1-sum(x)})
    bnds = [(0,1) for i in weights]
    res = minimize(risk_parity_function , x0=weights , args = (cov) , method = 'SLSQP' , constraints = cons , bounds = bnds , tol = 1e-30)
    return(res.x)

# Backtest

rolling_window = 365

N = crypto_price_df.shape[0]

for i in range(N - rolling_window):
    crypto_ret = crypto_ret_df[i:i+rolling_window]
    weights = risk_parity(crypto_ret)
    if (i == 0):
        Portfolio = np.matrix(weights)
    else :
        Portfolio = np.append(Portfolio,[weights] , axis = 0)
        
# Performance REview

index = crypto_ret_df.index[365:]

index_df = pd.DataFrame(index)