#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 23:16:05 2021

@author: ccm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import MLP
import Performance
import torch
from torch import nn , optim
from torch.utils.data import TensorDataset , DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def generate_dataset(price , seq_len):
    X_list , y_list = [] , []
    for i in range(len(price) - seq_len):
        X = np.array(price[i : i+seq_len])
        y = np.array(price[i+seq_len ])
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list) , np.array(y_list)

def daily_weight(price , model):
    weight = []
    model = model
    pred = model.forward_reg(price)
    for i in range(len(pred)):
        weight.append(1 if pred[i] > price[i][-1] else 0)
        
    return np.array(weight)


#Load the Train and test Dataset

train_crypto_price_df = pd.read_csv("crypto_price(2016-2020).csv",index_col=0)

train_price_df = train_crypto_price_df.drop(labels = 'USDT' , axis = 1).loc["01-01-2017" : ]

test_crypto_price_df = pd.read_csv("crypto_price(2021).csv",index_col=0)

test_price_df = test_crypto_price_df.drop(labels = 'USDT' , axis = 1)

##Setting HyperParaMeter

bs = 64
learning_rate = {'BTC' : 0.1 , 'ETH' : 0.001 , 'LTC' : 0.005 , 'XRP' : 0.0005}
window = 3

input_nodes = window
hidden_nodes = [512,256,128]
output_nodes = 1

epochs = 100

## Initialize the Variable

training_loss = 0
testing_loss = 0
steps = 0

print_every = 5

Train_performance  = []

Test_performance = []

Train_loss = []

Vaild_loss = []

Model = []

## Training the model and Predict the Price

for i in train_price_df.columns:
    
    training_losses = []
        
    testing_losses = []
    
    ## Prepare for the data
    
    sc = StandardScaler()

    train_price = sc.fit_transform(np.array(train_price_df[i]).reshape(-1,1))

    test_price = sc.transform(np.array(test_price_df[i]).reshape(-1,1))

    X , y = generate_dataset(train_price , window)

    X_test , y_test = generate_dataset(test_price , window)
    
    X_train = X[:-50]
    y_train = y[:-50]
        
    X_vaild = X[-50:]
    y_vaild = y[-50:]
        
    tensor_X_train = torch.Tensor(X_train)
    tensor_y_train = torch.Tensor(y_train)
        
    tensor_X_vaild = torch.Tensor(X_vaild)
    tensor_y_vaild = torch.Tensor(y_vaild)
        
    Training_set = TensorDataset(tensor_X_train , tensor_y_train)
    Vaild_set = TensorDataset(tensor_X_vaild , tensor_y_vaild)
        
    Trainloader = DataLoader(Training_set , batch_size = bs , shuffle = True)
    Vaildloader = DataLoader(Vaild_set , batch_size = bs , shuffle =True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Load the Model
    model = MLP.NetWork(input_nodes , hidden_nodes , output_nodes)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr = learning_rate[i]) if i == 'BTC' else optim.Adam(model.parameters(),lr = learning_rate[i])
    criterion = nn.MSELoss()
    
    for e in range(epochs):
        
        training_loss = 0
        
        for inputs , result in Trainloader :
            
            steps += 1
        
            inputs , result = inputs.to(device) , result.to(device)
        
            optimizer.zero_grad()
        
            outputs = model.forward_reg(inputs)
            loss = criterion(outputs , result)
            loss.backward()
            optimizer.step()
        
            training_loss += loss.item()

            model.eval()
            
        with torch.no_grad():   
                    
            for inputs , result in Vaildloader :
                        
                        inputs , result = inputs.to(device) , result.to(device)
        
                        optimizer.zero_grad()
        
                        outputs = model.forward_reg(inputs)
                        loss = criterion(outputs , result)
        
                        testing_loss += loss.item()
                    
            print(f"Epoch {e+1}/{epochs}.. "
                          f"Train loss: {training_loss/(print_every * steps):.3f}.."
                          f"Test loss: {testing_loss/len(Vaildloader):.3f}")
                    

            training_losses.append(training_loss / len(Trainloader))
            testing_losses.append(testing_loss / len(Vaildloader))
                    
            model.train()
            testing_loss = 0
                        
                        
      
    ## Predict the price on Train set
    
    tensor_x_train  = torch.Tensor(X)

    tensor_x_train = tensor_x_train.view(tensor_x_train.shape[0] , -1)

    pred_price_train = model.forward_reg(tensor_x_train)

    pred_price_train_array = sc.inverse_transform(pred_price_train.detach().numpy())
    
    ## Predict the price on Test set
    
    start = time.time()
                
    tensor_x_test  = torch.Tensor(X_test)

    tensor_x_test = tensor_x_test.view(tensor_x_test.shape[0] , -1)

    pred_price_test = model.forward_reg(tensor_x_test)

    pred_price_test_array = sc.inverse_transform(pred_price_test.detach().numpy())
    
    print ("Time used in prediction : " , time.time() - start)
    
    Train_performance.append(pred_price_train_array)
    
    Test_performance.append(pred_price_test_array)
    
    Train_loss.append(training_losses)
    
    Vaild_loss.append(testing_losses)
    
    Model.append(model)
    
## Plotting the graph to see the Training Process

fig, axs = plt.subplots(2,2)

axs[0,0].plot(range(len(Train_loss[0])),Train_loss[0], label='Training loss')
axs[0,0].plot(range(len(Vaild_loss[0])),Vaild_loss[0], label='Validation loss')
axs[0,0].legend(frameon=False)
axs[0,0].set_title('BTC')

axs[0,1].plot(range(len(Train_loss[1])),Train_loss[1], label='Training loss')
axs[0,1].plot(range(len(Vaild_loss[1])),Vaild_loss[1], label='Validation loss')
axs[0,1].legend(frameon=False)
axs[0,1].set_title('ETH')

axs[1,0].plot(range(len(Train_loss[2])),Train_loss[2], label='Training loss')
axs[1,0].plot(range(len(Vaild_loss[2])),Vaild_loss[2], label='Validation loss')
axs[1,0].legend(frameon=False)
axs[1,0].set_title('LTC')

axs[1,1].plot(range(len(Train_loss[3])),Train_loss[3], label='Training loss')
axs[1,1].plot(range(len(Vaild_loss[3])),Vaild_loss[3], label='Validation loss')
axs[1,1].legend(frameon=False)
axs[1,1].set_title('XRP')
    
## Ploting the Graph to see the Prediction Similarity

X_train , y_train = generate_dataset(np.array(train_price_df) , window)

X_test , y_test = generate_dataset(np.array(test_price_df) , window)



plt.plot(Test_performance[0] , label = "Predict")
plt.plot(y_test[:,0] , label = "True")
plt.title("Prediction on BTC")
plt.ylabel("price" )
plt.xlabel("trading days" )
plt.legend(loc = "lower right" )
plt.show()


plt.plot(Test_performance[1] , label = "Predict")
plt.plot(y_test[:,1] , label = "True")
plt.title("Prediction on ETH")
plt.ylabel("price" )
plt.xlabel("trading days" )
plt.legend(loc = "lower right" )
plt.show()


plt.plot(Test_performance[2] , label = "Predict")
plt.plot(y_test[:,2] , label = "True")
plt.title("Prediction on LTE")
plt.ylabel("price" )
plt.xlabel("trading days" )
plt.legend(loc = "lower right" )
plt.show()


plt.plot(Test_performance[3] , label = "Predict")
plt.plot(y_test[:,3] , label = "True")
plt.title("Prediction on XRP")
plt.ylabel("price" )
plt.xlabel("trading days" )
plt.legend(loc = "lower right" )
plt.show()

## Portfolio with Smart Beta 

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

# MSR

def Sharpe_Ratio_function(weights, ret):
 mu = ret.mean()
 cov = ret.cov()
 numator = np.dot(weights , mu - 0.016/12)
 demonmator = (np.dot(np.dot(weights,cov),weights))**(1/2)
 return (-1 * numator / demonmator)

def Maximum_Sharpe_Ratio(data, long = 1):
 cov = data.cov()
 n = cov.shape[0]
 weights = np.ones(n) /n
 cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)}) 
 bnds = [(0 ,0.1) for i in weights]
 if long == 1:
  res = minimize(Sharpe_Ratio_function, x0 = weights, args = (data), method = 'SLSQP', constraints = cons,
  bounds = bnds, tol = 1e-30)
 else:
  res = minimize(Sharpe_Ratio_function, x0 = weights, args = (data), method = 'SLSQP', constraints = cons, tol = 1e-30)
 return res.x

## Backktest (Train Set)

## BTC Investment

BTC_weight =[]

for i in range(len(y_train[:,0] )-1):
    
    BTC_weight.append(1 if Train_performance[0][i+1] > y_train[:,0][i] else 0)
    
BTC_weight_arr = np.array(BTC_weight)

BTC_daily_ret = np.array(pd.DataFrame(y_train[:,0]).pct_change().dropna())

BTC_cumret = np.cumprod(BTC_weight_arr * BTC_daily_ret.T + 1)

mea = Performance.Measures(BTC_cumret)

BTC_cumret_df = pd.DataFrame(BTC_cumret)
    
BTC_cumret_df.plot()

## ETH Investment

ETH_weight =[]

for i in range(len(y_train[:,1] )-1):
    
    ETH_weight.append(1 if Train_performance[1][i+1] > y_train[:,1][i] else 0)
    
ETH_weight_arr = np.array(ETH_weight)

ETH_daily_ret = np.array(pd.DataFrame(y_train[:,1]).pct_change().dropna())

ETH_cumret = np.cumprod(ETH_weight_arr * ETH_daily_ret.T + 1)

mea = Performance.Measures(ETH_cumret)

ETH_cumret_df = pd.DataFrame(ETH_cumret)
    
ETH_cumret_df.plot()

## LTE Investment

LTE_weight =[]

for i in range(len(y_train[:,2] )-1):
    
    LTE_weight.append(1 if Train_performance[2][i+1] > y_train[:,2][i] else 0)
    
LTE_weight_arr = np.array(LTE_weight)

LTE_daily_ret = np.array(pd.DataFrame(y_train[:,2]).pct_change().dropna())

LTE_cumret = np.cumprod(LTE_weight_arr * LTE_daily_ret.T + 1)

mea = Performance.Measures(LTE_cumret)

LTE_cumret_df = pd.DataFrame(LTE_cumret)
    
LTE_cumret_df.plot()

#XRP Investment 

XRP_weight =[]

for i in range(len(y_train[:,3] )-1):
    
    XRP_weight.append(1 if Train_performance[3][i+1] > y_train[:,3][i] else 0)
    
XRP_weight_arr = np.array(XRP_weight)

XRP_daily_ret = np.array(pd.DataFrame(y_train[:,3]).pct_change().dropna())

XRP_cumret = np.cumprod(XRP_weight_arr * XRP_daily_ret.T + 1)

mea = Performance.Measures(XRP_cumret)

XRP_cumret_df = pd.DataFrame(XRP_cumret)
    
XRP_cumret_df.plot()


asset_weight = [BTC_weight_arr , ETH_weight_arr , LTE_weight_arr , XRP_weight_arr]

rolling_window = 30


N = len(asset_weight[0])

backtest = train_crypto_price_df['05-12-2016':]

backtest_ret_df = backtest.pct_change().dropna()

train_crypto_ret_df = train_crypto_price_df['01-01-2017':].pct_change().dropna()

for i in range(N):
    crypto_ret = backtest_ret_df[i:i+rolling_window]
    weights = Maximum_Sharpe_Ratio(crypto_ret)
    weights[0] = weights[0]*asset_weight[0][i]
    weights[1] = weights[1]*asset_weight[1][i]
    weights[2] = weights[2]*asset_weight[2][i]
    weights[3] = weights[3]*asset_weight[3][i]
    weights[4] = 1-np.sum(weights[:-1]) 
    if (i == 0):
        Portfolio = np.matrix(weights)
    else :
        Portfolio = np.append(Portfolio,[weights] , axis = 0)
        
Port_ret = np.sum(np.array(Portfolio) * np.array(train_crypto_ret_df[window:]) , axis = 1 )

Port_cumret = np.cumprod(Port_ret + 1)

mea = Performance.Measures(Port_cumret)
        
Port_df = pd.DataFrame(Port_cumret)

Port_df.plot()

# Backtest (Test Set)
        
## BTC Investment

BTC_weight =[]

for i in range(len(y_test[:,0] )-1):
    
    BTC_weight.append(1 if Test_performance[0][i+1] > y_test[:,0][i] else 0)
    
BTC_weight_arr = np.array(BTC_weight)

BTC_daily_ret = np.array(pd.DataFrame(y_test[:,0]).pct_change().dropna())

BTC_cumret = np.cumprod(BTC_weight_arr * BTC_daily_ret.T + 1)

mea = Performance.Measures(BTC_cumret)

BTC_cumret_df = pd.DataFrame(BTC_cumret)
    
BTC_cumret_df.plot()

## ETH Investment

ETH_weight =[]

for i in range(len(y_test[:,1] )-1):
    
    ETH_weight.append(1 if Test_performance[1][i+1] > y_test[:,1][i] else 0)
    
ETH_weight_arr = np.array(ETH_weight)

ETH_daily_ret = np.array(pd.DataFrame(y_test[:,1]).pct_change().dropna())

ETH_cumret = np.cumprod(ETH_weight_arr * ETH_daily_ret.T + 1)

mea = Performance.Measures(ETH_cumret)

ETH_cumret_df = pd.DataFrame(ETH_cumret)
    
ETH_cumret_df.plot()

## LTE Investment

LTE_weight =[]

for i in range(len(y_test[:,2] )-1):
    
    LTE_weight.append(1 if Test_performance[2][i+1] > y_test[:,2][i] else 0)
    
LTE_weight_arr = np.array(LTE_weight)

LTE_daily_ret = np.array(pd.DataFrame(y_test[:,2]).pct_change().dropna())

LTE_cumret = np.cumprod(LTE_weight_arr * LTE_daily_ret.T + 1)

mea = Performance.Measures(LTE_cumret)

LTE_cumret_df = pd.DataFrame(LTE_cumret)
    
LTE_cumret_df.plot()

#XRP Investment 

XRP_weight =[]

for i in range(len(y_test[:,3] )-1):
    
    XRP_weight.append(1 if Test_performance[3][i+1] > y_test[:,3][i] else 0)
    
XRP_weight_arr = np.array(XRP_weight)

XRP_daily_ret = np.array(pd.DataFrame(y_test[:,3]).pct_change().dropna())

XRP_cumret = np.cumprod(XRP_weight_arr * XRP_daily_ret.T + 1)

mea = Performance.Measures(XRP_cumret)

XRP_cumret_df = pd.DataFrame(XRP_cumret)
    
XRP_cumret_df.plot()


asset_weight = [ETH_weight_arr , LTE_weight_arr , XRP_weight_arr]

rolling_window = 30

test_crypto_ret_df = test_crypto_price_df.pct_change().dropna()

N = len(asset_weight[0])

train = train_crypto_price_df.iloc[- 27 :]

test = test_crypto_price_df

backtest = train.append(test)

backtest_ret_df = backtest.pct_change().dropna()

for i in range(N):
    crypto_ret = backtest_ret_df[i:i+rolling_window]
    weights = Maximum_Sharpe_Ratio(crypto_ret)
    weights[0] = 0
    weights[1] = weights[1]*asset_weight[0][i]
    weights[2] = weights[2]*asset_weight[1][i]
    weights[3] = weights[3]*asset_weight[2][i]
    weights[4] = 1-np.sum(weights[:-1]) 
    if (i == 0):
        Portfolio = np.matrix(weights)
    else :
        Portfolio = np.append(Portfolio,[weights] , axis = 0)
        
Port_ret = np.sum(np.array(Portfolio) * np.array(test_crypto_ret_df[window:]) , axis = 1 )

Port_cumret = np.cumprod(Port_ret + 1)

mea = Performance.Measures(Port_cumret)
        
Port_df = pd.DataFrame(Port_cumret)

Port_df.plot()

        
    
    
    
    
    
    
    
    

    
    
        
        
    







    
    
        