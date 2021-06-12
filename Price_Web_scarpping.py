#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 00:17:12 2021

@author: ccm
"""
from cryptocmd import CmcScraper

# initialise scraper with time interval
scraper = CmcScraper("USDT" , "01-01-2021", "04-06-2021")

# get raw data as list of list
headers, data = scraper.get_data()

# get dataframe for the data
USDT_df = scraper.get_dataframe()


data = {'BTC' : BTC_df['Close'] , 'ETH' : ETH_df['Close'] , 'LTC' : LTC_df['Close'] , 'XRP' : XRP_df['Close'] , 'USDT' : USDT_df['Close']}

data_df = pd.DataFrame(data)

data_df.index = BTC_df['Date']

data_df.to_csv('crypto_price(2021).csv')