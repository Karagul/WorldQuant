#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:52:28 2019

@author: sadrachpierre
"""

from wqpt import predict, fit, set_state
import matplotlib.pyplot as plt
# from wqpt import Datahub
import pandas as pd 
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = numpy.random.seed(7)
import numpy as np
from wqpt import predict, fit, set_state
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
pd.options.mode.chained_assignment = None 

def date_mapper(date_str):
    return datetime.strptime(date_str, '%m/%d/%Y')

class Alpha:
    def __init__(self, datafile='wqpt_tutorial_20190304_researchers.csv'):
        self.df = pd.read_csv(datafile)
        self.df_MINUTEMAID  = {}
        self.df_MINUTEMAID['data'] = self.df[['Time Period End Date', 'Brand', 'Dollars', 'Dollars, Promo', 
                                            'Units, Promo', 'Base Dollars', 'Base Units', 'Units', 
                                            "Velocity Dollars", "Velocity Units","Velocity Dollars, Yago",
                                            "Velocity Units, Yago",'discount_percentage']]
        self.df_MINUTEMAID['data']['Base Price'] = self.df_MINUTEMAID['data']['Base Dollars']/self.df_MINUTEMAID['data']['Base Units']
        self.df_MINUTEMAID['data']['Promo Price'] = self.df_MINUTEMAID['data']['Dollars, Promo']/self.df_MINUTEMAID['data']['Units, Promo']
        self.df_MINUTEMAID['data']['list_price'] = self.df_MINUTEMAID['data']['Dollars']/self.df_MINUTEMAID['data'] ['Units']  
        self.df_MINUTEMAID['data']['net_price'] = self.df_MINUTEMAID['data']['list_price']*(1- self.df_MINUTEMAID['data']['discount_percentage'])
        self.df_MINUTEMAID['data'].dropna(inplace = True)
        self.df_MINUTEMAID['data']['Time Period End Date'] = pd.to_datetime(self.df_MINUTEMAID['data']['Time Period End Date'], format='%m/%d/%Y')
        self.START_DATE = self.df_MINUTEMAID['data']['Time Period End Date'].loc[0]
        self.df_MINUTEMAID['data']['weeks_since_start'] =  (self.df_MINUTEMAID['data']['Time Period End Date']  - self.START_DATE).dt.days // 7
        self.df_MINUTEMAID['data'].sort_values('Time Period End Date', inplace = True)  
        self.training_max_date = None
        self.models = {}        
    def build_neural_network(self, input_neurons, number_of_layers):
        self.model = Sequential()
        self.model.add(Dense(input_neurons, kernel_initializer='normal', input_dim=3, activation='relu'))
        for i in range(1, number_of_layers):
            self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mape', optimizer='adam', metrics=['mape'])
        return self.model 
    def set_state(self, date):
        self.training_max_date =  date_mapper(date)
    def fit_models(self):
        if self.training_max_date is None:
            raise Exception(
                'attempting to fit models before any data '
                'is made available')
        mask = (self.df_MINUTEMAID['data']['Time Period End Date'] > datetime(self.training_max_date.year, self.training_max_date.day, self.training_max_date.month)) 
        X = np.array(self.df_MINUTEMAID['data'].loc[mask][['list_price', 'discount_percentage', 'weeks_since_start']])
        y = np.array(self.df_MINUTEMAID['data'].loc[mask]["Velocity Units"])
        scalar = MinMaxScaler()
        scalar.fit(X)
        X = scalar.transform(X)
        model = self.build_neural_network(12, 10)
        reg = model.fit(X, y, epochs=500, batch_size=10)
        self.models['reg'] = reg
    def predict(self, list_price, discount_percentage, date):
        START_DATE = date_mapper('02/07/2016')
        weeks_since_start= (date_mapper(date)  - START_DATE).days // 7
        result = self.model.predict(np.array([[list_price, discount_percentage, weeks_since_start]]))
        return result if result > 0 else 0
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    def remove_outliers(self):
        pass 
    
ALPHA_INSTANCE = Alpha()


@set_state
def set_state(time_period_end: str):
    """Sets state for the alpha model

    Parameters
    ----------
    date: str
        Newest date for which data should be made available for the
        alpha to fit its underlying model(s)
    """
    ALPHA_INSTANCE.set_state(date=time_period_end)


@fit
def fit():
    """(Re-)fits the alpha model using the latest state"""
    ALPHA_INSTANCE.fit_models()


@predict
def predict(time_period_end: str, list_price: float, discount_percent: float):
    """
    Predict demand for the given SKU in the given store on the given date

    Parameters
    ----------
    list_prices: dict of dict of float
        Dictionary of prices for all SKUs across all stores for the
        given date. It should have structure {store_id: {sku_id: price}}
    discount_percentages: dict of dict of float
        Dictionary of discount percentages for all SKUs across all stoers for
        the given date. It should have structure {store_id: {sku_id: dscount_perc}}
    date: str
        Date in MM-DD-YYYY format

    Returns
    -------
    float
        Predicted demand on date

    """
    result = ALPHA_INSTANCE.predict(list_price=list_price, discount_percentage=discount_percent, date=time_period_end)
    return float(result[0])


def main():
    ALPHA_INSTANCE.remove_outliers()
    print('setting state to "10/02/2017" ...')
    ALPHA_INSTANCE.set_state('10/02/2017')
    print('ok')
    print('fitting models...')
    ALPHA_INSTANCE.fit_models()
    print('ok')
    print('calling predict...')
    test_params =[{'list_price':  2.378013, 'discount_percentage': 0.167605, 'date': '05/02/2019'},
                  {'list_price':  2.172965, 'discount_percentage': 0.119188, 'date': '05/09/2019'},
                  {'list_price':  1.745326, 'discount_percentage': 0.122433, 'date': '05/16/2019'},
                  {'list_price':  1.050123, 'discount_percentage': 0.152387, 'date': '05/25/2019'},
                  {'list_price':  2.378013, 'discount_percentage': 0.092456, 'date': '06/01/2019'},
                  {'list_price':  2.598546, 'discount_percentage': 0.127363, 'date': '06/08/2019'},
                  {'list_price':  2.398546, 'discount_percentage': 0.127363, 'date': '06/15/2019'}]
    result = []
    for params in test_params:
        result = ALPHA_INSTANCE.predict(**params)
        print('parameters = ', params)
        print('  -> demand =', result)
    mask = (ALPHA_INSTANCE.df_MINUTEMAID['data']['Time Period End Date'] > datetime(2016, 2, 7)) & \
           (ALPHA_INSTANCE.df_MINUTEMAID['data']['Time Period End Date'] < datetime(2017, 2, 7))
    y_true = ALPHA_INSTANCE.df_MINUTEMAID['data']["Velocity Units"].loc[mask]
    x_pred = ALPHA_INSTANCE.df_MINUTEMAID['data'].loc[mask][['list_price', 'discount_percentage', 'weeks_since_start']]   
    scalar = MinMaxScaler()
    scalar.fit(x_pred)
    x_pred = scalar.transform(x_pred)
    y_pred = ALPHA_INSTANCE.model.predict(x_pred)
    error_value = ALPHA_INSTANCE.mean_absolute_percentage_error(y_true[0], y_pred[0])
    print("Mean Absolute Percent Error (MAPE) is: ",  error_value)
    
if __name__ == '__main__':
    main()