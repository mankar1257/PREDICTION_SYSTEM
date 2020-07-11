# --------------------------IMPORTS----------------------------------

import pandas as pd 
import datetime
import calendar
#from pymongo import MongoClient
import numpy as np 
from models import sim_avg,mov_avg,exp_smooth,Holt_linear,Holt_Winter,Auto_ARIMA
from models import M_LinearRegression , M_Ridge , M_RidgeCV ,  M_Lasso , M_ElasticNet , M_LassoLars ,  M_BayesianRidge , M_HuberRegressor , M_ElasticNetCV ,  M_SGDRegressor

import datetime as dt
from datetime import datetime
from datetime import timedelta

from Data import purchase_data

from sklearn.model_selection import train_test_split 
from dateutil.relativedelta import relativedelta

len_to_forecast = 2 # months 


"""
1) Read the data from the database:
    key-value pair for non-relational database 
    "HosName_ItemName" : {'time' : listdata , 
                          "val" : listdata ,
                          "Purchase_freq" : int_val_representing months, ex: 1 => monthly , 
                                                                                12 => yearly , 
                                                                                0 => not-time series 
                          "m" (Seasonality) : int_value ,  THE NUMBER OF PERIODES IN EACH SEASON 
                                              For example, 
                                              m is 4 for quarterly data, 12 for monthly data, or 1 for annual 
                                              (non-seasonal) data.
                          }
    

2) Process the data
    
3) Get the predictions 
    
4) save the predictions to the database
    key-value pair for non-relational database 
    "HosName_ItemName" : {"Forecast" : is_forecast ,  'time' : time_f , "val"  : list(df['val'])  + list(forecast)  , "Confidance_Interval" : ct } 
"""

#-----------------------Read_data_from_the_database--------------

PD = purchase_data()

dd = PD.Get_data()

#print(dd)

def Get_Data(key): 
    
    data = pd.DataFrame()
    
    data['time'] = [ i.strftime('%Y-%d-%m') for i in dd[key]['time'] ]
    data['val'] = dd[key]['value']
    
    
    fqp = dd[key]['purchese_fre']
    m = dd[key]['seasomality'] # monthly
    
    return data , fqp , m 

# Using the mock data ---------------------------------------------------------------
    
# H = Hospital Name , I = item Name 

H = 1
I = 1
Id = str(H) + " " + str(I)

data , fqp , m = Get_Data(Id)

#print(data)
data['time'] = pd.to_datetime (data['time'])
data['time'] = data['time'].dt.date

seq = list(data['val'])
time= list(data['time'])

#print(list(data['val'])[0])


# --------------------------- Data-Preprocessing ----------------------





def process_seq(seq , split = 0.9 ):
        df = pd.DataFrame({'val' : seq})
        train=df[0:int(len(df)*split)] 
        test=df[int(len(df)*split):]
        return df , train , test

def add_months(sourcedate, months):
        month = sourcedate.month - 1 + months
        year = sourcedate.year + month // 12
        month = month % 12 + 1
        day = min(sourcedate.day, calendar.monthrange(year,month)[1])
        return dt.date(year, month, day)



def generate_data_for_regression(seq , time , len_to_forecast = 2   , split = 0.9):
      df= pd.DataFrame({"seq" : seq , "time" : time })
      X =df['time']
      y = np.asarray( df['seq'])
      #X = pd.to_datetime(X)
      X = X.map(dt.datetime.toordinal)
      
      X_future = []
      for i in range(len_to_forecast):
          X_future1 =  list(df['time'])[len(df['time'])-1] +  timedelta(days = 30 + 30*i )
          X_future.append(X_future1.toordinal())
          
      X = np.asarray(X).reshape(-1, 1)
      X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= split ,  random_state=0)
    
      #X_future = X_future.toordinal()
      X_future = np.asarray(X_future).reshape(-1, 1)
    
      return X , y , X_train, X_test, y_train, y_test,X_future



df , train , test = process_seq(seq , split = 0.9 )

X , Y , X_train, X_test, y_train, y_test,X_future = generate_data_for_regression(seq , time , len_to_forecast = len_to_forecast   , split = 0.9) 

 
time_f = time 
#print(time_f)

for i in range(len_to_forecast):
    sourcedate = time_f[-1]
    md = add_months(sourcedate,1 )
    time_f.append(md)
    

#--------------------------------Get-Pred--------------------------

def get_pred_ts(df , train , test , m ,len_to_forcast , use_AA_only = True) :
    if use_AA_only :
        MAPE ,future_forecast , conf_int = Auto_ARIMA(df ,train, test , fl = len_to_forcast, M= m, return_conf_int = True  ,  alpha=0.05   )
        if MAPE > 50 :
            print('error is considerably high try using other algorithms' )
        return future_forecast , conf_int , "Auto-ARIMA"
    else : 
        MAPE = []
        print(len_to_forcast)
        
        error , forcast = Holt_Winter(df ,train, test , fl = len_to_forcast , seasonal_periods = m  )
        MAPE.append((error , forcast , "Holt_Winter"))
        
        
        error , forcast = Holt_linear(df ,train, test , fl = len_to_forcast )
        MAPE.append((error , forcast , "Holt_Linear"))
        
        
        error , forcast = exp_smooth(df ,train, test , fl = len_to_forcast )
        MAPE.append((error , forcast , "exp_smooth("))
        
        
        error , forcast = mov_avg(df ,train, test , fl = len_to_forcast )
        MAPE.append((error , forcast , "mov_avg"))
        
        
        error , forcast = sim_avg(df ,train, test , fl = len_to_forcast )
        MAPE.append((error , forcast , "sim_avg"))
        
        
        error , forcast , ct  = Auto_ARIMA(df ,train, test , fl = len_to_forcast, M= m, return_conf_int =  True  )
        MAPE.append((error , forcast , "Auto_ARIMA")) 
        
        MAPE.sort()
        
        print(MAPE[0])
        
        return MAPE[0][1] , ['False']*len(MAPE[0][1])  ,  MAPE[0][2]
        
 
###########################################################################################################################
###########################################################################################################################
        
def get_best_re_model( X, Y , X_train, X_test, y_train, y_test,X_future):

          Future_P = []
          Error = [] 
        
          E,F = M_LinearRegression(X , Y , X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F = M_Ridge(X, Y , X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F = M_RidgeCV(X, Y, X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F = M_Lasso(X, Y , X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F = M_ElasticNet(X,Y,X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F = M_LassoLars(X,Y,X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F =M_BayesianRidge(X,Y,X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F = M_HuberRegressor(X,Y,X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F =  M_ElasticNetCV(X,Y,X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          E,F =  M_SGDRegressor(X,Y,X_train, X_test, y_train, y_test,X_future )
          Error.append(E)
          Future_P.append(F)
        
          return min(Error) , Future_P[Error.index(min(Error))]

if fqp != 0 :
    forecast  , ct , name = get_pred_ts(df , train , test , m ,len_to_forecast , use_AA_only = True) 
    is_forecast = ['False']len(df) + ['True'] len_to_forecast
    time_f = [str(obj) for obj in time_f]
    ct = [list(obj) for obj in ct] 
 
    ct = [[np.nan ,  np.nan ]]*len(df) +  list(ct)
    
    df_f = {"Forecast" : is_forecast ,  'time' : time_f , "val"  : list(df['val'])  + list(forecast)  , "Confidance_Interval" : ct } 
 
    r = PD.insert_data(Id,df_f)
 
    
    print(pd.DataFrame(df_f).tail())
    
else :
    
    err, forecast = get_best_re_model( X, Y , X_train, X_test, y_train, y_test,X_future)
    is_forecast = ['False']len(df) + ['True'] len_to_forecast
    ct = [[ np.nan ,  np.nan]]*len(time_f)
    time_f = [str(obj) for obj in time_f]
    ct = [list(obj) for obj in ct] 
 
 
    
    df_f = {"Forecast" : is_forecast ,  'time' : time_f , "val"  : list(df['val'])  + list(forecast)  , "Confidance_Interval" : ct } 
    
    r = PD.insert_data(Id,df_f)
    #print(r)
    
 
    


"""
Note :  The tuning of the models like Auto-Arima can be done when we get exact data.
        To make the system more automated the required tuning parameters like seasonality 
        can be stored with the data in the data set after doing some data analysis.
"""
