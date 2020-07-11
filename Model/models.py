
# -------------------- IMPORTS ---------------------

import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from pyramid import auto_arima
from errors import mean_absolute_percentage_error
 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model
 

# --------------------- MODELS ---------------------


def sim_avg(df ,train, test , fl = 10 ):
    
        #---------------Algorithm ------------------------
    
        y_hat_avg = test.copy()
        y_hat_avg['avg_forecast'] = train['val'].mean()
        
        # -------------Error-----------------------------
        
        error =   mean_absolute_percentage_error (test, y_hat_avg['avg_forecast'])
 
        #--------------Forcast--------------------------
        
        forcast = df['val'].mean()
          
        return error , [forcast]*fl  # error is on test data and forcastis done using the full data 
    
def mov_avg(df ,train, test , fl = 10 ):
    
        #---------------Algorithm ------------------------
    
        y_hat_avg = test.copy()
        y_hat_avg['moving_avg_forecast'] = train['val'].rolling(20).mean().iloc[-1]

        # -------------Error-----------------------------
        
        error =   mean_absolute_percentage_error (test, y_hat_avg['moving_avg_forecast'])

        #--------------Forcast--------------------------
        
        forcast = df['val'].rolling(20).mean().iloc[-1]
          
        return error ,  [forcast]*fl   # error is on test data and forcastis done using the full data 
    
def exp_smooth(df ,train, test , fl = 10 ):
    
        #---------------Algorithm ------------------------
    
        y_hat_avg = test.copy()
        fit2 = SimpleExpSmoothing(np.asarray(train['val'])).fit(smoothing_level=0.6,optimized=False)
        y_hat_avg['SES'] = fit2.forecast(len(test))

        # -------------Error-----------------------------
        
        error =   mean_absolute_percentage_error (test, y_hat_avg['SES'])
    
        
        #--------------Forcast--------------------------
        
        fit2 = SimpleExpSmoothing(np.asarray(df['val'])).fit(smoothing_level=0.6,optimized=False)
        forcast = fit2.forecast(fl)
          
        return error , forcast  # error is on test data and forcastis done using the full data 
    
def Holt_linear(df ,train, test , fl = 10 ): # Holt’s Linear Trend method 
    
        #---------------Algorithm ------------------------
    
        y_hat_avg = test.copy()
        fit1 = Holt(np.asarray(train['val'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
        y_hat_avg['Holt_linear'] = fit1.forecast(len(test))

        # -------------Error-----------------------------
        
        error =   mean_absolute_percentage_error (test, y_hat_avg['Holt_linear'])
        
        #--------------Forcast--------------------------
        
        fit1 = Holt(np.asarray(df['val'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
        forcast = fit1.forecast(fl)
          
        return error , forcast  # error is on test data and forcastis done using the full data 
    
def Holt_Winter(df ,train, test , fl = 10   , seasonal_periods = 12 ):
    
        #---------------Algorithm ------------------------
    
        y_hat_avg = test.copy()
        fit1 = ExponentialSmoothing(np.asarray(train['val']) ,seasonal_periods= seasonal_periods  ,trend='add', seasonal='add',).fit()
        y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
        

            
        # -------------Error-----------------------------
        
        error =   mean_absolute_percentage_error (test, y_hat_avg['Holt_Winter'])
        
        #--------------Forcast--------------------------
        
        fit1 = ExponentialSmoothing(np.asarray(df['val']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
        forcast = fit1.forecast(fl)
          
        return error , forcast  # error is on test data and forcastis done using the full data 
    
    
def Auto_ARIMA(df ,train, test , fl = 10 , M=12  , alpha=0.05  , return_conf_int = True ): # assuming the data is monthly 
    
        #---------------Algorithm ------------------------

        stepwise_model = auto_arima(df, start_p=1, start_q=1,
                                   max_p=3, max_q=3,m= M ,
                                   start_P=0, seasonal=True,
                                   d=1, D=1, trace=True,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True)
        stepwise_model.fit(train)
        future_forecast   = stepwise_model.predict(n_periods=len(test))
        forcast = test.copy()
        forcast['forcast'] = future_forecast
 
        # -------------Error-----------------------------
        
        error =   mean_absolute_percentage_error (test, forcast['forcast'] )
        
        #--------------Forcast--------------------------
        
        stepwise_model.fit(df)
        future_forecast , conf_int = stepwise_model.predict(n_periods= fl  , return_conf_int= return_conf_int ,alpha=alpha)

          
        return error ,list(future_forecast) , conf_int # error is on test data and forcastis done using the full data 
    
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
        
# M_LinearRegression , M_Ridge , M_RidgeCV ,  M_Lasso , M_ElasticNet , M_LassoLars ,  M_BayesianRidge , M_HuberRegressor , M_ElasticNetCV ,  M_SGDRegressor
    
def M_LinearRegression(X,Y,X_train, X_test, y_train, y_test,X_future ): 
    
      #---------------Algorithm ------------------------
    
      model = LinearRegression()
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_Ridge(X , Y , X_train, X_test, y_train, y_test,X_future ):  
    
    
      #---------------Algorithm ------------------------
    
      model =linear_model.Ridge(alpha=.5)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_RidgeCV(X , Y , X_train, X_test, y_train, y_test,X_future ):  
   
      #---------------Algorithm ------------------------
    
      model = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_Lasso(X , Y, X_train, X_test, y_train, y_test,X_future ):  
    
      #---------------Algorithm ------------------------
    
      model = linear_model.Lasso(alpha=0.1)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_ElasticNet(X , Y , X_train, X_test, y_train, y_test,X_future ):  

      #---------------Algorithm ------------------------
    
      model = linear_model.ElasticNet(alpha=0.1)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_LassoLars(X , Y , X_train, X_test, y_train, y_test,X_future ):  
    
      #---------------Algorithm ------------------------
    
      model =linear_model.LassoLars(alpha=0.1)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_BayesianRidge(X , Y , X_train, X_test, y_train, y_test,X_future ):  

      #---------------Algorithm ------------------------
    
      model =linear_model.BayesianRidge()
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_HuberRegressor(X,Y,X_train, X_test, y_train, y_test,X_future ):  

      #---------------Algorithm ------------------------
    
      model =linear_model.HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001,tol=1e-05)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P


def M_ElasticNetCV(X,Y,X_train, X_test, y_train, y_test,X_future ):  

      #---------------Algorithm ------------------------
    
      model =linear_model.ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, max_iter=1000, tol=0.0001)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P

  
def M_SGDRegressor(X , Y,X_train, X_test, y_train, y_test,X_future ):  
    
      #---------------Algorithm ------------------------
    
      model =   linear_model.SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      
      
      # -------------Error-----------------------------
      
      
      Error = metrics.mean_squared_error(y_test,predictions)
      
      
      #--------------Forcast--------------------------
      
      
      model = LinearRegression()
      model.fit(X,Y)
      Future_P = model.predict(X_future)
      return Error , Future_P
  
  




    