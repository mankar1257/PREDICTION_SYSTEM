import pandas as pd  
import numpy as np  
from statistics import mean
from pymongo import MongoClient
import os


class purchase_data:
    """docstring for  Hospital_data."""

    def __init__(self):
        super(purchase_data, self).__init__()
    
    def get_data(self):
        df = pd.read_csv('/data/train.csv')
        DF = pd.DataFrame(columns=['month','Hospital','items','Demand'])
        df_s = df.groupby('store')
        for i in range(1,11):
      
      
          df_1 = df_s.get_group(i)
          month = [int(h.split('-')[1]+h.split('-')[0]) for  h in df_1['date']]
      
          df_1['month'] = month
          df_1_month = df_1.groupby('month')
      
      
          for j in df_1['month'].unique():
            df_1_month_item = df_1_month.get_group(j).groupby('item')
            #print(df_1_month_item.get_group(2)['item'])
            for k in range(1,51):
              dd = df_1_month_item.get_group(k)
              ll = [dd.iloc[1,4],dd.iloc[1,1] , dd.iloc[1,2] , round(mean(dd['sales']) ,2)]
              DF.loc[len(DF)] = ll
              
        return DF
    
    def get_item_for_H(self,DF,H = 1 , I = 1):
        DF_I = DF.groupby('items')
        DF_I_1 = DF_I.get_group(I)
        DF_I_H_1 = DF_I_1.groupby('Hospital')
        DF_I_1_H_1 = DF_I_H_1.get_group(H) 
        return DF_I_1_H_1
    
    
    def get_date(self,x):
        l = str(x).split('.')[0]
        y = l[-4:]
        m = l[:-4]
        d = 1
        return ' ' + str(d) + ' ' + str(m) + ' ' + str(y) + ' ' 
    
    def generate_data(self,DF_I_1_H_1):
        
        DF_I_1_H_1['month'] = DF_I_1_H_1['month'].apply(lambda x : self.get_date(x))
        DF_I_1_H_1['month']=pd.to_datetime(DF_I_1_H_1['month'])
        
        
        time = list(DF_I_1_H_1['month'])
        value = list(DF_I_1_H_1['Demand'])
      
        purchese_fre = 1
      
        seasomality = 12
      
        dict = {'time': time , 'value':value , 'purchese_fre':purchese_fre , 'seasomality':seasomality}
      
        return dict
    
    
    def Update_data(self):
    
        DF = self.get_data()
        
        F_dict = {}
        for H in range(1,11):
          for I in range(1,51):
            df = self.get_item_for_H(DF,H , I )
            F_dict[str(H) + " " + str(I) ] = self.generate_data(df)
          
        try :
            # set connection to the cloud database
             
            MONGO_URL = os.getenv("DB")
            conn = MongoClient(MONGO_URL)
            # set path
            db = conn['Hospital_data']

            # Update document

            #records = json.loads(data.T.to_json()).values()
            db.History.delete_many({})
            #records = json.dumps(records)  
            db.History.insert(F_dict)

            return True
        
        except:
            
            return False
    
    def Get_data(self):
        
        try :
            # set connection to the cloud database
             
            MONGO_URL = os.getenv("DB")
            conn = MongoClient(MONGO_URL)
            # set path
            db = conn['Hospital_data']
             
            d = db.History.find_one()

            return d
        
        except:
            
            return False
        
    def insert_data(self,Id,data):
        
        try :
            # set connection to the cloud database
             
            MONGO_URL = os.getenv("DB")
            conn = MongoClient(MONGO_URL)
            
            # set path
            db = conn['Hospital_data']
            
            data['_id'] = Id
                
            d = db.Prediction_data.insert(data)

            return True
        
        except:
            
            return False
    
        
        
        
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

