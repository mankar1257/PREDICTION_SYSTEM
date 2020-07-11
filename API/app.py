#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:44:30 2020

@author: vaibhav
"""


from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from pymongo import MongoClient
import os


app = Flask(__name__)
api = Api(app)

 
# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('H', type=int)
parser.add_argument('I', type=int)
 
 
class Demand(Resource):

    def get(self,):

        args = parser.parse_args()
        H =  args['H'] 
        I =  args['I']
        
        try :
            # set connection to the cloud database
             
            MONGO_URL = os.getenv("DB")
            conn = MongoClient(MONGO_URL)
            # set path
            db = conn['Hospital_data']
            
        except:
            
            return "Problem in DB connectivity"
        
        if( (H == None and I == None) or H == None ):
            dd = db.Prediction_data.find()
            D = []
            for d in dd:
                D.append(d)
            return D
              
        elif (I == None):
            H_D = [] 
            for i in range(1,51):
                ID = str(H) + ' ' +str(i)
                try:
                    d = db.Prediction_data.find_one({"_id": ID})
                    H_D.append(d)
                except:
                    continue
            return H_D
        
        else:
            ID = str(H) + ' ' +str(I)
            d = db.Prediction_data.find_one({"_id": ID})
            return d
         
            
         

api.add_resource(Demand, '/find')




if __name__ == '__main__':
    app.run(debug=True)
