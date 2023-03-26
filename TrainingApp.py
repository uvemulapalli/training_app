import json
import time
import logging
import requests
from flask import Flask, jsonify, request , Response
from BlackScholes import BlackScholes
import numpy as np
from datetime import datetime
import pandas as pd
import redis
from pymongo import MongoClient
from datetime import datetime

app = Flask("Training Application")
instrumentModelMap = {}
size = 512
redis_host = "a8216942522c.mylabserver.com"
redis_port = 8095
myclient = MongoClient("mongodb://21af924e8e2c.mylabserver.com:8080/?connect=false")

@app.route('/train/PersistTrainingSetForInstruments', methods=['POST'])
def PersistTrainingSetForInstruments():
        #connecting to Mongo db
        print('conecting to Mongo')
        db = myclient["TradeData"]
        Collection = db["Options"]
        record = Collection.find()
        print(record)
        r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        for item in record :
            instrumentId= item.get('contractSymbol')
            strikePrice = float(item.get('strikePrice'))
            expiry = item.get('expirationDate')
            date_object = datetime.strptime(expiry, '%Y-%m-%d').date()
            expiryInYears = float((date_object.month-datetime.now().month)/12).__round__(1)
            spotPrice = float(item.get('spotPrice')).__round__(3)
            volatality = float(item.get('volatility')).__round__(3)
            if(trainingSetExists(instrumentId,r)):
                print('training set exists')
            else:
                xTrain,yTrain,dydxTrain = generateTrainingData(instrumentId,spotPrice,strikePrice, volatality, expiryInYears)
                model_training_data = np.concatenate((xTrain, yTrain, dydxTrain), axis=1)
                df = pd.DataFrame(model_training_data, columns=['spot', 'price', 'differential'])
                trainingSetDict= df.to_dict(orient="records")
                trainingDB = db["TrainingDB"]
                records = {"instrumentId": instrumentId,
                       "data": trainingSetDict}
                trainingDB.insert_one(records)
                r.set(instrumentId,json.dumps(trainingSetDict))
                print("record added",records,r)

@app.route('/train/GetTrainingSetForInstruments', methods=['POST'])
def GetTrainingSetForGivenInstruments():
    instrumentList = request.get_json()
    print('Request received',instrumentList)
    response = getRawResponse()
    seed = np.random.randint(0, 10000)
    for item in instrumentList:
        instrumentId, strikePrice, expiryInYears, spotPrice, volatality = getRequestParam(item)
        # connecting to Redis cache
        print('conecting to Redis')
        r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
       # r.delete(instrumentId)
        try:
            if (trainingSetExists(instrumentId, r)):
                print('training set exists')
                trainingSetDict = r.get(instrumentId)
                sub_response = {}
                sub_response['instrumentId'] = instrumentId
                sub_response['training_data'] = json.loads(trainingSetDict)
                response["data"].append(sub_response)
                continue
            else:
                xTrain, yTrain, dydxTrain = generateTrainingData(instrumentId, spotPrice, strikePrice, volatality,
                                                           expiryInYears)
                model_training_data = np.concatenate((xTrain, yTrain, dydxTrain), axis=1)
                df = pd.DataFrame(model_training_data, columns=['spot', 'price', 'differential'])
                trainingSetDict = df.to_dict(orient="records")
                try:
                    r.__setitem__(instrumentId, json.dumps(trainingSetDict))
                except:
                    print("redis not available")
                sub_response = {}
                sub_response['instrumentId'] = instrumentId
                sub_response['training_data'] = trainingSetDict
                response["data"].append(sub_response)
                continue
        except:
            xTrain, yTrain, dydxTrain = generateTrainingData(instrumentId, spotPrice, strikePrice, volatality,
                                                       expiryInYears)
            model_training_data = np.concatenate((xTrain, yTrain, dydxTrain), axis=1)
            df = pd.DataFrame(model_training_data, columns=['spot', 'price', 'differential'])
            trainingSetDict = df.to_dict(orient="records")
            try:
                r.__setitem__(instrumentId, json.dumps(trainingSetDict))
            except:
                print("redis not available")
            sub_response = {}
            sub_response['instrumentId'] = instrumentId
            sub_response['training_data'] = trainingSetDict
            response["data"].append(sub_response)
            continue
    return jsonify(response)

def generateTrainingData(instrumentId,spotPrice,strikePrice,volatality,expiryInYears):
    generator = BlackScholes()
    generator.__init__(spot=(spotPrice), K=(strikePrice), vol=volatality, T2=(1 + expiryInYears))
    xTrain, yTrain, dydxTrain = generator.trainingSet(size)
    return xTrain, yTrain, dydxTrain
def generateTestData(instrumentId,spotPrice,strikePrice,volatality,expiryInYears):
    generator = BlackScholes()
    generator.__init__(spot=(spotPrice), K=(strikePrice), vol=volatality, T2=(1 + expiryInYears))
    lowerBound = ( spotPrice - (spotPrice * 0.05) )
    upperBound = ( spotPrice + (spotPrice * 0.05) )
    xTrain, yTrain, dydxTrain = generator.testSet(lowerBound,upperBound)
    return xTrain, yTrain, dydxTrain

def getRawResponse():
    return {"data":[]}

def getRequestParam(request_data):
    instrumentId = request_data['ticker']
    strikePrice = float(request_data['strikeprice'])
    expiry = request_data['expiry']
    date_object = datetime.strptime(expiry, '%Y-%m-%d').date()
    expiryInYears = float((date_object.month - datetime.now().month) / 12).__round__(1)
    spotPrice = float(request_data['spotprice'])
    volatality = float(request_data['volatility'])

    return instrumentId , strikePrice, expiryInYears, spotPrice, volatality

def populateModelCache(instrumentId, model):
    if not instrumentModelMap.get(instrumentId):
        print("Adding training set to redis  cache for ", instrumentId)
        instrumentModelMap.__setitem__(instrumentId, model)
    else:
        print("Model Already Exist in cache")

def trainingSetExists(instrumentId, r):
    return True if r.get(instrumentId) else False

@app.route('/train/generateTestSet', methods=['POST'])

def generateTestSet():
    instrumentList = request.get_json()
    response = getRawResponse()
    seed = np.random.randint(0, 10000)
    for item in instrumentList:
        instrumentId, strikePrice, expiryInYears, spotPrice, volatality = getRequestParam(item)
        # connecting to Redis cache
        xTrain, yTrain, dydxTrain = generateTestData(instrumentId, spotPrice, strikePrice, volatality,
                                                             expiryInYears)
        model_training_data = np.concatenate((xTrain, yTrain, dydxTrain), axis=1)
        df = pd.DataFrame(model_training_data, columns=['spot', 'spot', 'simulatedPrice'])
        trainingSetDict = df.to_dict(orient="records")
        sub_response = {}
        sub_response['instrumentId'] = instrumentId
        sub_response['test_data'] = trainingSetDict
        response["data"].append(sub_response)
    return jsonify(response)

def main():
    response_API = requests.post('http://127.0.0.1:5000/train/PersistTrainingSetForInstruments')
    print(response_API.status_code)
if __name__ == '__main__':
  #  main()
  app.run(host='0.0.0.0', port=5000)
