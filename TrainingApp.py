import json
import time
import logging

import pymongo
import requests
from flask import Flask, jsonify, request, Response
from pip._internal.req import install_given_reqs

from BlackScholes import BlackScholes
import numpy as np
from datetime import datetime
import pandas as pd
import redis
import os
from pymongo import MongoClient
from datetime import datetime
from flask_cors import cross_origin

app = Flask("Training Application")
instrumentModelMap = {}
size = 8192
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# this function is to  retrive all the instruments form mongo and persist to Redis.

@app.route('/train/PersistTrainingSetForInstruments', methods=['POST'])
@cross_origin()
def PersistTrainingSetForInstruments():
        #connecting to Mongo db
        print('conecting to Mongo')
        myclient = connectToMongo()
        db = myclient["TradeData"]
        Collection = db["Options"]
        record = Collection.find()
        #print(record)
        try:
            redis_instrument_con = connectToRedis()
            keys= redis_instrument_con.keys('*')
            redis_instrument_con.delete(*keys)
        except Exception as e:
            print('Could not connect to redis', str(e))
            error_message = 'Could not connect to redis  ' + ' cause : ' + str(e)
        for item in record :
            instrumentId= item.get('contractSymbol')
            strikePrice = float(item.get('strikePrice'))
            expiry = item.get('expirationDate')
            date_object = datetime.strptime(expiry, '%Y-%m-%d')
            expiryInYears = date_object - datetime.now()
            expiryInDays = ((expiryInYears.days.__int__()) / 365).__round__(2)
            spotPrice = float(item.get('spotPrice'))
            volatality = float(item.get('volatility'))
            if(trainingSetExists(instrumentId,redis_instrument_con)):
                print('training set exists')
                redis_instrument_con.delete(instrumentId)
                logger.info("training data exists in Redis and deleted for ", instrumentId)
            else:
                logger.info("training data doesn't exists in Redis")
            xTrain,yTrain,dydxTrain = generateTrainingData(instrumentId,spotPrice,strikePrice, volatality, expiryInDays)
            model_training_data = np.concatenate((xTrain, yTrain, dydxTrain), axis=1)
            df = pd.DataFrame(model_training_data, columns=['spot', 'price', 'differential'])
            trainingSetDict= df.to_dict(orient="records")
            redis_instrument_con.set(instrumentId,json.dumps(trainingSetDict))
            print("record added",instrumentId)

# This function is to retrieve all the training set for given instruments.

@app.route('/train/GetTrainingSetForInstruments', methods=['POST'])
@cross_origin()
def GetTrainingSetForGivenInstruments():
    instrumentList = request.get_json()
    print('Request received',instrumentList)
    response = getRawResponse()
    seed = np.random.randint(0, 10000)
    try:
        redis_instrument_con = connectToRedis()
    except Exception as e:
        print('Could not connect to redis', str(e))
        error_message = 'Could not fetch instrument list from Redis ' + ' cause : ' + str(e)
    for item in instrumentList:
        instrumentId, strikePrice, expiryInYears, spotPrice, volatality = getRequestParam(item)
        # connecting to Redis cache
        print('conecting to Redis')
        try:
            if (trainingSetExists(instrumentId, redis_instrument_con)):
                print('training set exists')
                logger.info("training data exists in Redis")
                trainingSetDict = redis_instrument_con.get(instrumentId)
                sub_response = {}
                sub_response['instrumentId'] = instrumentId
                sub_response['training_data'] = json.loads(trainingSetDict)
                response["data"].append(sub_response)
                continue
            else:
                logger.info("training data doesn't exists in Redis")
                xTrain, yTrain, dydxTrain = generateTrainingData(instrumentId, spotPrice, strikePrice, volatality,
                                                           expiryInYears)
                model_training_data = np.concatenate((xTrain, yTrain, dydxTrain), axis=1)
                df = pd.DataFrame(model_training_data, columns=['spot', 'price', 'differential'])
                trainingSetDict = df.to_dict(orient="records")
                try:
                    redis_instrument_con.__setitem__(instrumentId, json.dumps(trainingSetDict))
                    logger.info("training key exist")
                except:
                    print("redis not available")
                    logger.info("training data exception ")
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
                redis_instrument_con.__setitem__(instrumentId, json.dumps(trainingSetDict))
            except:
                print("redis not available")
            sub_response = {}
            sub_response['instrumentId'] = instrumentId
            sub_response['training_data'] = trainingSetDict
            response["data"].append(sub_response)
            continue
    return jsonify(response)

# this function is to generate test data for given instruments and return test data

@app.route('/train/generateTestSet', methods=['POST'])
@cross_origin()
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

def connectToRedis(db=0):
    REDIS_HOST = os.environ.get('REDIS_HOST')
    REDIS_PORT = os.environ.get('REDIS_PORT')
    #redis_host = "a8216942522c.mylabserver.com" if not REDIS_HOST else REDIS_HOST
    #redis_port = 8095 if not REDIS_PORT else int(REDIS_PORT)
    redis_host = "13.68.144.231" if not REDIS_HOST else REDIS_HOST
    redis_port = 6379 if not REDIS_PORT else int(REDIS_PORT)
    print('Connecting to Redis host ={redis_host} , port ={redis_port}')
    rs = redis.StrictRedis(host=redis_host, port=redis_port, db=db, decode_responses=True)
    rs.ping()
    print("Connecting to Redis successfull")
    return rs

def connectToMongo(db=0):
    MONGO_URL = os.environ.get('MONGO_URL')
    mongoclient =  "mongodb://21af924e8e2c.mylabserver.com:8080/" if not MONGO_URL else MONGO_URL
    try:
        return pymongo.MongoClient(mongoclient)
    except pymongo.errors.ConnectionFailure:
        print("Failed to connect to server {}".format(mongoclient))

def generateTrainingData(instrumentId,spotPrice,strikePrice,volatality,expiryInYears):
    generator = BlackScholes()
    spotPrice = spotPrice / 100
    strikePrice = strikePrice / 100
    generator.__init__(spot=(spotPrice), K=(strikePrice), vol=volatality, T2=(1 + expiryInYears))
    xTrain, yTrain, dydxTrain = generator.trainingSet(size)
    return xTrain, yTrain, dydxTrain


def generateTestData(instrumentId,spotPrice,strikePrice,volatality,expiryInYears):
    generator = BlackScholes()
    spotPrice = spotPrice/100
    strikePrice = strikePrice/100
    generator.__init__(spot=(spotPrice), K=(strikePrice), vol=volatality, T2=(1 + expiryInYears))
    lowerBound = ( spotPrice - (spotPrice * 0.65) )
    upperBound = ( spotPrice + (spotPrice * 0.65) )
    xTrain, yTrain, dydxTrain = generator.testSet(lowerBound,upperBound)
    return xTrain, yTrain, dydxTrain

def getRawResponse():
    return {"data":[]}

def getRequestParam(request_data):
    instrumentId = request_data['ticker']
    strikePrice = float(request_data['strikeprice'])
    expiry = request_data['expiry']
    date_object = datetime.strptime(expiry, '%Y-%m-%d')
    expiryInYears = date_object - datetime.now()
    expiryInDays =  ((expiryInYears.days.__int__())/365).__round__(2)
    #expiryInYears = float((date_object.month - datetime.now().month) / 12).__round__(1)
    spotPrice = float(request_data['spotprice'])
    volatality = float(request_data['volatility'])
    return instrumentId , strikePrice, expiryInDays, spotPrice, volatality

def populateModelCache(instrumentId, model):
    if not instrumentModelMap.get(instrumentId):
        print("Adding training set to redis  cache for ", instrumentId)
        instrumentModelMap.__setitem__(instrumentId, model)
    else:
        print("Model Already Exist in cache")

def trainingSetExists(instrumentId, r):
    return True if r.get(instrumentId) else False


def main():
    response_API = requests.post('http://127.0.0.1:5000/train/PersistTrainingSetForInstruments')
    print(response_API.status_code)
if __name__ == '__main__':
  #  main()
  app.run(debug=True,host='0.0.0.0', port=5001)
