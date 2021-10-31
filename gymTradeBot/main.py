import time
import json
import pandas as pd
import numpy as np

from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from BinanceKey import *

from tradeEnv import TradeEnv

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from constants import *




def get_pairs():
    with open('params.json','r') as f:
        try:
            pcr = json.load(f)
            crypto = ''
            for pcri in pcr:
                crypto = pcri
            return crypto
        except:
            pass


def api_get_balance():
    baltemp = {}
    try:
        balancei = k.get_asset_balance(asset='EUR')
        baltemp['EUR'] = [balancei['free'],balancei['locked']]

        global pair
        pairi = pair[:-3]
        try:
            balancei = k.get_asset_balance(asset=pairi)
            baltemp[pair] = [balancei['free'],balancei['locked']]
        except BinanceAPIException as e:
            print('BinanceAPIException',e)
        except BinanceOrderException as e:
            print('BinanceOrderException',e)

        try:
            balancei = k.get_asset_balance(asset='BNB')
            baltemp['BNBEUR'] = [balancei['free'],balancei['locked']]
        except BinanceAPIException as e:
            print('BinanceAPIException',e)
        except BinanceOrderException as e:
            print('BinanceOrderException',e)

        return baltemp
    except BinanceAPIException as e:
        print('BinanceAPIException',e)
    except BinanceOrderException as e:
        print('BinanceOrderException',e)

def api_get_ticker_ohlc(asset,since,inttime):
    try:
        data = k.get_historical_klines(asset, str(inttime) + 'm', since, limit=1000)
        return data
    except:
        return [[0, '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', 31]]


def get_asset_params(name):
    params = {}
    global inttime, ema1, ema2, ema3, izeur
    try:
        params = json.load(f)

        with open('params1.json','w') as f:
            try:
                json.dump(params,f,indent=4)
            except:
                pass

        inttime = int(params[name]['inttime'])
        ema1 = int(params[name]['ema1'])
        ema2 = int(params[name]['ema2'])
        ema3 = int(params[name]['ema3'])
    
    except:
        pass





if __name__ == '__main__':

    liveMode = False

    nMode = 2
    # nMode = 1     Download Data, process and save in pickle
    # nMode = 2     Train model and save it
    # nMode = 3     Retrain model
    # nMode = 4     Load Model and test it

    inttime = 1
    ema1 = 3
    ema2 = 7
    ema3 = 14

    if (nMode==1):

        
        ndays = 365
        since = str(int(time.time()-(ndays*24*3600))*1000)
    
        k = Client(api_key,api_secret)

        pair = get_pairs()
        balance = api_get_balance()

        crypto_data = api_get_ticker_ohlc(pair,since,inttime)
        df = pd.DataFrame(crypto_data)
        df.columns = ['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore']
        df.drop(['Open time','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore'],axis=1,inplace=True)

        # Trading indicators
        # EMA
        get_asset_params(pair)
        df['EMA1'] = df['Close'].ewm(span=ema1,adjust=False).mean()
        df['EMA2'] = df['Close'].ewm(span=ema2,adjust=False).mean()
        df['EMA3'] = df['Close'].ewm(span=ema3,adjust=False).mean()

        # Store the open and close values in a pandas dataframe
        real_df = pd.DataFrame()
        real_df['Real open'] = df['Open'].astype(float)
        real_df['Real close'] = df['Close'].astype(float)

        # Normalize the data
        normalizer = MinMaxScaler().fit(df)
        df = normalizer.transform(df)
        df = pd.DataFrame(df,columns=['Open','High','Low','Close','Volume','EMA1','EMA2','EMA3'])

        # Concatenate normaliced and original Data
        df = pd.concat([df,real_df],axis=1)
        df = df[:-1]

        df.to_pickle('df.pkl')
        print(df)
        print("Data saved!")


    elif (nMode==2):
        df = pd.read_pickle('df.pkl')

        env = TradeEnv(df)
        model = PPO(MlpPolicy,env,gamma=1,learning_rate=0.5,verbose=0)

        model.learn(total_timesteps=CONST_SIM_STEPS,reset_num_timesteps=True)

        env.render(graph=True)

        model.save("tradeModel")

        obs = env.reset()
        env.render()
        for i in range(CONST_MAX_STEPS):
            action,_states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(print_step=True)


    elif (nMode==3):
        df = pd.read_pickle('df.pkl')
        print(df)

        env = TradeEnv(df)
        model = PPO(MlpPolicy,env,gamma=1,learning_rate=0.1,verbose=0)
        model.load("tradeModel")
        model.set_env(env)

        model.learn(total_timesteps=CONST_SIM_STEPS,reset_num_timesteps=True)

        env.render(graph=True)

        model.save("tradeModel")

        obs = env.reset()
        env.render()
        for i in range(CONST_MAX_STEPS):
            action,_states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(print_step=True)



    elif (nMode==4):
        df = pd.read_pickle('df.pkl')

        print(df)

        env = TradeEnv(df)
        model = PPO(MlpPolicy,env,gamma=1,learning_rate=0.1,verbose=0)
        model.load("tradeModel")

        obs = env.reset()
        env.render()
        for i in range(CONST_TEST_STEPS):
            action,_states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(print_step=True)



        # The algorithms require a vectorized environment to run
        #env = DummyVecEnv([lambda: TradeEnv(df)])

        #model = PPO(MlpPolicy, env, verbose=1)
        #model.learn(total_timesteps=20000)

        #obs = env.reset()
        #for i in range(20000):
        #    action, _states = model.predict(obs)
        #    obs, rewards, done, info = env.step(action)
        #    env.render()

