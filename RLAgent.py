import numpy as np
import pandas as pd
from tensorflow import keras
from collections import deque
from random import sample
import tensorflow as tf
class RLAgent:
    def __init__(self, env,eps_start=1,
                 eps_end=0.01,
                 eps_decay_steps=250,
                 eps_exponential_decay=0.99,
                 batch_size = 4096,
                 replay_capacity=int(1e6),
                 tau = 100):
        '''
        추가적인 변수 선언
        1. 거래세
        2. 거래 수수료
        '''

        # Agent 활동 환경
        self.env = env

        # Agent 행동
        self.Action_Buy = 1
        self.Action_Sell = 2
        self.Action_Hold = 0
        self.Actions = [self.Action_Buy,self.Action_Hold]

        # 현금
        self.cash = 0

        # 주식 보유량 / 보유 주식 가격
        self.stock_holdings = 0
        self.stock_price = 0
        self.holdings_stock_list = []
        self.stock_holdings_amount = 0

        #model
        self.Q_model_train = self.build_model()
        self.Q_model_actual = self.build_model(trainable=False)
        self.update_Q_model()
        self.train = True

        #epsilon
        self.eps = eps_start
        self.eps_decay_steps = eps_decay_steps
        self.eps_decay = (eps_start-eps_end)/eps_decay_steps
        self.eps_history =[]
        self.eps_exponential_decay =eps_exponential_decay

        #episode
        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes =0
        self.episode_reward = 0
        self.rewards_history = []
        self.step_per_episode =[]

        #활용 변수
        self.total_step = 0
        self.profit_list = []
        self.batch_size = batch_size
        self.experience = deque([],maxlen=replay_capacity)
        self.idx = tf.range(batch_size)
        self.losses =[]
        self.tau = tau

    def build_model(self,trainable=True):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=[1]))
        model.add(keras.layers.Dense(20,activation='relu',trainable=trainable))
        model.add(keras.layers.Dense(1,name='Output'))
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model

    def update_Q_model(self):
        return self.Q_model_actual.set_weights(self.Q_model_train.get_weights())


    def eps_policy(self,obs):
        self.total_step += 1
        random_index = np.random.rand()
        if random_index <= self.eps:
            act = np.random.choice(self.Actions)
            return act
        close = np.array(obs[1]).reshape(-1,1)
        Q = self.Q_model_train.predict(close)


        act = np.argmax(Q)
        print(Q,act)
        return act



    def set_cash(self, cash):
        self.cash = cash

    def get_cash(self):
        return self.cash

    def set_stock_state(self, date, stock_holdings, strock_price):
        self.stock_holdings = stock_holdings
        self.stock_price = strock_price
        self.stock_list = [date, self.stock_holdings, self.stock_price]
        self.holdings_stock_list.append(self.stock_list)

    def get_stock_state(self):
        return self.holdings_stock_list

    def val_action(self, obs):
        cash = self.get_cash()
        max_buy_quatity = 1
        # if obs <= cash:
        #     max_buy_quatity = cash//obs
        return max_buy_quatity

    def do_action(self, act,obs):
        #수익률 계산
        reward = 0

        #실행 상태
        done = 1
        if act == 0:

            reward = self.profit(obs)

            print('관측',reward)
        elif act == 1:
            val_act =self.val_action(obs[1])

            if val_act == 0:
                act_list = [0,0]
                self.holdings_stock_list.append(act_list)
            else:
                buy_quantity = np.random.randint(1,val_act+1)
                self.stock_holdings_amount += buy_quantity
                self.cash -= buy_quantity * obs[1]
                self.holdings_stock_list.append([buy_quantity,obs[1]])
            reward = self.profit(obs)
            print('매수', reward)
        elif act == 2:
            print('매도')
        if self.cash < obs[1]:
            done = 0
        return done , reward



    def profit(self,obs):
        buy_amount = 0
        rate = 0
        profit = 0
        if self.holdings_stock_list:
            for stock_list in self.holdings_stock_list:
                buy_amount += stock_list[0] * stock_list[1]
        cu_rate = self.stock_holdings_amount * obs[1]
        if buy_amount != 0:
            rate = (cu_rate - buy_amount) / buy_amount * 100
        if len(self.profit_list) != 0:
            profit = rate - self.profit_list[-1]
        self.profit_list.append(rate)
        return profit

    def memorize(self,obs,act,reward,next_obs,not_done):
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.eps_decay_steps:
                    self.eps -= self.eps_decay
                else:
                    self.eps += self.eps_exponential_decay
            self.episodes +=1
            self.rewards_history.append(self.episode_reward)
            self.step_per_episode.append(self.episode_length)
            self.episode_reward,self.episode_length = 0,0
        self.experience.append((obs[1],act,reward,next_obs[1],not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            print(len(self.experience))
            return
        minibatch = map(np.array,zip(*sample(self.experience,self.batch_size)))
        obs,act,reward,next_obs,not_done=minibatch
        next_obs = np.array(next_obs).reshape(-1,1)


        next_Q_val = self.Q_model_train.predict_on_batch(next_obs)
        best_actions = tf.argmax(next_Q_val,axis=1)
        next_Q_val_target = self.Q_model_actual.predict_on_batch(next_obs)

        target_Q_val = tf.gather_nd(next_Q_val_target,
                                    tf.stack((self.idx,tf.cast(best_actions,tf.int32)),axis=1))

        targets = reward + not_done*0.99 * target_Q_val # gamma 0.99
        close = np.array(obs).reshape(-1, 1)
        Q_val = self.Q_model_train.predict_on_batch(close)
        print(Q_val)
        #loss = self.Q_model_train.train_on_batch(close,df_)
        #self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_Q_model()


