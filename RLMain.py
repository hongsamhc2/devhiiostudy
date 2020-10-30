import RLAgent,RLTrainEnv
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
def train_model(env,epochs=1,episode=10):
    start = time.time()
    for _ in range(epochs):
        env.reset()
        now_obs = env.observe() # 초기화 후 관측치
        print('reset_state',now_obs)
        for step in range(1,episode+1):
            action = agent.eps_policy(now_obs) #행동
            done, reward = agent.do_action(action,now_obs) # 액션 실행
            next_obs = env.observe()
            agent.memorize(now_obs,action,reward,next_obs,0.0 if done==0 else 1.0)
            if agent.train:
                agent.experience_replay()
            if done == 0:
                break
            now_obs = next_obs

    end=time.time()
    print(end-start)
    return
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

df = pd.read_csv('.\\DB\\CSV\\daily\\DA000020.csv')
df.sort_values(by='date',inplace=True)

env = RLTrainEnv.RLtradingTestEnv(df)
agent = RLAgent.RLAgent(env)
agent.set_cash(1000000 * len(df))

train_model(env,episode=4096)





