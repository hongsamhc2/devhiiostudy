import pandas as pd
import numpy as np
import RLEnvTrain,RLAgent
df = pd.read_csv('.\\DB\\CSV\\daily\\DA000020.csv')
df.sort_values(by='date',inplace=True)
env = RLEnvTrain.RLEnv(df)
agent = RLAgent.Agent()


for k in range(1000):
    obs = env.reset()

    for i in range(int(len(df)*0.5)):
        quant = np.random.randint(1, 10)
        action = agent.policy(obs)


        if not env.validation_(action, quant, obs):
            action = 0
        next_obs, reward, done, info = env.next_step(action, quant)
        agent.memorize_transition(obs,action,reward,next_obs,0.0 if done else 1.0)
        if agent.train:
            agent.experience_replay()

        if done:
            break

        obs = next_obs
    if k%10 == 0:
        print(env.reward)


    print('------------')