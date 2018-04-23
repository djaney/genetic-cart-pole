import gym

import socket
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('', 8888))

env = gym.make('CartPole-v0')

done = False
generationSize = 10
iterations = 100

# learn
gen = 0
while True:

    max_score = 0
    for i in range(100):
        reward_sum = 0
        ob = env.reset()
        while True:
            ob_string = ','.join([str(o) for o in ob])
            s.send('act 0 {} {}'.format(i, ob_string).encode())
            res = s.recv(1024)
            res = res.decode('utf-8').split(',')
            res = [float(a) for a in res]
            action = np.argmax(res)
            ob, reward, done, info = env.step(action)
            reward_sum = reward_sum + reward
            if done:
                break
            max_score = np.max([max_score, reward_sum])

        s.send('rec 0 {} {}'.format(i, reward_sum).encode())
    print('max score {}'.format(max_score))
