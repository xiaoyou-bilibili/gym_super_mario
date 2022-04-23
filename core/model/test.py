import numpy as np
import torch
from core.model.model import CNNDQN
from core.model.wrappers import wrap_environment
from os.path import join


def test(environment, action_space, iteration):
    flag = False
    env = wrap_environment(environment, action_space, monitor=True,
                           iteration=iteration)
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(join('pretrained_models',
                                        '%s.dat' % environment)))

    total_reward = 0.0
    state = env.reset()
    while True:
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            flag = True
        if done:
            print(total_reward)
            break

    env.close()
    return flag
