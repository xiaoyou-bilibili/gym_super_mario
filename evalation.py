import numpy as np
import torch
from core.model.wrappers import  wrap_environment
from core.model.model import CNNDQN
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,
                                          RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
model_path = ''

if __name__ == '__main__':
    flag = False
    # 初始化我们的环境。这里需要指定一下我们的关卡和策略
    env = wrap_environment("SuperMarioBros-1-1-v0", SIMPLE_MOVEMENT)
    # 初始化我们的网络信息,包括输入信息和我们可以执行的action数
    # (4, 84, 84) 7
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    # 手动加载我们的模型
    net.load_state_dict(torch.load("pretrained_models/SuperMarioBros-1-1-v0.dat"))
    # 初始化奖励信息以及重置我们的环境
    total_reward = 0.0
    state = env.reset()
    while True:
        # 这个state其实就是我们游戏模拟器当前的一个状态，也就是 (4, 84, 84)的数组
        state_v = torch.tensor(np.array([state], copy=False))
        # 我们这里使用我们的网络进行预测，获取到下一步的操作
        # 这里返回的是一个预测的列表，因为我们有7种操作，这里分别表示每种操作的可能性
        # 比如 [79.629974 81.05619  80.52793  83.71866  71.73175  79.65819  81.28025 ]
        q_vals = net(state_v).data.numpy()[0]
        # 然后我们从这个模型里面获取到最大的那个值就可以了，这里会返回0-6
        action = np.argmax(q_vals)
        # 然后我们给环境输入我们待执行的操作，这个环境就会返回4个参数
        # reward 表示这一步拿到的奖励
        # done表示游戏是否结束
        # info表示游戏的一些信息
        # 比如： {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 392, 'world': 1, 'x_pos': 397, 'x_pos_screen': 112, 'y_pos': 181}
        state, reward, done, info = env.step(action)
        # 渲染界面，主要是方便我们取观察状态，这个可以不加
        env.render()
        # 这里我们可以获取当前的总的奖励数，我们游戏的最终目的就是拿到的奖励最大，并且拿到旗子
        total_reward += reward
        # 如果夺旗了就表示我们成功完成了
        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            flag = True
        if done:
            # 游戏完成后打印一下总奖励数
            print(total_reward)
            break

    env.close()