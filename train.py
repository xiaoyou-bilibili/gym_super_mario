from core.model.replay_buffer import PrioritizedBuffer
from core.model.train_information import TrainInformation
from core.model.wrappers import wrap_environment
from os.path import join
from shutil import copyfile, move
from core.model.test import test
from core.model.helpers import (compute_td_loss,
                                initialize_models,
                                set_device,
                                update_beta,
                                update_epsilon)
from torch import save
from torch.optim import Adam
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,
                                          RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)


# 更新我们的模型
def update_graph(model, target_model, optimizer, replay_buffer, device,
                 info, beta):
    if len(replay_buffer) > initial_learning:
        if not info.index % target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        # 计算Q信息
        compute_td_loss(model, target_model, replay_buffer, gamma, device,
                        batch_size, beta)
        optimizer.step()


def test_new_model(model, info):
    # 保存一下我们的模型
    save(model.state_dict(), join('pretrained_models', '%s.dat' % environment))
    print('Testing model...')
    # 测试我们的模型
    flag = test(environment, action_space, info.new_best_counter)
    # 如果模型通过测试我们就拷贝一下模型
    if flag:
        copyfile(join('pretrained_models', '%s.dat' % environment),
                 'recording/run%s/%s.dat' % (info.new_best_counter,
                                             environment))


# 完成本轮训练
def complete_episode(model, info, episode_reward, episode,
                     epsilon, stats):
    # 首先计算一下本轮训练的奖励得分
    new_best = info.update_rewards(episode_reward)
    # 如果是新的记录，我们就测试测试一下我们的模型
    if new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
        # 这里我们传入模型
        test_new_model(model, info)
    elif stats['flag_get']:
        # 如果我们本次训练拿到了flag就记录一下最好训练的次数
        info.update_best_counter()
        test_new_model(model, info)
    print('Episode %s - Reward: %s, Best: %s, Average: %s '
          'Epsilon: %s' % (episode,
                           round(episode_reward, 3),
                           round(info.best_reward, 3),
                           round(info.average, 3),
                           round(epsilon, 4)))


# 开始一轮训练
def run_episode(env, model, target_model, optimizer, replay_buffer,
                device, info, episode):
    # 设置当前训练的总得分，然后重置一下环境
    episode_reward = 0.0
    state = env.reset()

    while True:
        # 计算一下
        epsilon = update_epsilon(info.index)
        if len(replay_buffer) > batch_size:
            beta = update_beta(info.index)
        else:
            beta = 0.4
        # 调用我们的模型获取下一步的操作
        action = model.act(state, epsilon, device)
        # 是否显示游戏界面
        if render:
            env.render()
        # 我们把我们模型的预测结果传入到模拟器中
        next_state, reward, done, stats = env.step(action)
        # 把我们探索到的数据放到经验回放区中
        replay_buffer.push(state, action, reward, next_state, done)
        # 获取下一步的状态并且更新得分
        state = next_state
        episode_reward += reward
        # 更新一下当前的索引，其实就是+1操作
        info.update_index()
        # 更新一下我们模型的权重信息
        update_graph(model, target_model, optimizer, replay_buffer,
                     device, info, beta)
        # 如果游戏结束我们就完成本轮的训练
        if done:
            complete_episode(model, info, episode_reward,
                             episode, epsilon, stats)
            break


# 模型训练
def train(env, model, target_model, optimizer, replay_buffer, device):
    # 当前训练的信息
    info = TrainInformation()

    for episode in range(num_episodes):
        # 开始每一轮的训练
        run_episode(env, model, target_model, optimizer, replay_buffer,
                    device, info, episode)


def main():
    # 初始化模型
    env = wrap_environment(environment, action_space)
    # 设置我们的设备 GPU or CPU
    device = set_device(force_cpu)
    # 初始化我们的模型，这里就包括了两个模型，让我拟合更加轻松
    model, target_model = initialize_models(environment, env, device,
                                            transfer)
    # 设置我们的优化器。用于保存状态和更新餐护士
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # 设置一下缓冲区
    replay_buffer = PrioritizedBuffer(buffer_capacity)
    # 开始训练我们的模型
    train(env, model, target_model, optimizer, replay_buffer, device)
    # 训练完毕后关闭环境
    env.close()


# -- 训练的参数配置
# 关卡配置
environment = "SuperMarioBros-1-1-v0"
# 操作模式
action_space = SIMPLE_MOVEMENT
# 是否使用强制使用CPU来计算
force_cpu = False
# 是否使用预训练模型的权重信息（可以加快训练的书序）
transfer = True
# 设置学习率信息
learning_rate = 1e-4
# 经验回放的大小
buffer_capacity = 20000
# 训练轮次
num_episodes = 50000
# 一次取多少条数据
batch_size = 32
# 是否显示游戏界面
render = True
# 经过多少次尝试才开始正式更新模型
initial_learning = 10000
# 模型的更新频率
target_update_frequency = 1000
# 对未来reward的衰减值
gamma = 0.99

if __name__ == '__main__':
    main()
