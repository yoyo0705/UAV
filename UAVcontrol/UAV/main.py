import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import optim
import matplotlib.pyplot as plt
import gym
import math

# 参数设置
EPSILON = 0.9
GAMMA = 0.01  # 学习率
LR = 0.01
MEMORY_CAPACITY = 2000  # 存储器容量
BATCH_SIZE = 64  # 神经网络学习时从记忆存储单元中抽取的经验个数
Q_NETWORK_ITERATION = 200  # 评估网络每学习200次就将参数传给目标网络
EPISODES = 400


env = gym.make('UAVEnv-v0', uav_num=2, map_w=3000, map_h=3000, map_z=5000,
               Init_state=[[240.0, 3000.0, 3000.0, 2800.0, 0.0, -3 * math.pi / 4],
                           [250.0, 0.0, 0.0, 0.0, 0.0, math.pi / 4]])
env = env.unwrapped
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 7)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Dqn:
    def __init__(self):  # 构造函数
        self.eval_net, self.target_net = Net(), Net()  # 评估网络 目标网络
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 3 + 1))  # 经验池设置
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0  # 用于跟踪经验池中的数据数量和学习次数
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)  # adam优化器，用于更新神经网络的权重
        self.loss = nn.MSELoss()  # 均方误差损失函数

        self.fig, self.ax = plt.subplots()

    def store_trans(self, state, action, reward, next_state):  # 将状态转换存储到经验池中
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, [action], [reward], next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):  # 根据当前状态选择一个动作(从动作控制器中选择）
        # notation that the function return the action's index nor the real action
        # EPSILON
        state = np.array(state).reshape(1, 12)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 转化为pytorch张量
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)  # 计算7种动作分别对应的Q值
            # action = torch.max(action_value, 1)[1].data # get action whose q is max    # 选出最大Q值对应的动作
            action = torch.argmax(action_value)
        else:
            action = np.random.randint(0, 7)  # 随机选择一个动作
        action = action.item()  # 将张量转换为整数

        def flying():
            return 0, 1, 0

        def speed_up():
            return 2, 1, 0

        def speed_down():
            return -1, 1, 0

        def turn_left():
            return 0, 5, -math.pi/3

        def turn_right():
            return 0, 5, -math.pi/3

        def pull_up():
            return 0, 5, 0

        def drive_down():
            return 0, 5, 0

        cases = {
            0: flying(),
            1: speed_up(),
            2: speed_down(),
            3: turn_left(),
            4: turn_right(),
            5: pull_up(),
            6: drive_down(),
        }
        action = cases[action]
        return action

    def plot(self, ax, x):
        ax.cla()  # 清空图表
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def learn(self):
        # learn 200 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 评估网络每学习200次就将参数传给目标网络
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1: NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # 评估网络Q值
        q_next = self.target_net(batch_next_state).detach()  # 目标网络Q值
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # 计算目标Q值

        loss = self.loss(q_eval, q_target)  # 计算损失函数
        self.optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新神经网络参数


def main():
    done = False
    # # 初始化render模块
    # render = env.render()
    # # 初始化MVController模块
    # mvcontroller = env.MvController()

    net = Dqn()

    for episode in range(EPISODES):
        state = env.reset()
        Init_state = state
        step_counter = 0
        while True:
            step_counter += 1
            render = env.render()
            action = net.choose_action(state)  # 选择动作
            next_state, reward, done, _, _ = env.step(action)
            net.store_trans(state, action, reward, next_state)  # 将状态转换存储到经验池中
            env. recorder(step_counter)
            render.render3D()

            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn()
                if done:
                    print("episode {}, the reward is {}".format(episode, round(reward, 3)))
            if done:
                # step_counter_list.append(step_counter)
                # net.plot(net.ax, step_counter_list)
                break
            elif step_counter >= 500:
                break
            state = next_state


if __name__ == '__main__':
    main()
