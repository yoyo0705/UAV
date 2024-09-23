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
# MEMORY_CAPACITY = 2000  # 存储器容量
MEMORY_CAPACITY = 2000  # 存储器容量
BATCH_SIZE = 64  # 神经网络学习时从记忆存储单元中抽取的经验个数
Q_NETWORK_ITERATION = 200  # 评估网络每学习200次就将参数传给目标网络
STEPS_PER_LEARNING = 5  # 无人机每决策 5 次，神经网络就学习一次
EPISODES = 3000

env = gym.make('UAVEnv-v0', uav_num=2, map_w=3000, map_h=3000, map_z=5000,
               Init_state=[[240.0, 3000.0, 3000.0, 2800.0, 0.0, -3 * math.pi / 4],
                          [250.0, 0.0, 0.0, 2900.0, 0.0, math.pi / 4]])
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
    def __init__(self, double_q=True):  # 构造函数
        self.eval_net, self.target_net = Net(), Net()  # 评估网络 目标网络
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))  # 经验池设置
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0  # 用于跟踪经验池中的数据数量和学习次数
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)  # adam优化器，用于更新神经网络的权重
        self.loss = nn.MSELoss()  # 均方误差损失函数
        self.losses = []
        self.double_q = double_q
        self.q = []
        # self.fig, self.ax = plt.subplots()

    def store_trans(self, state, action, reward, next_state):  # 将状态转换存储到经验池中
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, action, reward, next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):  # 根据当前状态选择一个动作(从动作控制器中选择）
        state = np.array(state).reshape(1, 12)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 转化为pytorch张量
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)  # 计算7种动作分别对应的Q值
            # action = torch.max(action_value, 1)[1].data # get action whose q is max    # 选出最大Q值对应的动作
            q_eval = torch.max(action_value).item()
            self.q.append(q_eval)
            # self.steps_q += 1
            action = torch.argmax(action_value)
        else:
            action = np.random.randint(0,7)  # 随机选择一个动作
        action = np.array(action).item()  # 将张量转换为整数
        return action

    def learn(self):
        # learn 200 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 评估网络每学习200次就将参数传给目标网络
        self.learn_counter += 1

        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1: NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])
        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # 评估网络Q值
        if self.double_q:
            best_next_actions = self.eval_net(batch_next_state).argmax(1, keepdim=True)
            q_next = self.target_net(batch_next_state).gather(1, best_next_actions)
        else:
            q_next = self.target_net(batch_next_state).detach()  # 目标网络Q值
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # 计算目标Q值
        # q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        loss = self.loss(q_eval, q_target)  # 计算损失函数
        # self.losses.append(loss)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新神经网络参数

        return self.losses, self.q

def train(self):
    # netr = Dqn(double_q=False)
    for episode in range(1500):
        state = env.reset()
        step_counter = 0
        done = False
        while True:
            step_counter += 1
            action_r = self.choose_action(state)  # 选择动作
            action_b = 0    # 蓝方默认正常飞行
            action = [action_r, action_b]
            next_state, reward_r, done, result, _ = env.step(action, step_counter)
            self.store_trans(np.array(state).ravel(), action_r, reward_r, np.array(next_state).ravel())  # 将状态转换存储到经验池中
            env.recorder(step_counter)
            state = next_state
            if step_counter % STEPS_PER_LEARNING == 0:
                # if netr.memory_counter >= MEMORY_CAPACITY and netb.memory_counter >= MEMORY_CAPACITY:
                self.learn()
            if done:
                print("episode {}, the reward_r is {}".format(episode, round(reward_r, 3)))
                break
    # env.render(1, state, map_w=3000 / 100, map_h=3000 / 100, map_z=5000 / 100)

def main():
    netr = Dqn(double_q=True)
    netb = Dqn(double_q=True)
    state = env.reset()
    result_list = np.zeros(4)
    losses_r = []
    losses_b = []
    q_evallist_r = []
    q_evallist_b = []
    train(netr)
    for episode in range(EPISODES):
        episode_reward = 0
        state = env.reset()
        step_counter = 0
        result = 0
        done = False
        state_pool = []
        while True:
            step_counter += 1
            action_r = netr.choose_action(state)  # 选择动作
            action_b = netb.choose_action(state)  # 选择动作
            action = [action_r, action_b]
            # q_evallist_r.append(q_eval_r.item())
            # q_evallist_b.append(q_eval_b.item())
            next_state, reward_r, done, result, _ = env.step(action, step_counter)
            netr.store_trans(np.array(state).ravel(), action_r, reward_r, np.array(next_state).ravel())  # 将状态转换存储到经验池中
            netb.store_trans(np.array(state).ravel(), action_b, -reward_r, np.array(next_state).ravel())  # 将状态转换存储到经验池中
            env.recorder(step_counter)
            state = next_state
            state_pool.append(np.array(state).ravel())
            if step_counter % STEPS_PER_LEARNING == 0:
                # if netr.memory_counter >= MEMORY_CAPACITY and netb.memory_counter >= MEMORY_CAPACITY:
                losses_r, q_evallist_r = netr.learn()
                losses_b, q_evallist_b = netb.learn()
                # losses_r.append(loss_r)
                # losses_b.append(loss_b)
                # steps_loss += 1
            if done:
                print("episode {}, the reward_r is {}, the reward_b is {}".format(episode, round(reward_r, 3), round(-reward_r, 3)))
                # if reward_r == 5 or reward_r == 10:
                #     env.render(1, state, map_w=3000 / 100, map_h=3000 / 100, map_z=5000 / 100)
                #     plt.savefig('./img/l-{}.png'.format(episode))
                #     plt.close()
                # rewards_list.append(episode_reward / step_counter)
                if result == 1:
                    result_list[0] += 1
                elif result == 2:
                    result_list[1] += 1
                elif result == 3:
                    result_list[2] += 1
                elif result == 4:
                    result_list[3] += 1
                break
    filename = "states.csv"
    np.savetxt(filename, state_pool, delimiter=',')
    # 打印胜负结果
    print("对战结果如下：\t红胜\t蓝胜\t平局\t无效")
    for i in range(4):
        print("\t{}".format(result_list[i]))

    # plt.plot(list(range(EPISODES)), rewards_list)
    # plt.savefig("episode_reward.png")

    # env.render(1, state, map_w=3000 / 100, map_h=3000 / 100, map_z=10000 / 100)
    # plt.figure()
    # plt.savefig('./img/pic-{}.png'.format(1))
    plt.figure()
    # plt.plot(range(len(losses_b)), losses_r[:len(losses_b)], c='r', label='natural')
    plt.plot(range(200), losses_b[:200], c='b', label='double')
    plt.ylabel('Loss')
    plt.xlabel('training steps')
    plt.legend()
    plt.savefig('./img/loss.png')
    plt.figure()
    # plt.plot(list(range(len(q_evallist_b))), q_evallist_r[:len(q_evallist_b)], c='r', label='natural')
    # plt.plot(list(range(len(q_evallist_b))), q_evallist_b, c='b', label='double')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.legend()
    plt.savefig('./img/Q eval.png')
    # plt.show()


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    main()
