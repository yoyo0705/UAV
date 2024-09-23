import csv
import random
import sys
import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from matplotlib.image import imread
import matplotlib.style as mplstyle

G = 10
mplstyle.use('fast')

class UAVEnv(gym.Env):
    # 构造无人机环境
    def __init__(self, uav_num, map_w, map_h, map_z, Init_state):
        super(UAVEnv, self).__init__()
        self.uav_num = uav_num
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.position_pool = [[] for _ in range(2)]
        self.position_pool_new = [[] for _ in range(2)]
        self.state = Init_state
        # 定义无人机的动作空间和观测空间
        self.action_space = spaces.Box(low=np.array([-10, -10, -math.pi] * self.uav_num),
                                       high=np.array([10, 10, math.pi] * self.uav_num), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([90, 0, 0, 0, -math.pi / 2, -math.pi] * 2),
                                            high=np.array([400, 3000, 3000, 10000, math.pi / 2, math.pi]
                                                          * 2), dtype=np.float32)


    # 无人机的状态更新函数
    def step(self, actions, step_counter):
        for i in range(2):
            
            def flying():
                return 0, 1, 0
    
            def speed_up():
                return 2, 1, 0
    
            def speed_down():
                return -1, 1, 0
    
            def turn_left():
                return 0, 5, -math.pi / 3
    
            def turn_right():
                return 0, 5, -math.pi / 3
    
            def pull_up():
                return 0, 5, 0
    
            def drive_down():
                return 0, -5, 0
    
            cases = {
                0: flying(),
                1: speed_up(),
                2: speed_down(),
                3: turn_left(),
                4: turn_right(),
                5: pull_up(),
                6: drive_down(),
            }
            actions[i] = cases[actions[i]]
            # actions = np.array(actions).reshape(self.uav_num, 3)    #改变形状
        # 根据动力学方程更新
        for i in range(2):
            t = 0.25
            self.state[i][0] += t * G * (actions[i][0] - math.sin(self.state[i][4]))
            self.state[i][5] += t * G / self.state[i][0] * actions[i][1] * math.sin(actions[i][2]) /math.cos(self.state[i][4])
            self.state[i][4] += t * G / self.state[i][0] * (actions[i][1] * math.cos(actions[i][2]) - math.cos(self.state[i][4]))
            self.state[i][1] += t * self.state[i][0] * math.cos(self.state[i][4]) * math.sin(self.state[i][5])
            self.state[i][2] += t * self.state[i][0] * math.cos(self.state[i][4]) * math.cos(self.state[i][5])
            self.state[i][3] += t * self.state[i][0] * math.sin(self.state[i][4])
            # 判断是否出界
            # if x < 0 or y < 0 or z < 0 or x > 3000 or y > 3000 or z > 5000
                

        #空战相对态势
        psi_r = self.state[0][5]
        psi_b = self.state[1][5]
        theta_r = self.state[0][4]
        theta_b = self.state[1][4]
        vr = [math.sin(psi_r) * math.cos(theta_r), math.cos(psi_r) * math.cos(theta_r) , math.sin(theta_r)]
        vb = [math.sin(psi_b) * math.cos(theta_b), math.cos(psi_b) * math.cos(theta_b) , math.sin(theta_b)]
        dd = [self.state[0][1] - self.state[1][1], self.state[0][2] - self.state[1][2] , self.state[0][3] - self.state[1][3]]
        d = np.linalg.norm(dd)
        AA = math.acos(np.dot(vb, dd) / d)
        ATA = math.acos(np.dot(vr, dd) / d)
        AAb = math.pi - ATA
        ATAb = math.pi - AA
        detla_h = self.state[0][3] - self.state[1][3]
        detla_v = np.linalg.norm(vr) - np.linalg.norm(vb)

        #奖励函数设计

        #角度奖励
        reward_a = 1-(AA+ATA)/math.pi  # [-1,1]

        #距离奖励 双方相同
        reward_d = math.exp(-math.fabs(d-1000)/1000) # [-1,1]

        #综合奖励函数
        reward_r = reward_a * reward_d    #红方奖励

        
        # 判断胜负
        if step_counter > 500:
            done = True
            result = 3
        elif self.state[0][3] < 100:
            done = True
            result = 2
            reward_r = -5
        elif self.state[1][3] < 100:
            done = True
            result = 1
            reward_r = 5
        elif AA < math.pi / 3 and ATA < math.pi/6 and d < 2000:
            done = True
            result = 1
            reward_r = 10
        elif AA < math.pi * 5 / 6 and ATA > math.pi * 2 / 3 and d < 2000:
            done = True
            result = 2
            reward_r = -10
        # elif self.state[i][1] < 0 or self.state[i][2] < 0 or self.state[i][3] < 0 or self.state[i][1] > 3000 or \
        #         self.state[i][2] > 3000 or self.state[i][3] > 5000:
        #     done = True
        #     result = 4
        # elif self.state[i][0] < 90 or self.state[i][0] > 400:
        #     done = True
        #     result = 4
        else:
            done = False
            result = 0
        return self.state, reward_r, done, result, {}

    def reset(self):
        self.state = [[240.0, 3000.0, 3000.0, 2800.0, 0.0, -3 * math.pi / 4],
                      [250.0, 0.0, 0.0, 2900.0, 0.0, math.pi / 4]]
        self.position_pool_new = [[], []]
        return self.state

    # 记录无人机的飞行轨迹函数
    def recorder(self, env_t):
        for i in range(2):
            x, y, z = self.state[i][1:4]    # python数据索引规则
            position = [x / 100, y / 100, z / 100, env_t]
            self.position_pool_new[i].append(position)
            self.position_pool[i].append(position)

                
    # 画面渲染函数，使用matplotlib库绘制地图、障碍物、无人机轨迹
    def render(self, uav_num, state, map_w, map_h, map_z):
        self.uav_num = uav_num
        self.state = state
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.line = []
        # self.AimsPoint = [[] for _ in range(2)]
        self.Head = []

        # 创建画布
        self.fig = plt.figure(figsize=(self.map_w, self.map_h))  # 设置画布大小
        ax = self.fig.add_subplot(111, projection='3d')  # 创建三维坐标系

        ax.set_xlim(0, map_w + 1)
        ax.set_ylim(0, map_h + 1)
        ax.set_zlim(0, map_z + 1)
        ax.set_xlabel('X', fontsize=50)
        ax.set_ylabel('Y', fontsize=50)
        ax.set_zlabel('Z', fontsize=50)

        x_traj, y_traj, z_traj, _ = zip(*self.position_pool_new[0])
        l = ax.plot(x_traj, y_traj, z_traj, color='r', alpha=1, linewidth=5, label='natural')
        self.line.append(l)
        head = ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='r', s=300)
        self.Head.append(head)

        x_traj, y_traj, z_traj, _ = zip(*self.position_pool_new[1])
        l = ax.plot(x_traj, y_traj, z_traj, color='b', alpha=1, linewidth=5, label='double')
        self.line.append(l)
        head = ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='b', s=300)
        self.Head.append(head)

        # ax2d = ax.add_axes([0.7, 0.5, 0.2, 0.3])  # 调整位置和大小
        # ax2d.legend(handles=[scatter1, scatter2], labels=['Group 1', 'Group 2'])
        ax.legend(fontsize=70)
        # plt.show()
