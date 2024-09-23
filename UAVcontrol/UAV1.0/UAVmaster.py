import csv
import random
import sys
import gym
from gym import spaces
import numpy as np
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
        self.state = Init_state
        # 定义无人机的动作空间和观测空间
        self.action_space = spaces.Box(low=np.array([-10, -10, -math.pi] * self.uav_num),
                                       high=np.array([10, 10, math.pi] * self.uav_num), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([90, 0, 0, 0, -math.pi / 2, -math.pi] * 2),
                                            high=np.array([400, 3000, 3000, 5000, math.pi / 2, math.pi]
                                                          * 2), dtype=np.float32)


    # 无人机的状态更新函数
    def step(self, actions):
        actions = np.array(actions).reshape(self.uav_num, 3)    #改变形状
        # 根据动力学方程更新
        for i in range(self.uav_num):
            t = 0.25
            self.state[i][0] += t * G * (actions[i][0] - math.sin(self.state[i][4]))
            self.state[i][5] += t * G / self.state[i][0] * actions[i][1] * math.sin(actions[i][2]) /math.cos(self.state[i][4])
            self.state[i][4] += t * G / self.state[i][0] * (actions[i][1] * math.cos(actions[i][2]) - math.cos(self.state[i][4]))
            self.state[i][1] += t * self.state[i][0] * math.cos(self.state[i][4]) * math.sin(self.state[i][5])
            self.state[i][2] += t * self.state[i][0] * math.cos(self.state[i][4]) * math.cos(self.state[i][5])
            self.state[i][3] += t * self.state[i][0] * math.sin(self.state[i][4])

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
        if AA < math.pi / 3 and ATA < math.pi/6 and d < 2000 :
            done = True
            print("红方击败蓝方")
        elif AA < math.pi * 5 / 6 and ATA > math.pi * 2 / 3 and d < 2000:
            done = True
            print("红方被蓝方击败")
        elif self.state[0][3] <= 100:
            done = True
            print("红方坠毁")
        elif self.state[1][3] <= 100:
            done = True
            print("蓝方坠毁")
        else:
            done = False

        return self.state, reward_r, done, {}

    def reset(self):
        self.state = [[240.0, 3000.0, 3000.0, 2800.0, 0.0, -3 * math.pi / 4],
                        [250.0, 1500.0, 1500.0, 2500.0, 0.0, math.pi / 4]]
        return self.state

    # 记录无人机的飞行轨迹函数
    def recorder(self, env_t):
        for i in range(2):
            x, y, z = self.state[i][1:4]    # python数据索引规则
            if x < 0 or y < 0 or z < 0 or x > 3000 or y > 3000 or z > 5000:
                print("超出地图范围")
                done = True
            else:
                position = [x / 100, y / 100, z / 100, env_t]
                self.position_pool[i].append(position)
                done = False
        return done

                
    # 画面渲染函数，使用matplotlib库绘制地图、障碍物、无人机
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
        self.ax = self.fig.add_subplot(111, projection='3d')  # 创建三维坐标系

        self.ax.set_xlim(0, map_w + 1)
        self.ax.set_ylim(0, map_h + 1)
        self.ax.set_zlim(0, map_z + 1)

    # 绘制无人机
    def render3D(self):
        for i in range(2):
            x_traj, y_traj, z_traj, _ = zip(*self.position_pool[i])
            l = self.ax.plot(x_traj, y_traj, z_traj, color='black', alpha=1, linewidth=3)
            self.line.append(l)
            head = self.ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='darkorange', s=300)
            self.Head.append(head)