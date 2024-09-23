import gym
import math
import matplotlib.pyplot as plt


env = gym.make('UAVEnv-v0', uav_num=1, map_w=3000, map_h=3000, map_z=5000,
               Init_state=[[240.0, 3000.0, 3000.0, 2800.0, 0.0, -3 * math.pi / 4],
                           [250.0, 0.0, 0.0, 0.0, 0.0, math.pi / 4]])
env = env.unwrapped
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]


# def plot(self, ax, x):  # ？
#     ax.cla()  # 清空图表
#     ax.set_xlabel("episode")
#     ax.set_ylabel("total reward")
#     ax.plot(x, 'b-')
#     plt.pause(0.000000000000001)  # 用于暂停绘图???


def choose_action():  # 根据当前状态选择一个动作(从动作控制器中选择）

    def flying():
        return 0, 1, 0

    def speed_up():
        return 2, 1, 0

    def speed_down():
        return -1, 1, 0

    def turn_left():
        return 0, 5, -math.pi / 3

    def turn_right():
        return 0, 5, math.pi / 3

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
    action = cases[4]
    return action


def main():
    state = env.reset()
    env.render(1, state, map_w=3000 / 100, map_h=3000 / 100, map_z=5000 / 100)
    step_counter = 0
    while True:
        step_counter += 1
        action = choose_action()
        next_state, reward, done, _ = env.step(action)
        env.recorder(step_counter)
        state = next_state
        if done:
            break
        print(step_counter)
    env.render3D()
    plt.show()
        # if step_counter >=50:
        #     break

if __name__ == '__main__':
    main()
