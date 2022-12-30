import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import *

# 定义三维数据
xx = np.arange(0, 24)
yy = np.arange(0, 24)
X, Y = np.meshgrid(xx, yy)
# 作图
Z = np.zeros((len(xx), len(yy)))
Z[0:2, 8:19] = 10

Z_shopping = Z
######################################

# 定义三维数据
xx = np.arange(0, 24)
yy = np.arange(0, 12)
X, Y = np.meshgrid(xx, yy)
# 作图
Z = np.zeros((len(yy), len(xx)))
for k in (20, 19, 18, 17, 16, 15):
    for i in range(0, 12):
        for j in range(0, 24):
            if i + j == k:
                Z[i, j] = k - 14

for k in (9, 10, 11, 12):
    for i in range(0, 12):
        for j in range(0, 24):
            if i + j == k:
                Z[i, j] = 26 - 2 * k

for i in range(0, 12):
    for j in range(0, 24):
        if i + j <= 8:
            Z[i, j] = 10
        if i + j <= 1:
            Z[i, j] = 9
        if i + j >= 20:
            Z[i, j] = 9
        if i + j >= 27:
            Z[i, j] = 10
        if i + j >= 30:
            Z[i, j] = 9
        if i + j >= 33:
            Z[i, j] = 0

Z_home = Z

# 定义三维数据
xx = np.arange(0, 24)
yy = np.arange(0, 12)
X, Y = np.meshgrid(xx, yy)
# 作图
Z = np.zeros((len(yy), len(xx)))
for i in range(0, 12):
    for j in range(0, 24):
        Z[i, j] = 1

for k in range(13, 19):
    for i in range(0, 12):
        for j in range(0, 24):
            if i + j == k:
                Z[i, j] = k - 9

for i in range(0, 12):
    for j in range(0, 24):
        if i + j >= 19:
            Z[i, j] = 10
        if i + j >= 24:
            Z[i, j] = 0

Z_leisure = Z

# 定义三维数据
xx = np.arange(0, 24)
yy = np.arange(0, 12)
X, Y = np.meshgrid(xx, yy)
# 作图
Z = np.zeros((len(yy), len(xx)))

for i in range(0, 10):
    for j in range(0, 24):
        Z[i, j] = 10 - 10 * i / 10

for i in range(9, 11):
    for j in range(0, 24):
        Z[i, j] = 400

for i in range(0, 12):
    for j in range(0, 24):
        if i + j <= 8 or i + j >= 20:
            Z[i, j] = 0

Z_work = Z


class Env:
    def __init__(self):
        self.reward_leisure = Z_leisure
        # self.reward_leisure = np.zeros((len(yy), len(xx)))
        self.reward_home = Z_home
        self.reward_shop = Z_shopping
        # self.reward_shop = np.zeros((len(yy), len(xx)))
        self.reward_work = Z_work

    def get_reward(self, activity_id, start_time, dur):
        cur_act = {
            0: self.reward_home,
            1: self.reward_work,
            2: self.reward_shop,
            3: self.reward_leisure,
            4: self.reward_home,
        }
        re_table = cur_act.get(activity_id)
        try:
            res = re_table[dur][start_time]
        except:
            print(dur, start_time)
            print(re_table)
            print(activity_id)
        return re_table[dur][start_time]

    def step(self, state, action):
        reward = 0
        activity_no = 0 if state[0] == 4 else state[0]
        start_time = state[1]
        dur = state[2]
        if action == 0:
            # 超24小时了
            if start_time + dur + 1 >= 24:
                return True, reward, None

            # dur 超过 12 * 4 自动跳转下一个行为
            if dur + 1 >= 12:
                reward -= travel_time[activity_no]
                # 加上旅程超24小时了
                if start_time + dur + travel_time[activity_no] >= 24:
                    return True, reward, None
                # 正常转换
                start_time = start_time + dur + travel_time[activity_no]
                activity_no += 1
                dur = 0
            else:
                dur += 1

            reward += self.get_reward(activity_no, start_time, dur)
        else:
            # travel cost
            reward -= travel_time[activity_no]
            start_time = start_time + dur + travel_time[activity_no]
            # 超24小时了
            if start_time >= 24:
                return True, reward, None
            if state[0] == 3:
                # go home 并且计算剩下的reward
                i = 0
                while start_time + i < 24 and i < 12:
                    reward += self.get_reward(0, start_time, i)
                    i += 1
                return True, reward, None
            else:
                activity_no += 1
                if activity_no == 4:
                    activity_no = 0
                dur = 0
                reward += self.get_reward(activity_no, start_time, dur)
        return False, reward, (activity_no, start_time, dur)


def train(env, epo):
    episodes = epo
    runs = 1
    smooth_rewards = np.zeros(episodes)
    rewards_q = np.zeros(episodes)
    test_score = []
    for r in range(runs):
        print(r)
        q_table_q = np.ones((4, 24, 12, 2)) * 0.1
        with open('data.txt', 'w') as f:
            for ep in tqdm(range(episodes)):
                # old_q_table = q_table_q.copy()
                f.writelines("------------------------------------" + str(ep) + '\n')
                rewards_q[ep] += q_learning(f, env, q_table_q, 0.2, 0.1)
                # diff = np.sum(np.abs(old_q_table - q_table_q))
                # print("diff = ",diff)
                if ep == episodes - 1:
                    test_score.append(test(q_table_q, env, True, f))
                else:
                    test_score.append(test(q_table_q, env, True, f))
                smooth_rewards[ep] += test_score[-1]
        f.close()
        if r == runs - 1:
            # print(q_table_q)
            # plt.plot(test_score)
            smooth_rewards /= runs
            plt.plot(scipy.signal.savgol_filter(smooth_rewards, 20, 3), label="smooth")
            plt.plot(smooth_rewards, label="non_smooth", alpha=0.2)
            plt.legend()
            plt.show()
            return q_table_q


def test(q_table, env, need_print, f):
    s = (0, 0, 0)
    score = 0
    while True:
        a = 0 if s[0] == 4 else s[0]
        best_act = find_action_from_q_table(q_table, (a, s[1], s[2]))
        is_finish, reward, new_s = env.step(s, best_act)
        if need_print:
            np.argmax(q_table[a][s[1]][s[2]])
            str = "state {} qt {} best_action {} reward = {}".format(s, q_table[a][s[1]][s[2]], best_act, reward)
            f.writelines(str + '\n')
        score += reward
        if is_finish:
            break
        s = new_s
    return score


def q_learning(f, env: Env, q_table, alpha, epsilon):
    old_q_table = None
    state = (np.random.choice(4), np.random.randint(0, 24), np.random.randint(0, 12))
    # state = (1, np.random.randint(0, 24), np.random.randint(0, 12))
    # print("init_state : ", state)
    while True:

        if np.random.binomial(1, epsilon) == 1:
            action = np.random.choice(2)
        else:
            action = find_action_from_q_table(q_table, state)

        is_finish, reward, next_state = env.step(state, action)
        # print("action = ", action, "next state = ", next_state)
        if is_finish:
            break

        act = 0 if state[0] == 4 else state[0]
        next_activity = 0 if next_state[0] == 4 else next_state[0]
        # if next_state[0] == 1 and next_state[2] in (9, 10):
        str1 = "old {} {} {} [{}], value {} ".format(act, state[1], state[2], action,
                                                     q_table[act][state[1]][state[2]][action])
        str2 = "=> reward {} , next_state {} {} {} ,next_value {}".format(reward, next_activity, next_state[1],
                                                                          next_state[2],
                                                                          q_table[next_activity][next_state[1]][
                                                                              next_state[2]])
        # str1 = "old ", act, state[1], state[2], [action], "value ", q_table[act][state[1]][state[2]][action],
        #           " => ", "re ", reward, "next state", [next_activity], [next_state[1]], [next_state[2]],
        #           " next value ", q_table[next_activity][next_state[1]][next_state[2]]
        f.writelines(str1 + '\n')
        f.writelines(str2 + '\n')
        old_q = q_table[act][state[1]][state[2]][action]
        q_table[act][state[1]][state[2]][action] += alpha * (
                reward + 0.99 * np.max(q_table[next_activity, next_state[1], next_state[2], :]) - old_q)
        # - q_table[act][state[1]][state[2]][action])
        # if next_state[0] == 1 and next_state[2] in (9, 10):
        str3 = " new value {}".format(q_table[act][state[1]][state[2]][action])
        f.writelines(str3 + '\n')
        f.writelines("***** " + '\n')
        state = next_state
    f.writelines(str(q_table[0][0]))
    return reward


def find_action_from_q_table(q_table, state):
    activity = 0 if state[0] == 4 else state[0]
    values = q_table[activity][state[1]][state[2]]
    return np.random.choice(np.where(values == np.max(values))[0])


if __name__ == '__main__':
    env = Env()
    travel_time = [1, 1, 1, 1]
    # 构建state矩阵
    # 4 activity , 2 action state_space(activity,start_time,dur)
    state = np.zeros((1, 3))
    action = [0, 1]
    epsilon = 0.8
    alpha = 0.01
    qt = train(env, 100000)
    Z = np.zeros((4, 24, 12))
    for a in range(4):
        for i in range(0, 24):
            for j in range(0, 12):
                # print(i,j)
                # print(np.argmax(qt[0][i][j]))
                Z[a, i, j] = np.argmax(qt[a][i][j])
