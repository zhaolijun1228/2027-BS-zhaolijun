import numpy as np
import matplotlib.pyplot as plt

# 环境参数
CHANNEL_NUM = 4       # 信道数量
EPISODES = 200        # 训练轮数
LEARNING_RATE = 0.1
GAMMA = 0.95
EPSILON = 0.1

# Q表初始化
Q_table = np.zeros((CHANNEL_NUM, CHANNEL_NUM))

def choose_action(state):
    if np.random.uniform(0,1) < EPSILON:
        return np.random.randint(0, CHANNEL_NUM)
    else:
        return np.argmax(Q_table[state, :])

def get_reward(state, action):
    # 若动作和当前状态相同，判定信道冲突，奖励低
    if action == state:
        return -1
    else:
        return 1

# 训练
reward_list = []
for epi in range(EPISODES):
    state = np.random.randint(0, CHANNEL_NUM)
    total_reward = 0
    while True:
        action = choose_action(state)
        reward = get_reward(state, action)
        next_state = np.random.randint(0, CHANNEL_NUM)
        # Q值更新
        Q_table[state, action] = Q_table[state, action] + LEARNING_RATE * (
            reward + GAMMA * np.max(Q_table[next_state,:]) - Q_table[state, action]
        )
        state = next_state
        total_reward += reward
        break
    reward_list.append(total_reward)

# 绘图
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning Channel Selection Reward")
plt.show()
