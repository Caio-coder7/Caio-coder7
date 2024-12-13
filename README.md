- 👋 Hi, I’m @Caio-coder7

import gym
from gym import spaces
import numpy as np
import random

class SimpleGameEnv(gym.Env):
    def __init__(self):
        super(SimpleGameEnv, self).__init__()
        
        # Espaço de ações: 0 (esquerda), 1 (direita)
        self.action_space = spaces.Discrete(2)
        
        # Espaço de observação: posição no intervalo [0, 10]
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        
        self.state = 5  # Posição inicial
        self.goal = 10  # Objetivo

    def reset(self):
        self.state = 5
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # Atualiza a posição com base na ação
        if action == 0:  # Esquerda
            self.state = max(0, self.state - 1)
        elif action == 1:  # Direita
            self.state = min(10, self.state + 1)

        # Calcula a recompensa
        done = self.state == self.goal
        reward = 1 if done else -0.1

        return np.array([self.state], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        print(f"Posição atual: {self.state}")

# Ambiente
env = SimpleGameEnv()

# Hiperparâmetros para o Q-Learning
alpha = 0.1       # Taxa de aprendizado
gamma = 0.99      # Fator de desconto
epsilon = 1.0     # Taxa de exploração
epsilon_decay = 0.995
min_epsilon = 0.01

# Inicialização da Tabela-Q
q_table = np.zeros((11, 2))

# Episódios de treinamento
n_episodes = 1000

for episode in range(n_episodes):
    state = int(env.reset())
    done = False
    total_reward = 0

    while not done:
        # Escolhe ação (exploração vs exploração)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploração
        else:
            action = np.argmax(q_table[state])  # Exploração

        # Executa ação
        next_state, reward, done, _ = env.step(action)
        next_state = int(next_state[0])

        # Atualiza a Tabela-Q
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    # Decai o epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Log de progresso
    if (episode + 1) % 100 == 0:
        print(f"Episódio {episode + 1}, Recompensa total: {total_reward}")

# Teste do agente treinado
state = int(env.reset())
done = False

env.render()
while not done:
    action = np.argmax(q_table[state])
    state, _, done, _ = env.step(action)
    state = int(state[0])
    env.render()

print("Treinamento concluído!")

<!---
Caio-coder7/Caio-coder7 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
