import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. Yapay Sinir Ağı Mimarisi (Neural Network)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 5 Giriş (Saat, Dış Isı, İç Isı, Yalıtım, İnsan) -> 64 -> 64 -> 3 Çıkış (Aksiyonlar)
        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# 2. Deneyim Hafızası (Replay Buffer)
class ReplayBuffer:
    def __init__(self, capacity=10000):
        # Ajanın son 10.000 adımını (tecrübesini) hafızada tutar
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Hafızadan rastgele bir mini-batch çeker (Unutmayı engellemek için)
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# 3. DQN Ajanı (Kontrolcü)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # RL Hiperparametreleri
        self.gamma = 0.99           # Gelecekteki ödüllerin önemi (Optimizasyon için kritik)
        self.lr = 1e-3              # Öğrenme hızı
        self.batch_size = 64
        self.epsilon = 1.0          # Başlangıçta %100 keşif (rastgele hareket)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  
        
        # Cuda (GPU) kontrolü - Sistemin ekran kartını destekliyorsa onu kullanır
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Kararlılık için İki Ayrı Ağ (Policy ve Target)
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()

    def act(self, state):
        # Epsilon-Greedy Stratejisi
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Rastgele aksiyon (Keşif)
        
        # Ağı kullanarak en iyi aksiyonu seçme (Sömürü)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def learn(self):
        # Yeterli tecrübe birikmediyse öğrenmeyi bekle
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Verileri PyTorch Tensörlerine Çevir
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Mevcut Q Değerleri
        q_values = self.policy_net(states).gather(1, actions)
        
        # Hedef Q Değerleri (Bellman Denklemi)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        
        # Kayıp (Loss) Hesaplama ve Ağı Güncelleme (Geriye Yayılım)
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Hedef ağı periyodik olarak ana ağın ağırlıklarıyla güncelleriz
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # Zamanla rastgele hareket etme ihtimalini düşür
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay