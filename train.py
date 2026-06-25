import numpy as np
import torch
from env import SmartThermostatDRLEnv
from agent import DQNAgent

def train_agent():
    # Çevreyi ve Ajanı Başlat
    env = SmartThermostatDRLEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Eğitim Parametreleri
    episodes = 500  # Ajan 500 gün boyunca ofisi yönetecek
    batch_size = 64
    target_update_freq = 10 # Hedef ağı her 10 bölümde bir güncelle
    
    print("DQN Eğitim Süreci Başlıyor...")
    print("-" * 50)
    
    # Ödülleri kaydedeceğimiz boş liste
    reward_history = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for time_step in range(env.max_steps):
            # 1. Ajan durumu analiz eder ve bir aksiyon seçer
            action = agent.act(state)
            
            # 2. Çevre bu aksiyona tepki verir (Yeni durum ve ödül döner)
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Bu tecrübe hafızaya kaydedilir
            agent.memory.push(state, action, reward, next_state, done)
            
            # 4. Durum güncellenir
            state = next_state
            
            # 5. Ajan hafızasından rastgele örnekler çekerek öğrenir (Backpropagation)
            agent.learn()
            
            if done:
                break
                
        # Bölüm sonu işlemleri
        agent.decay_epsilon()
        reward_history.append(total_reward)
        
        # Hedef ağı güncelleme
        if e % target_update_freq == 0:
            agent.update_target_network()
            
        # İlerleme Çıktısı (Her 10 bölümde bir ekrana yazdır)
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Bölüm: {e + 1}/{episodes} | Ortalama Ödül: {avg_reward:.2f} | Keşif Oranı (Epsilon): {agent.epsilon:.3f}")

    # Eğitilen Modeli Kaydet
    print("-" * 50)
    print("Eğitim Tamamlandı! Model 'dqn_model.pth' olarak kaydediliyor...")
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    
    # --- İŞTE YENİ EKLENEN KISIM ---
    # Eğitim geçmişini %100 organik olarak ayrı bir dosyaya kaydediyoruz
    np.save("reward_history.npy", reward_history)
    print("Eğitim geçmişi 'reward_history.npy' olarak başarıyla kaydedildi!")
    print("Kaydedildi. Artık test aşamasına ve grafik çizimine geçebiliriz!")

if __name__ == "__main__":
    train_agent()