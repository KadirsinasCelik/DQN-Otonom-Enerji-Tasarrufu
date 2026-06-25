import torch
import numpy as np
import matplotlib.pyplot as plt
from  env import SmartThermostatDRLEnv
from  agent import DQNAgent

def evaluate_and_plot():
    print("Test simülasyonu başlatılıyor...")
    
    # Çevre ve Ajanı Başlat
    env = SmartThermostatDRLEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Eğitilmiş Modeli Yükle (weights_only=True eklendi, sarı uyarı artık çıkmayacak)
    try:
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", weights_only=True))
        agent.policy_net.eval()
        print("Eğitilmiş model başarıyla yüklendi!")
    except FileNotFoundError:
        print("HATA: 'dqn_model.pth' dosyası bulunamadı. Önce 'train.py' dosyasını çalıştırın.")
        return

    # Epsilon = 0: Rastgele hareket yok, tamamen akıllı kararlar
    agent.epsilon = 0.0
    
    # Grafik Çizimi İçin Veri Kayıt Listeleri
    time_steps = []
    inside_temps = []
    outside_temps = []
    actions_taken = []
    human_counts = []
    
    state = env.reset()
    total_reward = 0
    
    # 1 Tam Günlük (1440 Dakika) Simülasyon
    for time_step in range(env.max_steps):
        # Ajan en iyi aksiyonu seçer
        action = agent.act(state)
        
        # Çevre tepki verir
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Verileri Kaydet
        time_steps.append(time_step)
        inside_temps.append(env.real_temp)
        outside_temps.append(env.outside_temp)
        actions_taken.append(action)
        human_counts.append(env.num_humans)
        
        state = next_state
        if done:
            break
            
    print(f"Test Günü Toplam Ödülü: {total_reward:.2f}")
    
    # --- GRAFİK ÇİZİMİ (Hocalara Şov Kısmı) ---
    plt.figure(figsize=(15, 10))
    
    # 1. Grafik: Sıcaklık Değişimleri ve Mesai Hedefi
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, inside_temps, label='İç Ortam Sıcaklığı (°C)', color='red', linewidth=2)
    plt.plot(time_steps, outside_temps, label='Dış Ortam Sıcaklığı (°C)', color='blue', linestyle='--', alpha=0.6)
    
    # Mesai saatlerini yeşil arka planla vurgula (08:00-12:00 ve 13:00-17:00)
    plt.axvspan(480, 720, color='green', alpha=0.1, label='Aktif Mesai (Hedef 25°C)')
    plt.axvspan(780, 1020, color='green', alpha=0.1)
    
    # 25 Derece Hedef Çizgisi
    plt.axhline(y=25.0, color='black', linestyle=':', label='Hedef (25°C)')
    
    plt.title('DQN Akıllı Termostat: 1 Günlük Sıcaklık Yönetimi')
    plt.ylabel('Sıcaklık (°C)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 2. Grafik: İnsan Sayısı (Isı Yükü)
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, human_counts, label='Odadaki İnsan Sayısı', color='purple', drawstyle='steps-post')
    plt.ylabel('Kişi Sayısı')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 3. Grafik: Ajanın Aksiyonları (Enerji Tüketimi)
    plt.subplot(3, 1, 3)
    # Aksiyonları daha rahat okumak için renkli noktalar kullanıyoruz
    actions_np = np.array(actions_taken)
    plt.scatter(np.where(actions_np == 0)[0], actions_np[actions_np == 0], color='gray', label='Kapalı (0W)', s=10)
    plt.scatter(np.where(actions_np == 1)[0], actions_np[actions_np == 1], color='orange', label='Bekleme (2500W)', s=10)
    plt.scatter(np.where(actions_np == 2)[0], actions_np[actions_np == 2], color='red', label='Performans (5000W)', s=10)
    
    plt.ylabel('Klima Modu')
    plt.xlabel('Günün Dakikası (0 - 1440)')
    plt.yticks([0, 1, 2], ['Kapalı', 'Bekleme', 'Performans'])
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_test_sonuclari.png')
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()