import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve():
    print("Organik eğitim verileri okunuyor...")
    
    try:
        # Veriyi çekme
        rewards = np.load("reward_history.npy")
    except FileNotFoundError:
        print("HATA: 'reward_history.npy' dosyası bulunamadı. Lütfen önce train.py dosyasını çalıştırın.")
        return

    episodes = range(1, len(rewards) + 1)
    
    # --- HAREKETLİ ORTALAMA (MOVING AVERAGE) HESAPLAMA ---
    window = 20 # 20 günlük hareketli ortalama  
    weights = np.repeat(1.0, window) / window
    moving_avg = np.convolve(rewards, weights, 'valid')
    ma_episodes = range(window, len(rewards) + 1)

    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(12, 6))
    
    # 1. Arka Planda Ham Veriler (Dalgalı Mavi Çizgi)
    plt.plot(episodes, rewards, color='blue', alpha=0.4, linewidth=1, label='Episode Reward')
    
    # 2. Ön Planda Hareketli Ortalama (Kalın Kırmızı Çizgi)
    plt.plot(ma_episodes, moving_avg, color='red', linewidth=2.5, label='20 Episode Ortalama')
    
    # Sıfır noktasına başarı eşiği çizgisi
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Sıfır Noktası (Başarı Eşiği)')

    plt.title('DQN Akıllı Termostat: Eğitim Sonuçları ve Öğrenme Eğrisi', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Toplam Reward', fontsize=12, fontweight='bold')
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_egitim_grafigi.png', dpi=300)
    plt.show()
    
    print("Şov Zamanı! Profesyonel eğitim grafiği 'dqn_egitim_grafigi.png' adıyla kaydedildi.")

if __name__ == "__main__":
    plot_learning_curve()
