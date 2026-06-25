import random
import numpy as np

class SmartThermostatDRLEnv:
    """
    Predictive HVAC Optimization Environment for Deep Reinforcement Learning
    - State: [Minute, OutsideTemp, InsideTemp, U-Value, Humans]
    - Action: 0 (Off), 1 (Standby), 2 (Performance)
    - Schedule: 08:00-12:00 (Active), 12:00-13:00 (Break), 13:00-17:00 (Active)
    """

    def __init__(self):
        # DRL Sabitleri
        self.state_size = 5   # Yapay Sinir Ağına girecek 5 parametre
        self.action_size = 3  
        self.max_steps = 1440 # 1 Gün = 1440 dakika

        self.action_names = {
            0: "Klima Kapali (0W)",
            1: "Bekleme Modu (2500W)",
            2: "Performans Modu (5000W)"
        }

        self.ac_power_limit = {
            0: 0,       
            1: 2500,    
            2: 5000     
        }
        
        self.reset()

    def reset(self):
        """Her bölüm (episode) yepyeni bir gün olarak başlar."""
        self.current_step = 0 # Gece 00:00 (0. dakika)
        
        # Odanın fiziksel özellikleri (Bölüm boyunca sabit)
        self.area = random.uniform(10.0, 50.0)         
        self.height = 2.8                              
        self.u_value = random.uniform(0.4, 0.9)        
        self.thermal_mass = (self.area * self.height) * 1.2 * 1005 

        # Günlük hava durumu başlangıcı
        self.outside_temp = random.uniform(-10.0, 35.0) 
        self.real_temp = self.outside_temp # Gece saatlerinde içarısı dışarısı ile aynıdır
        
        self.num_humans = 0 # Gece ofis boş

        return self._get_state()

    def _get_state(self):
        """PyTorch için float32 formatında numpy dizisi döndürür."""
        return np.array([
            self.current_step, 
            self.outside_temp, 
            self.real_temp, 
            self.u_value, 
            self.num_humans
        ], dtype=np.float32)

    def _update_schedule(self):
        """Mesai saatlerine göre ofisteki insan sayısını günceller."""
        # 08:00(480) - 12:00(720) ve 13:00(780) - 17:00(1020) arası mesai
        is_working_hours = (480 <= self.current_step < 720) or (780 <= self.current_step < 1020)
        
        if is_working_hours:
            if self.num_humans == 0:
                self.num_humans = random.randint(5, 15) # Mesai başladı, ofis doldu
        else:
            self.num_humans = 0 # Mesai dışı veya öğle molası, ofis boş

    def _update_weather(self):
        """Dış sıcaklığı gün içinde hafifçe değiştirir (basit bir dalgalanma)."""
        if random.random() < 0.1:
            self.outside_temp += random.uniform(-0.5, 0.5)

    def step(self, action):
        self.current_step += 1
        
        self._update_schedule()
        self._update_weather()

        # Inverter Mantığı (Eski koddan miras)
        p_limit = self.ac_power_limit[action]
        if self.real_temp < 24.5:      # Hedef 25°C olduğu için eşik değişti
            q_ac = p_limit
        elif self.real_temp > 25.5:    
            q_ac = -p_limit
        else:                          
            q_ac = 0

        # Termodinamik Hesaplamalar
        heat_leak = self.u_value * self.area * (self.outside_temp - self.real_temp)
        internal_heat = self.num_humans * 125 
        
        delta_t = ((heat_leak + internal_heat + q_ac) * 60) / self.thermal_mass
        self.real_temp += delta_t

        # Optimizasyon Odaklı Yeni Ödül Sistemi
        reward = self.calculate_reward(action)
        
        next_state = self._get_state()
        done = self.current_step >= self.max_steps

        return next_state, reward, done

    def calculate_reward(self, action):
        reward = 0.0
        
        # Mesai (08:00-12:00 / 13:00-17:00)
        is_working_hours = (480 <= self.current_step < 720) or (780 <= self.current_step < 1020)
        
        # HAZIRLIK EVRESİ: Mesai başlamadan önceki 30 dakika (07:30-08:00 / 12:30-13:00)
        is_prep_hours = (450 <= self.current_step < 480) or (750 <= self.current_step < 780)

        if is_working_hours:
            # Mesai saatinde tek hedef 25°C olmaktır
            temp_diff = abs(25.0 - self.real_temp)
            if temp_diff <= 1.0:
                reward += 20.0 # Kusursuz konfor
            else:
                # KAUP KURALI: Üstel (Karesel) Ceza! 
                # 1 derece saparsa -2, 3 derece saparsa -18 puan ceza yer. AI mecbur soğutacak.
                reward -= (temp_diff ** 2) * 2.0 
        else:
            if is_prep_hours:
                # Ajan mesaiye hazırlık için klimayı açarsa mesai dışı cezası YEMEZ! (Öngörü Teşviki)
                pass 
            else:
                # Gerçekten boş saatlerde (gece 3 vs.) açarsa acımasız ceza
                if action > 0:
                    reward -= 10.0 

        # Genel Enerji Cezası
        if action == 1: reward -= 1.0
        if action == 2: reward -= 3.0

        return reward