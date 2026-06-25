# DQN Otonom Enerji Tasarrufu

Türkçe: Derin Q-Öğrenme (DQN) tabanlı akıllı termostat simülasyonu.

## Proje Özeti
Bu proje, ofis HVAC (klima) kontrolünü enerji tasarrufu ve konfor dengesinde optimize etmek için bir ortam (env.py) ve DQN ajanı (agent.py) kullanır. Eğitim, değerlendirme ve görselleştirme araçları mevcuttur.

## İçerik
- env.py — Ortam (SmartThermostatDRLEnv): durum, adımlar, ödül fonksiyonu.
- agent.py — DQN model, replay buffer ve eğitim mantığı.
- train.py — Ajanı eğitir; çıktı: `dqn_model.pth`, `reward_history.npy`.
- evaluate.py — Eğitilmiş modeli yükleyip 1 günlük simülasyon koşar, çıktı: `dqn_test_sonuclari.png`.
- plot_training.py — `reward_history.npy` dosyasından eğitim grafiği üretir: `dqn_egitim_grafigi.png`.
- gif_thermostat.py — Simülasyon görsellerinden dashboard GIF oluşturur: `smart_thermostat_dashboard.gif`.
- dqn_model.pth, reward_history.npy, .png/.gif dosyaları — örnek çıktılar.

## Gereksinimler
- Python 3.8+ (kullanılan interpreter: Python 3.8)
- PyTorch >= 2.0.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0

Kurulum:

```powershell
python -m pip install -r requirements.txt
```

(Not: GPU kullanımı opsiyoneldir; torch GPU build yüklüyse eğitim GPU'da çalışır.)

## Hızlı Başlangıç
1. Eğitmek:
   ```powershell
   python train.py
   ```
   Çıktılar: `dqn_model.pth`, `reward_history.npy`.

2. Eğitim grafiğini çizmek:
   ```powershell
   python plot_training.py
   ```
   Çıktı: `dqn_egitim_grafigi.png`.

3. Eğitilmiş modeli test etmek ve günlük simülasyon görselleştirmesini almak:
   ```powershell
   python evaluate.py
   ```
   Çıktı: `dqn_test_sonuclari.png`.

4. Dashboard GIF üretmek:
   ```powershell
   python gif_thermostat.py
   ```
   Çıktı: `smart_thermostat_dashboard.gif` (veya klasörde ilgili görseller).

## 📊 Görsel Çıktılar ve Performans Analizi
Ajan, 500 günlük simülasyon boyunca aldığı cezalarla kendi tecrübesini (`reward_history.npy`) oluşturmuş ve final testinde **6768.66** gibi rekor bir test skoruna ulaşmıştır.

**📈 Öğrenme Eğrisi (Dipten Zirveye Tırmanış):**
![Eğitim Eğrisi](dqn_egitim_grafigi.png)

**🎯 1 Günlük Test Simülasyonu (25°C Hedefine Oturma):**
![Test Sonuçları](dqn_test_sonuclari.png)

**🖥️ Arayüz ve Karar Simülasyonu:**
![Dashboard](smart_thermostat_dashboard.gif)

### ⚙️ DQN Eğitim Hiperparametreleri
Projenin temel sinir ağı mimarisi aşağıdaki değerlerle optimize edilmiştir:

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| **Toplam Bölüm (Episodes)** | 500 | Ajanın ofisi yönetmeyi denediği toplam gün sayısı |
| **Batch Size** | 64 | Replay Buffer'dan tek seferde çekilen tecrübe sayısı |
| **Target Update Freq** | 10 | Hedef ağın (Target Network) güncellenme periyodu |
| **Başlangıç Keşif Oranı (epsilon)** | ~1.00 | Eğitimin başındaki rastgele karar (keşif) oranı |
| **Bitiş Keşif Oranı (Min epsilon)** | ~0.08 | Eğitim tamamlandığındaki minimum rastgelelik oranı |

## Örnek Terminal Çıktıları
Aşağıda eğitim, grafik, değerlendirme ve GIF üretim süreçlerine ait tam terminal çıktısı gösterilmiştir:

```
PS C:\Users\Kadir\Desktop\DQN Otonom Enerji Tasarrufu> python train
C:\Users\Kadir\AppData\Local\Programs\Python\Python38\python.exe: can't open file 'train': [Errno 2] No such file or directory
PS C:\Users\Kadir\Desktop\DQN Otonom Enerji Tasarrufu> python train.py
DQN Eğitim Süreci Başlıyor...
--------------------------------------------------
Bölüm: 10/500 | Ortalama Ödül: -5888.73 | Keşif Oranı (Epsilon): 0.951
Bölüm: 20/500 | Ortalama Ödül: -10920.12 | Keşif Oranı (Epsilon): 0.905
Bölüm: 30/500 | Ortalama Ödül: -11175.32 | Keşif Oranı (Epsilon): 0.860
Bölüm: 40/500 | Ortalama Ödül: -8315.99 | Keşif Oranı (Epsilon): 0.818
Bölüm: 50/500 | Ortalama Ödül: -3744.04 | Keşif Oranı (Epsilon): 0.778
Bölüm: 60/500 | Ortalama Ödül: -4741.26 | Keşif Oranı (Epsilon): 0.740
Bölüm: 70/500 | Ortalama Ödül: -8379.16 | Keşif Oranı (Epsilon): 0.704
Bölüm: 80/500 | Ortalama Ödül: -4790.38 | Keşif Oranı (Epsilon): 0.670
Bölüm: 90/500 | Ortalama Ödül: -5587.80 | Keşif Oranı (Epsilon): 0.637
Bölüm: 100/500 | Ortalama Ödül: -1573.48 | Keşif Oranı (Epsilon): 0.606
Bölüm: 110/500 | Ortalama Ödül: -2240.06 | Keşif Oranı (Epsilon): 0.576
Bölüm: 120/500 | Ortalama Ödül: -3166.04 | Keşif Oranı (Epsilon): 0.548
Bölüm: 130/500 | Ortalama Ödül: -2476.69 | Keşif Oranı (Epsilon): 0.521
Bölüm: 140/500 | Ortalama Ödül: -664.70 | Keşif Oranı (Epsilon): 0.496
Bölüm: 150/500 | Ortalama Ödül: 1002.26 | Keşif Oranı (Epsilon): 0.471
Bölüm: 160/500 | Ortalama Ödül: -340.78 | Keşif Oranı (Epsilon): 0.448
Bölüm: 170/500 | Ortalama Ödül: 1272.10 | Keşif Oranı (Epsilon): 0.427
Bölüm: 180/500 | Ortalama Ödül: -624.91 | Keşif Oranı (Epsilon): 0.406
Bölüm: 190/500 | Ortalama Ödül: 1189.35 | Keşif Oranı (Epsilon): 0.386
Bölüm: 200/500 | Ortalama Ödül: -2209.84 | Keşif Oranı (Epsilon): 0.367
Bölüm: 210/500 | Ortalama Ödül: -411.84 | Keşif Oranı (Epsilon): 0.349
Bölüm: 220/500 | Ortalama Ödül: 835.27 | Keşif Oranı (Epsilon): 0.332
Bölüm: 230/500 | Ortalama Ödül: -666.50 | Keşif Oranı (Epsilon): 0.316
Bölüm: 240/500 | Ortalama Ödül: 2347.15 | Keşif Oranı (Epsilon): 0.300
Bölüm: 250/500 | Ortalama Ödül: 3358.41 | Keşif Oranı (Epsilon): 0.286
Bölüm: 260/500 | Ortalama Ödül: 633.83 | Keşif Oranı (Epsilon): 0.272
Bölüm: 270/500 | Ortalama Ödül: 1182.95 | Keşif Oranı (Epsilon): 0.258
Bölüm: 280/500 | Ortalama Ödül: 2340.91 | Keşif Oranı (Epsilon): 0.246
Bölüm: 290/500 | Ortalama Ödül: 1139.60 | Keşif Oranı (Epsilon): 0.234
Bölüm: 300/500 | Ortalama Ödül: 2636.07 | Keşif Oranı (Epsilon): 0.222
Bölüm: 310/500 | Ortalama Ödül: 1281.89 | Keşif Oranı (Epsilon): 0.211
Bölüm: 320/500 | Ortalama Ödül: 1924.97 | Keşif Oranı (Epsilon): 0.201
Bölüm: 330/500 | Ortalama Ödül: 1824.47 | Keşif Oranı (Epsilon): 0.191
Bölüm: 340/500 | Ortalama Ödül: 2032.08 | Keşif Oranı (Epsilon): 0.182
Bölüm: 350/500 | Ortalama Ödül: 94.10 | Keşif Oranı (Epsilon): 0.173
Bölüm: 360/500 | Ortalama Ödül: 1609.67 | Keşif Oranı (Epsilon): 0.165
Bölüm: 370/500 | Ortalama Ödül: 322.93 | Keşif Oranı (Epsilon): 0.157
Bölüm: 380/500 | Ortalama Ödül: 1155.04 | Keşif Oranı (Epsilon): 0.149
Bölüm: 390/500 | Ortalama Ödül: 2899.13 | Keşif Oranı (Epsilon): 0.142
Bölüm: 400/500 | Ortalama Ödül: 3532.28 | Keşif Oranı (Epsilon): 0.135
Bölüm: 410/500 | Ortalama Ödül: 2857.76 | Keşif Oranı (Epsilon): 0.128
Bölüm: 420/500 | Ortalama Ödül: 1332.76 | Keşif Oranı (Epsilon): 0.122
Bölüm: 430/500 | Ortalama Ödül: 3254.52 | Keşif Oranı (Epsilon): 0.116
Bölüm: 440/500 | Ortalama Ödül: 704.17 | Keşif Oranı (Epsilon): 0.110
Bölüm: 450/500 | Ortalama Ödül: 3371.76 | Keşif Oranı (Epsilon): 0.105
Bölüm: 460/500 | Ortalama Ödül: 3943.74 | Keşif Oranı (Epsilon): 0.100
Bölüm: 470/500 | Ortalama Ödül: 3592.33 | Keşif Oranı (Epsilon): 0.095
Bölüm: 480/500 | Ortalama Ödül: 3888.69 | Keşif Oranı (Epsilon): 0.090
Bölüm: 490/500 | Ortalama Ödül: 4896.30 | Keşif Oranı (Epsilon): 0.086
Bölüm: 500/500 | Ortalama Ödül: 3867.91 | Keşif Oranı (Epsilon): 0.082
--------------------------------------------------
Eğitim Tamamlandı! Model 'dqn_model.pth' olarak kaydediliyor...
Eğitim geçmişi 'reward_history.npy' olarak başarıyla kaydedildi!
Kaydedildi. Artık test aşamasına ve grafik çizimine geçebiliriz!
PS C:\Users\Kadir\Desktop\DQN Otonom Enerji Tasarrufu> python plot_training.py
Organik eğitim verileri okunuyor...
Şov Zamanı! Profesyonel eğitim grafiği 'dqn_egitim_grafigi.png' adıyla kaydedildi.
PS C:\Users\Kadir\Desktop\DQN Otonom Enerji Tasarrufu> python evaluate.py
Test simülasyonu başlatılıyor...
Eğitilmiş model başarıyla yüklendi!
Test Günü Toplam Ödülü: 6768.66
PS C:\Users\Kadir\Desktop\DQN Otonom Enerji Tasarrufu> python gif_thermostat.py
Dashboard GIF Simülasyonu başlatılıyor (DQN Entegreli)...
Model yüklendi. Görseller işleniyor...
İşlem tamam! 'smart_thermostat_dashboard.gif' klasörüne kaydedildi.
PS C:\Users\Kadir\Desktop\DQN Otonom Enerji Tasarrufu> 
```

### 📂 Proje Dosya Ağacı ve Hiyerarşisi
```text
DQN_Otonom_Enerji_Tasarrufu/
│
├── env.py                  # Termodinamik fizik motoru ve ofis simülasyonu
├── agent.py                # PyTorch tabanlı DQN Sinir Ağı ve Replay Buffer
├── train.py                # Eğitim döngüsü (500 episode)
├── evaluate.py             # Eğitilmiş model ile 1 günlük performans testi
├── plot_training.py        # Organik eğitim geçmişinden grafik oluşturucu
├── gif_thermostat.py       # Arayüz ve Time-lapse Dashboard simülasyonu
│
├── dqn_model.pth           # Eğitilmiş ajanın ağırlıkları (Yapay Zeka Beyni)
├── reward_history.npy      # Eğitim boyunca alınan organik puan geçmişi
├── requirements.txt        # Gerekli Python kütüphaneleri
└── README.md               # Proje dokümantasyonu
```

//////////////////////////////////////////





1. Markov Karar Süreci (MDP) ve Karesel Ceza OptimizasyonuProblem, ofis içi iklimlendirme dinamiğinin bir Markov Karar Süreci (MDP) $\langle S, A, P, R, \gamma \rangle$ olarak modellenmesiyle çözülmüştür.Durum Uzayı (State Space - $S$): Çevre, sürekli değişkenlerden (continuous variables) oluşan 5 boyutlu bir durum vektörü $s_t \in \mathbb{R}^5$ ile temsil edilir:$$s_t = [\tau, T_{ext}, T_{int}, U, N_{insan}]$$(Burada $\tau$ günün dakikasını, $T$ sıcaklık değerlerini, $U$ yalıtım katsayısını ve $N$ ısı yükü oluşturan insan sayısını ifade eder.)Eylem Uzayı (Action Space - $A$): Ajanın alabileceği kararlar, klima motorunun çekeceği elektrik gücü $P(a_t)$ ile ayrık (discrete) bir uzayda tanımlanmıştır:$$a_t \in \{0, 1, 2\} \implies P(a_t) \in \{0\text{W}, 2500\text{W}, 5000\text{W}\}$$Ödül Fonksiyonu ve Karesel Ceza (Reward Function - $R$): Sistem, klasik Q-Learning tablolarındaki doğrusal ve sabit ödüller yerine, hedef sıcaklıktan ($T_{ref} = 25^\circ\text{C}$) sapmayı Karesel Ceza (Quadratic Penalty) ile üstel olarak cezalandıran bir fonksiyon kullanır:$$R(s_t, a_t) = - \omega_1 (T_{int} - T_{ref})^2 - \omega_2 P(a_t) + C$$(Bu denklem sayesinde ajan, $\Delta T$ büyüdüğünde $5000\text{W}$ agresif gücü kullanmayı, $T_{int} \approx T_{ref}$ olduğunda ise rölanti/kapalı duruma geçerek $\omega_2 P(a_t)$ enerji cezasını minimize etmeyi matematiksel olarak öğrenir.)2. Termodinamik Çevre Dinamikleri (Transition Model - $P$)Ortam (env.py), ajan kararlarına göre bir sonraki durumu $P(s_{t+1} | s_t, a_t)$ termodinamik kurallara uygun olarak hesaplar. $\Delta t = 60s$ (1 dakika) için ofis içindeki net sıcaklık değişimi $\Delta T_{int}$, enerji korunumu yasalarına göre modellenmiştir:$$\Delta T_{int} = \frac{\left[ \dot{Q}_{leak} + \dot{Q}_{internal} + \dot{Q}_{hvac} \right] \cdot \Delta t}{V \cdot \rho_{hava} \cdot c_{p,hava}}$$Bu denklemdeki ısı akış ($\dot{Q}$) bileşenleri şu şekilde tanımlanır:İletimle Isı Kaybı (Heat Leak): $\dot{Q}_{leak} = U \cdot A \cdot (T_{ext} - T_{int})$İç Termal Yük (Internal Load): $\dot{Q}_{internal} = N_{insan} \cdot 125\text{W}$HVAC Etkisi: $\dot{Q}_{hvac} = \pm P(a_t)$ (Isıtma için pozitif, soğutma için negatif işaretli Inverter motor gücü)3. Deep Q-Network (DQN) ve Bellman OptimizasyonuAjan, durum-eylem (state-action) değerlerini tahmin etmek için derin bir sinir ağı (Multi-Layer Perceptron) olan $Q(s, a; \theta)$ kullanır. Eğitim (Training) sürecinde amaç, Bellman Optimalite Denklemi baz alınarak hesaplanan Temporal Difference (TD) hatasını minimize etmektir.Kayıp fonksiyonu (Loss Function), Deneyim Hafızasından (Experience Replay, $\mathcal{D}$) rastgele çekilen mini-batch'ler üzerinden Mean Squared Error (MSE) ile hesaplanır:$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$(Aşırı uyumu (overfitting) ve stabilite sorunlarını önlemek amacıyla, hedef değer hesaplamalarında periyodik olarak güncellenen ayrı bir Hedef Ağ (Target Network, $\theta^-$) kullanılmıştır.)4. Akademik Vizyon ve Gelecek Çalışmalar (Future Work)Sistem mevcut mimarisiyle otonom karar alma ve enerji optimizasyonu süreçlerinde yüksek stabilite ve $6768.66$ kümülatif ödül skoru ile başarısını kanıtlamıştır. İleri araştırmalar için planlanan mimari geliştirmeler şunlardır:Durum Normalizasyonu (Min-Max / Z-Score Scaling): Sürekli durum değişkenlerinin $s_t \in \mathbb{R}^5$, sinir ağındaki ağırlık güncellemelerinin stabilitesini artırmak amacıyla $[0, 1]$ veya $[-1, 1]$ aralığına izdüşürülmesi.Double DQN (DDQN) Entegrasyonu: Klasik DQN mimarisinde karşılaşılan $Q$-değeri aşırı kestirim (overestimation) problemini çözmek amacıyla, eylem seçimi ile eylem değerlendirmesi süreçlerinin iki bağımsız ağ üzerinden ayrıştırılması:$$Y^{DDQN}_t = r_{t+1} + \gamma Q\left(s_{t+1}, \arg\max_a Q(s_{t+1}, a; \theta_t); \theta^-_t\right)$$Fault Tolerance (Durum Korunumu): Eğitim esnasında optimizer_state_dict nesnesinin checkpoint olarak diske yazılması ve kesinti durumlarında (resume training) optimizasyon momentumunun korunması.
