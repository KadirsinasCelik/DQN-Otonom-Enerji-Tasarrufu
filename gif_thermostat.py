import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import os
import random
from env import SmartThermostatDRLEnv
from agent import DQNAgent

def create_frame(step, real_temp, outside_temp, num_humans, area, action_name, reward, total_reward):
    # Görsel Boyutu ve Arka Plan
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(230, 235, 240))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 32)
        font_main = ImageFont.truetype("arial.ttf", 26)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_tiny = ImageFont.truetype("arial.ttf", 14)
        font_temp_small = ImageFont.truetype("arial.ttf", 36)
    except:
        font_title = font_main = font_sub = font_tiny = font_temp_small = ImageFont.load_default()

    # --- 1. ANA PANEL VE BAŞLIK ---
    draw.text((150, 25), "DQN Akıllı Termostat: Dashboard Simülasyonu", fill=(50, 50, 100), font=font_title)
    draw.rectangle((50, 80, 750, 450), fill=(255, 255, 255), outline=(150, 150, 150), width=2) 

    # --- 2. PENCERE VE DIŞ HAVA DURUMU ---
    w_x1, w_y1, w_x2, w_y2 = 80, 110, 260, 240
    draw.rectangle((w_x1-8, w_y1-8, w_x2+8, w_y2+8), fill=(101, 67, 33)) 
    
    if outside_temp <= 0:
        draw.rectangle((w_x1, w_y1, w_x2, w_y2), fill=(200, 230, 255))
        for _ in range(25):
            sx, sy = random.randint(w_x1, w_x2), random.randint(w_y1, w_y2)
            draw.ellipse((sx, sy, sx+3, sy+3), fill=(255, 255, 255))
        weather_desc = "Kar Yağışlı"
    elif 0 < outside_temp <= 15:
        draw.rectangle((w_x1, w_y1, w_x2, w_y2), fill=(160, 170, 180))
        for _ in range(20):
            rx, ry = random.randint(w_x1, w_x2), random.randint(w_y1, w_y2)
            draw.line((rx, ry, rx-3, ry+8), fill=(100, 100, 255), width=1)
        weather_desc = "Yağmurlu"
    else:
        draw.rectangle((w_x1, w_y1, w_x2, w_y2), fill=(135, 206, 250))
        draw.ellipse((140, 130, 200, 190), fill=(255, 220, 0), outline=(255, 150, 0), width=2)
        weather_desc = "Güneşli"

    draw.line(((w_x1+w_x2)/2, w_y1, (w_x1+w_x2)/2, w_y2), fill=(101, 67, 33), width=4)

    # --- 3. DETAYLI TAKIM ELBİSELİ İNSAN FİGÜRLERİ ---
    display_humans = min(int(num_humans), 3)
    for h in range(display_humans):
        h_x = 80 + (h * 85) 
        h_y = 310 
        draw.rectangle((h_x, h_y+25, h_x+50, h_y+85), fill=(40, 50, 80)) 
        draw.polygon([(h_x+15, h_y+25), (h_x+35, h_y+25), (h_x+25, h_y+45)], fill=(255, 255, 255))
        draw.line((h_x+25, h_y+28, h_x+25, h_y+55), fill=(200, 0, 0), width=3) 
        draw.ellipse((h_x-8, h_y+55, h_x+2, h_y+65), fill=(240, 200, 180)) 
        draw.ellipse((h_x+48, h_y+55, h_x+58, h_y+65), fill=(240, 200, 180)) 
        draw.ellipse((h_x+10, h_y-10, h_x+40, h_y+25), fill=(240, 200, 180)) 
        draw.rectangle((h_x+10, h_y-10, h_x+40, h_y+2), fill=(60, 40, 20)) 
        draw.line((h_x+17, h_y+8, h_x+22, h_y+8), fill=(0, 0, 0), width=1)
        draw.line((h_x+28, h_y+8, h_x+33, h_y+8), fill=(0, 0, 0), width=1)
        draw.point([(h_x+19, h_y+12), (h_x+31, h_y+12)], fill=(0, 0, 0))
        draw.line((h_x+22, h_y+18, h_x+28, h_y+18), fill=(150, 50, 50), width=1)
        draw.rectangle((h_x+5, h_y+85, h_x+20, h_y+125), fill=(30, 30, 30)) 
        draw.rectangle((h_x+30, h_y+85, h_x+45, h_y+125), fill=(30, 30, 30)) 
        draw.rectangle((h_x, h_y+125, h_x+20, h_y+135), fill=(10, 10, 10))
        draw.rectangle((h_x+30, h_y+125, h_x+50, h_y+135), fill=(10, 10, 10))

    if num_humans > 3:
        dots_x = 80 + (3 * 85) - 15
        for i in range(3):
            draw.ellipse((dots_x + (i*12), 315, dots_x + 6 + (i*12), 321), fill=(100, 100, 100))

    # --- 4. KLİMA (AC) VE ÜFLEME EFEKTİ ---
    ac_rect = (550, 100, 720, 145)
    is_active = "Kapali" not in action_name
    ac_color = (0, 150, 255) if is_active else (200, 200, 200)
    draw.rectangle(ac_rect, fill=(240, 240, 240), outline=(0,0,0), width=2)
    draw.text((610, 108), "AC", fill=ac_color, font=font_main)
    
    if is_active:
        # Performans modunda daha çok, bekleme modunda daha az üfleme efekti
        lines = 8 if "Performans" in action_name else 3
        for _ in range(lines):
            line_x = random.randint(ac_rect[0]+10, ac_rect[2]-10)
            line_y_start = ac_rect[3] + 2
            line_y_end = line_y_start + random.randint(15, 35)
            draw.line((line_x, line_y_start, line_x - 5, line_y_end), fill=(0, 180, 255), width=2)

    # --- 5. İÇ SICAKLIK VE DURUM MESAJI ---
    # Yeni Hedef: 25.0 °C (Kabul edilebilir aralık 24.0 - 26.0)
    if 24.0 <= real_temp <= 26.0:
        temp_color = (0, 150, 0) # Yeşil İdeal
        status_text = "MUKEMMEL (25C HEDEFI)"
    elif real_temp < 24.0:
        temp_color = (0, 0, 200) # Mavi Soğuk
        status_text = "SOGUK: AI ISITIYOR"
    else:
        temp_color = (200, 0, 0) # Kırmızı Sıcak
        status_text = "SICAK: AI SOGUTUYOR"

    draw.text((300, 102), f"{real_temp:.2f}°C", fill=temp_color, font=font_temp_small)
    draw.text((290, 145), status_text, fill=temp_color, font=font_tiny)

    # --- 6. MASALAR VE KİTAPLAR ---
    for i in range(2):
        x_m = 380 + (i * 180)
        draw.rectangle((x_m, 380, x_m+120, 395), fill=(139, 69, 19), outline=(50, 20, 0))
        draw.rectangle((x_m+10, 395, x_m+25, 440), fill=(80, 40, 0))
        draw.rectangle((x_m+95, 395, x_m+110, 440), fill=(80, 40, 0))
        book_colors = [(200, 50, 50), (50, 150, 50), (50, 50, 200)]
        for b in range(3):
            book_y = 380 - (b * 6) - 6
            draw.rectangle((x_m+35, book_y, x_m+85, book_y+6), fill=book_colors[b], outline=(0,0,0))

    # --- 7. ÜÇ KOLONLU ALT BİLGİ BARI ---
    draw.rectangle((0, 470, 800, 600), fill=(30, 40, 50)) 
    
    # Saati formata çevirme (0 - 1440 dakikayı HH:MM formatına dönüştürme)
    hours = step // 60
    minutes = step % 60
    time_str = f"{hours:02d}:{minutes:02d}"

    draw.text((30, 490), f"SAAT: {time_str} (Dk: {step})", fill=(255, 255, 255), font=font_sub)
    draw.text((30, 520), f"ALAN: {area:.1f} m2", fill=(200, 200, 200), font=font_sub)
    draw.text((30, 550), f"İNSAN: {num_humans} Kişi", fill=(200, 200, 200), font=font_sub)
    
    draw.text((280, 490), f"DIŞ HAVA: {outside_temp:.1f}°C", fill=(255, 255, 255), font=font_sub)
    draw.text((280, 515), f"DURUM: {weather_desc}", fill=(200, 200, 200), font=font_sub)
    draw.text((280, 545), "AI KARARI:", fill=(255, 255, 255), font=font_sub)
    
    action_color = (255, 50, 50) if "Performans" in action_name else ((255, 200, 0) if "Bekleme" in action_name else (150, 150, 150))
    draw.text((280, 568), f"{action_name}", fill=action_color, font=font_sub)
    
    draw.text((580, 490), f"ANLIK ÖDÜL: {reward:.1f}", fill=(0, 255, 0) if reward >= 0 else (255, 50, 50), font=font_sub)
    draw.text((580, 530), "TOPLAM ÖDÜL:", fill=(255, 255, 255), font=font_sub)
    draw.text((580, 555), f"{total_reward:.1f}", fill=(255, 215, 0), font=font_main)

    return img

def main():
    print("Dashboard GIF Simülasyonu başlatılıyor (DQN Entegreli)...")
    
    env = SmartThermostatDRLEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Modeli Yükle
    try:
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", weights_only=True))
        agent.policy_net.eval()
        print("Model yüklendi. Görseller işleniyor...")
    except FileNotFoundError:
        print("HATA: dqn_model.pth bulunamadı!")
        return

    agent.epsilon = 0.0 # Sadece sömürü (akıllı hareketler)
    state = env.reset()
    
    frames = []
    total_reward = 0
    
    for time_step in range(env.max_steps):
        # Ajan aksiyon seçer
        action = agent.act(state)
        action_name = env.action_names[action]
        temp_before = env.real_temp
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Her 15 dakikada bir kare al (Hızlandırılmış gün)
        # Ayrıca mesai başlama (480) ve bitiş (1020) saatlerini kesin yakala
        if time_step % 15 == 0 or time_step == 480 or time_step == 1020:
            frame = create_frame(time_step, temp_before, env.outside_temp, env.num_humans, env.area, action_name, reward, total_reward)
            frames.append(frame)
            
        state = next_state
        if done:
            break

    # GIF olarak kaydetme (duration=0.1 ile saniyede 10 kare hızında çok şık bir timelapse olur)
    imageio.mimsave("smart_thermostat_dashboard.gif", frames, duration=0.1)
    print("İşlem tamam! 'smart_thermostat_dashboard.gif' klasörüne kaydedildi. Şov başlasın! 🚀")

if __name__ == "__main__":
    main()