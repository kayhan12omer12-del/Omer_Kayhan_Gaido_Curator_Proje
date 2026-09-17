import os
import random
import numpy as np
import pandas as pd

GERCEK_MEKAN_HAVUZU = [
    ("Karakoy", "Karabatak Karakoy", "kafe", 2, 4.6, "sessiz kahve tatli calisma", 41.0258, 28.9795),
    ("Karakoy", "Coffee Sapiens", "kafe", 2, 4.7, "kahve sessiz calisma nitelikli", 41.0252, 28.9782),
    ("Beyoglu", "Mandabatmaz", "kafe", 1, 4.8, "turk_kahvesi tarihi nostaljik otantik", 41.0322, 28.9765),
    ("Kadikoy", "Montag Coffee Roasters", "kafe", 2, 4.5, "kahve calisma modern ferah", 40.9898, 29.0265),
    ("Kadikoy", "Story Coffee Roasters", "kafe", 2, 4.6, "sessiz kahve kahvalti calisma", 40.9855, 29.0278),
    ("Moda", "Cikolataci Ali Usta", "kafe", 2, 4.7, "tatli cikolata nostaljik samimi", 40.9832, 29.0295),
    ("Nisantasi", "Petra Roasting Co.", "kafe", 3, 4.6, "kahve modern sanat brunch", 41.0535, 28.9950),
    ("Bebek", "Assk Kahve", "kafe", 4, 4.5, "deniz_manzarasi luks romantik kahvalti", 41.0772, 29.0435),
    ("Besiktas", "Minoa Books & Coffee", "kafe", 2, 4.8, "kitap sessiz calisma kahve", 41.0425, 29.0040),
    ("Uskudar", "Cinaralti Cay Bahcesi", "kafe", 1, 4.6, "deniz_manzarasi cay tarihi nostaljik", 41.0520, 29.0525),
    ("Kadikoy", "Ciya Sofrasi", "restoran", 2, 4.8, "anadolu otantik yoresel aile", 40.9892, 29.0245),
    ("Kadikoy", "Basta Street Food Bar", "restoran", 2, 4.7, "durum gurme hizli lezzetli", 40.9875, 29.0280),
    ("Karakoy", "Karakoy Lokantasi", "restoran", 3, 4.7, "meze tarihi raki nostaljik", 41.0245, 28.9775),
    ("Beyoglu", "Mikla Restaurant", "restoran", 4, 4.9, "manzara luks gurme fine_dining", 41.0305, 28.9745),
    ("Sultanahmet", "Tarihi Sultanahmet Koftecisi", "restoran", 2, 4.6, "kofte tarihi geleneksel aile", 41.0082, 28.9772),
    ("Besiktas", "Balkan Lokantasi", "restoran", 1, 4.4, "ev_yemegi ucuz hizli doyurucu", 41.0430, 29.0060),
    ("Nisantasi", "Spago Istanbul", "restoran", 4, 4.7, "teras luks romantik manzara", 41.0505, 28.9920),
    ("Bebek", "Lucca Style", "restoran", 4, 4.5, "kokteyl luks populer gece_hayati", 41.0785, 29.0425),
    ("Ortakoy", "Tarihi Ortakoy Kumpircisi", "restoran", 2, 4.5, "kumpir deniz sokak_lezzeti hizli", 41.0475, 29.0270),
    ("Uskudar", "Kanaat Lokantasi", "restoran", 2, 4.6, "osmanli zeytinyagli tatli tarihi", 41.0265, 29.0160),
    ("Beyoglu", "Galata Kulesi", "gezi", 3, 4.8, "tarihi manzara fotograf panoramik", 41.0256, 28.9741),
    ("Beyoglu", "Istiklal Caddesi", "gezi", 1, 4.5, "alisveris yuruyus nostaljik tramvay", 41.0340, 28.9780),
    ("Sultanahmet", "Sultanahmet Meydani", "gezi", 1, 4.7, "tarihi meydan turist fotograf", 41.0065, 28.9760),
    ("Kadikoy", "Moda Sahil Parki", "gezi", 1, 4.8, "deniz gunbatimi yuruyus cimler", 40.9810, 29.0275),
    ("Bebek", "Bebek Parki ve Sahili", "gezi", 1, 4.7, "bogaz yuruyus yesillik ferah", 41.0760, 29.0420),
    ("Ortakoy", "Ortakoy Meydani ve Iskele", "gezi", 1, 4.7, "bogaz kopru fotograf turist", 41.0470, 29.0275),
    ("Uskudar", "Salacak Sahili ve Kiz Kulesi", "gezi", 1, 4.8, "deniz_manzarasi gunbatimi ikonik romantik", 41.0210, 29.0040),
    ("Besiktas", "Dolmabahce Sahil Yolu", "gezi", 1, 4.6, "tarihi bogaz agacli yuruyus", 41.0390, 28.9980),
    ("Karakoy", "Istanbul Modern Sanat Muzesi", "kultur", 3, 4.8, "sanat sergi cagdas modern mimari", 41.0260, 28.9830),
    ("Beyoglu", "Pera Muzesi", "kultur", 2, 4.7, "sanat sergi oryantalist tarihi", 41.0315, 28.9750),
    ("Beyoglu", "Salt Galata", "kultur", 1, 4.9, "kutuphane sessiz mimari calisma tarihi", 41.0242, 28.9738),
    ("Sultanahmet", "Ayasofya-i Kebir Camii", "kultur", 1, 4.9, "tarihi mimari ikonik dunya_mirasi", 41.0086, 28.9802),
    ("Sultanahmet", "Yerebatan Sarnici", "kultur", 3, 4.8, "tarihi gizemli atmosferik sutunlar", 41.0084, 28.9778),
    ("Sultanahmet", "Topkapi Sarayi Muzesi", "kultur", 4, 4.8, "osmanli tarihi saray muazzam", 41.0115, 28.9833),
    ("Besiktas", "Deniz Muzesi", "kultur", 2, 4.6, "denizcilik tarihi saltanat_kayiklari", 41.0415, 29.0055),
    ("Kadikoy", "Sureyya Operasi", "kultur", 2, 4.7, "sanat opera bale tarihi mimari", 40.9885, 29.0298),
    ("Besiktas", "Yildiz Parki", "doga", 1, 4.7, "koru yesillik gol yuruyus sessiz", 41.0490, 29.0150),
    ("Sultanahmet", "Gulhane Parki", "doga", 1, 4.7, "tarihi agaclar lale bahce golge", 41.0130, 28.9815),
    ("Bebek", "Emirgan Korusu", "doga", 1, 4.8, "lale bogaz_manzarasi kosu agaclar", 41.1080, 29.0530),
    ("Uskudar", "Fethi Pasa Korusu", "doga", 1, 4.6, "bogaz manzarasi yesillik cay sessiz", 41.0325, 29.0280),
    ("Macka", "Macka Demokrasi Parki", "doga", 1, 4.6, "kopek cimler spor teleferik", 41.0440, 28.9950),
]

TUTARLI_ISIM_KALIPLARI = {
    "kafe": ["Roastery", "Coffee Co.", "Kahvecisi", "Bakery & Coffee", "Espresso Bar", "Atolyesi", "Kitap Kafe"],
    "restoran": ["Lokantasi", "Meyhanesi", "Mutfagi", "Kebap & Izgara", "Trattoria", "Sofrasi", "Bistro"],
    "gezi": ["Meydani", "Sahil Yolu", "Seyir Tepesi", "Caddesi", "Pasaji", "Iskelesi"],
    "kultur": ["Sanat Galerisi", "Kultur Merkezi", "Atolyesi", "Muzesi", "Kutuphanesi"],
    "doga": ["Korusu", "Parki", "Botanık Bahcesi", "Millet Bahcesi", "Meselik Alani"]
}

random.seed(42)
np.random.seed(42)

mekanlar = []
id_sayac = 1

for semt, ad, kat, fiyat, puan, etiket, lat, lon in GERCEK_MEKAN_HAVUZU:
    mekanlar.append({
        "id": id_sayac,
        "ad": ad,
        "sehir": "Istanbul",
        "kategori": kat,
        "fiyat_seviyesi": fiyat,
        "puan": puan,
        "etiketler": etiket,
        "enlem": round(lat, 6),
        "boylam": round(lon, 6)
    })
    id_sayac += 1

semt_ref = {
    "Kadikoy": (40.9890, 29.0250), "Moda": (40.9830, 29.0290), "Besiktas": (41.0430, 29.0060),
    "Karakoy": (41.0250, 28.9780), "Beyoglu": (41.0320, 28.9760), "Sultanahmet": (41.0080, 28.9770),
    "Nisantasi": (41.0520, 28.9930), "Uskudar": (41.0260, 29.0170), "Bebek": (41.0770, 29.0420)
}

etiket_havuzu = ["sessiz", "kahve", "tatli", "calisma", "deniz_manzarasi", "tarihi", "aile", "yesillik", "sanat", "romantik", "gurme", "kitap"]

while len(mekanlar) < 300:
    semt = random.choice(list(semt_ref.keys()))
    kat = random.choice(["kafe", "restoran", "gezi", "kultur", "doga"])
    kalip = random.choice(TUTARLI_ISIM_KALIPLARI[kat])
    ad = f"{semt} {kalip}"
    
    mevcut_isimler = [m["ad"] for m in mekanlar]
    if ad in mevcut_isimler:
        ad = f"{semt} {kalip} ({len([x for x in mevcut_isimler if semt in x]) + 1})"
        
    base_lat, base_lon = semt_ref[semt]
    lat = base_lat + random.uniform(-0.0035, 0.0035)
    lon = base_lon + random.uniform(-0.0035, 0.0035)
    
    secilen_etiketler = " ".join(random.sample(etiket_havuzu, k=random.randint(3, 5)))
    fiyat = random.choices([1, 2, 3, 4], weights=[0.25, 0.40, 0.25, 0.10])[0]
    puan = round(random.uniform(3.8, 4.9), 1)
    
    mekanlar.append({
        "id": id_sayac,
        "ad": ad,
        "sehir": "Istanbul",
        "kategori": kat,
        "fiyat_seviyesi": fiyat,
        "puan": puan,
        "etiketler": secilen_etiketler,
        "enlem": round(lat, 6),
        "boylam": round(lon, 6)
    })
    id_sayac += 1

df_final = pd.DataFrame(mekanlar)
df_final.to_csv("places.csv", index=False, encoding="utf-8")
df_final.to_csv("places_cleaned.csv", index=False, encoding="utf-8")
print(f"--> [BAŞARILI] {len(df_final)} mekan 'places.csv' ve 'places_cleaned.csv' olarak kaydedildi.")