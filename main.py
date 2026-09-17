import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("       GAIDO-CURATOR: UÇTAN UCA MEKAN ÖNERİ SİSTEMİ")
print("=" * 65)

# FAZ 1 KONTROLÜ
if not os.path.exists("places_cleaned.csv"):
    print("--> 'places_cleaned.csv' bulunamadı, 'generate_real_places.py' çalıştırılıyor...")
    subprocess.run(["python", "generate_real_places.py"])

df = pd.read_csv("places_cleaned.csv")
print(f"\n[FAZ 1] Toplam {len(df)} mekan başarıyla yüklendi.")

# ==========================================================
# FAZ 2: KEŞİFÇİ VERİ ANALİZİ (EDA) VE HARİTALANDIRMA
# ==========================================================
print("\n[FAZ 2] EDA Grafikleri ve Harita üretiliyor...")

# EDA 1: Kategori Dağılımı
plt.figure(figsize=(7, 4))
df["kategori"].value_counts().plot(kind="bar", color="#2b5c8f", edgecolor="black")
plt.title("Kategori Dağılımı")
plt.xlabel("Kategori")
plt.ylabel("Mekan Sayısı")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_kategori_dagilimi.png"), dpi=300)
plt.close()

# EDA 2: Puan Dağılımı
plt.figure(figsize=(7, 4))
df["puan"].plot(kind="hist", bins=10, color="#27ae60", edgecolor="black")
plt.title("Puan Dağılımı")
plt.xlabel("Puan (1-5)")
plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_puan_dagilimi.png"), dpi=300)
plt.close()

# EDA 3: Fiyat Seviyesi vs Puan
plt.figure(figsize=(7, 4))
plt.scatter(df["fiyat_seviyesi"], df["puan"], color="#e74c3c", alpha=0.6, s=50)
plt.title("Fiyat Seviyesi vs Puan")
plt.xlabel("Fiyat Seviyesi (1-4)")
plt.ylabel("Puan")
plt.xticks([1, 2, 3, 4])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_fiyat_puan.png"), dpi=300)
plt.close()

# Statik Folium Haritası
harita = folium.Map(location=[41.0375, 28.9860], zoom_start=12)
kategori_renkleri = {"kafe": "orange", "restoran": "red", "gezi": "blue", "doga": "green", "kultur": "purple"}
for _, r in df.iterrows():
    folium.Marker(
        location=[r["enlem"], r["boylam"]],
        popup=f"<b>{r['ad']}</b> ({r['kategori']})",
        tooltip=r["ad"],
        icon=folium.Icon(color=kategori_renkleri.get(r["kategori"], "gray"), icon="info-sign")
    ).add_to(harita)
harita.save("harita.html")
print("--> 3 adet EDA grafiği 'outputs/' dizinine ve 'harita.html' kaydedildi.")

# ==========================================================
# FAZ 3: TF-IDF VE KOSİNÜS BENZERLİK MODELİ
# ==========================================================
print("\n[FAZ 3] TF-IDF Matrisi ve Kosinüs Benzerliği hesaplanıyor...")

df["metadata"] = df["kategori"].astype(str) + " " + df["etiketler"].astype(str)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["metadata"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"--> TF-IDF Matris Boyutu: {tfidf_matrix.shape}")

ref_id = 1
secilen = df[df["id"] == ref_id].iloc[0]
sim_scores = cosine_sim[0]

df_faz3 = df.copy()
df_faz3["benzerlik"] = np.round(sim_scores, 3)
df_faz3["bilesik_skor"] = np.round((sim_scores * 0.7) + ((df["puan"] / 5.0) * 0.3), 3)
faz3_oneriler = df_faz3[df_faz3["id"] != ref_id].sort_values(by="bilesik_skor", ascending=False).head(5)
faz3_oneriler.to_csv(os.path.join(OUTPUT_DIR, "faz3_benzerlik_onerileri.csv"), index=False, encoding="utf-8")

# ==========================================================
# FAZ 4: KURAL TABANLI FİLTRELEME, MESAFE VE SOĞUK BAŞLANGIÇ
# ==========================================================
print("\n[FAZ 4] Filtreleme, Jaccard benzerliği ve Soğuk Başlangıç motoru çalıştırılıyor...")

def oklid_mesafe_km(lat1, lon1, lat2, lon2):
    return np.sqrt(((lat1 - lat2) * 111.0)**2 + ((lon1 - lon2) * 85.0)**2)

def jaccard_benzerligi(kume_a, kume_b):
    if not kume_a or not kume_b: return 0.0
    return len(kume_a.intersection(kume_b)) / len(kume_a.union(kume_b))

def akilli_mekan_oner(
    referans_mekan_id=None,
    kullanici_konum=(41.0375, 28.9860),
    secilen_kategoriler=None,
    maks_fiyat=None,
    min_puan=0.0,
    maks_mesafe_km=None,
    arama_etiketleri=None,
    top_n=5
):
    filtrelenmis_df = df.copy()
    u_lat, u_lon = kullanici_konum
    filtrelenmis_df["mesafe_km"] = oklid_mesafe_km(
        filtrelenmis_df["enlem"], filtrelenmis_df["boylam"], u_lat, u_lon
    ).round(2)
    
    if secilen_kategoriler:
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["kategori"].isin(secilen_kategoriler)]
    if maks_fiyat is not None:
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["fiyat_seviyesi"] <= maks_fiyat]
    if min_puan > 0:
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["puan"] >= min_puan]
    if maks_mesafe_km is not None:
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["mesafe_km"] <= maks_mesafe_km]
    
    if filtrelenmis_df.empty:
        filtrelenmis_df = df.copy()
        filtrelenmis_df["mesafe_km"] = oklid_mesafe_km(
            filtrelenmis_df["enlem"], filtrelenmis_df["boylam"], u_lat, u_lon
        ).round(2)
    
    # Soğuk Başlangıç
    if referans_mekan_id is None and not arama_etiketleri:
        norm_mesafe = 1 - (filtrelenmis_df["mesafe_km"] / (filtrelenmis_df["mesafe_km"].max() + 1e-5))
        filtrelenmis_df["benzerlik"] = 0.0
        filtrelenmis_df["bilesik_skor"] = np.round((filtrelenmis_df["puan"] / 5.0) * 0.7 + norm_mesafe * 0.3, 3)
        return filtrelenmis_df.sort_values(by="bilesik_skor", ascending=False).head(top_n)
    
    if referans_mekan_id is not None:
        idx = df[df["id"] == referans_mekan_id].index[0]
        filtrelenmis_df["benzerlik"] = [np.round(cosine_sim[idx][i], 3) for i in filtrelenmis_df.index]
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["id"] != referans_mekan_id]
    elif arama_etiketleri:
        kullanici_kume = set(arama_etiketleri)
        filtrelenmis_df["benzerlik"] = filtrelenmis_df["etiketler"].apply(
            lambda x: np.round(jaccard_benzerligi(kullanici_kume, set(str(x).split())), 3)
        )
    
    max_dist = filtrelenmis_df["mesafe_km"].max()
    min_dist = filtrelenmis_df["mesafe_km"].min()
    norm_mesafe = (filtrelenmis_df["mesafe_km"] - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0
        
    filtrelenmis_df["bilesik_skor"] = np.round(
        (filtrelenmis_df["benzerlik"] * 0.5) +
        ((filtrelenmis_df["puan"] / 5.0) * 0.3) +
        ((1 - norm_mesafe) * 0.2), 3
    )
    return filtrelenmis_df.sort_values(by="bilesik_skor", ascending=False).head(top_n)

# Faz 4 Test 1: Kural Tabanlı Filtreleme
faz4_filtre = akilli_mekan_oner(secilen_kategoriler=["kafe"], maks_fiyat=2, min_puan=4.0, arama_etiketleri=["sessiz", "kahve"], top_n=5)
faz4_filtre.to_csv(os.path.join(OUTPUT_DIR, "faz4_kural_tabanli_oneriler.csv"), index=False, encoding="utf-8")

# Faz 4 Test 2: Soğuk Başlangıç
faz4_cold = akilli_mekan_oner(top_n=5)
faz4_cold.to_csv(os.path.join(OUTPUT_DIR, "faz4_soguk_baslangic_onerileri.csv"), index=False, encoding="utf-8")

# Özet Test Raporu
with open(os.path.join(OUTPUT_DIR, "test_raporu.txt"), "w", encoding="utf-8") as f:
    f.write("=== GAIDO-CURATOR UÇTAN UCA MODEL ÇIKTI RAPORU ===\n\n")
    f.write(f"1. FAZ 3 REFERANS MEKAN: {secilen['ad']} ({secilen['kategori']})\n")
    f.write(faz3_oneriler[["id", "ad", "kategori", "puan", "fiyat_seviyesi", "benzerlik", "bilesik_skor"]].to_string(index=False))
    f.write("\n\n" + "="*60 + "\n\n")
    f.write("2. FAZ 4 KURAL TABANLI FİLTRELEME (Kafe, Bütçe<=2, Puan>=4.0):\n")
    f.write(faz4_filtre[["id", "ad", "kategori", "puan", "fiyat_seviyesi", "mesafe_km", "bilesik_skor"]].to_string(index=False))
    f.write("\n\n" + "="*60 + "\n\n")
    f.write("3. FAZ 4 SOĞUK BAŞLANGIÇ (Taksim Merkezli Popüler Mekanlar):\n")
    f.write(faz4_cold[["id", "ad", "kategori", "puan", "fiyat_seviyesi", "mesafe_km", "bilesik_skor"]].to_string(index=False))

print("--> [TAMAMLANDI] Faz 1, 2, 3 ve 4 çıktıları 'outputs/' klasörüne başarıyla kaydedildi.")