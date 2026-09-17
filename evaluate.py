import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. VERİYİ YÜKLE VE TF-IDF MODELİNİ OLUŞTUR
df = pd.read_csv("places_cleaned.csv")
df["metadata"] = df["kategori"] + " " + df["etiketler"]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["metadata"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 2. YARDIMCI METRİK FONKSİYONLARI
def jaccard_benzerligi(kume_a, kume_b):
    if not kume_a or not kume_b:
        return 0.0
    kesisim = len(kume_a.intersection(kume_b))
    birlesim = len(kume_a.union(kume_b))
    return kesisim / birlesim if birlesim > 0 else 0.0

def precision_at_k(onerilen_etiketler_listesi, hedef_etiketler, k=5):
    """İlk K öneri içinde hedef etiketlerle eşleşenlerin oranını ölçer."""
    hedef_kume = set(hedef_etiketler)
    isabetler = 0
    for etiketler in onerilen_etiketler_listesi[:k]:
        mekan_kume = set(etiketler.split())
        if len(hedef_kume.intersection(mekan_kume)) > 0:
            isabetler += 1
    return isabetler / k

def hit_rate_at_k(onerilen_etiketler_listesi, hedef_etiketler, k=5):
    """İlk K öneri içinde en az 1 doğru eşleşme varsa 1, yoksa 0 döndürür."""
    return 1.0 if precision_at_k(onerilen_etiketler_listesi, hedef_etiketler, k) > 0 else 0.0

# 3. 10 SENTETİK KULLANICI TEST SENARYOSU
test_senaryolari = [
    {"id": 1, "kullanici": "Öğrenci / Çalışma", "kategori": ["kafe"], "etiketler": ["sessiz", "calisma", "kahve"], "maks_fiyat": 2},
    {"id": 2, "kullanici": "Romantik Akşam", "kategori": ["restoran"], "etiketler": ["romantik", "deniz_manzarasi", "luks"], "maks_fiyat": 4},
    {"id": 3, "kullanici": "Kültür / Sanat Turisti", "kategori": ["kultur"], "etiketler": ["tarihi", "sanat", "muze"], "maks_fiyat": 3},
    {"id": 4, "kullanici": "Doğa / Yürüyüş", "kategori": ["doga"], "etiketler": ["yesillik", "yuruyus", "koru"], "maks_fiyat": 1},
    {"id": 5, "kullanici": "Tarihi Gezi", "kategori": ["gezi", "kultur"], "etiketler": ["tarihi", "manzara"], "maks_fiyat": 2},
    {"id": 6, "kullanici": "Nitelikli Kahve Sever", "kategori": ["kafe"], "etiketler": ["kahve", "tatli"], "maks_fiyat": 2},
    {"id": 7, "kullanici": "Gurme Aile Yemeği", "kategori": ["restoran"], "etiketler": ["aile", "gurme", "otantik"], "maks_fiyat": 3},
    {"id": 8, "kullanici": "Manzaralı Park Dinlenmesi", "kategori": ["doga", "gezi"], "etiketler": ["deniz_manzarasi", "yesillik"], "maks_fiyat": 1},
    {"id": 9, "kullanici": "Hızlı Tatlı Molası", "kategori": ["kafe"], "etiketler": ["tatli", "cikolata"], "maks_fiyat": 2},
    {"id": 10, "kullanici": "Modern Sanat Keşfi", "kategori": ["kultur"], "etiketler": ["sanat", "sergi", "modern"], "maks_fiyat": 3}
]

# 4. MODEL DEĞERLENDİRME VE METRİK HESAPLAMA
sonuclar = []
top_k = 5
toplam_sure = 0

print("=== FAZ 6: MODEL DEĞERLENDİRME ÇALIŞTIRILIYOR ===")

for senaryo in test_senaryolari:
    basla = time.time()
    
    # Kural tabanlı filtreleme
    adaylar = df[df["kategori"].isin(senaryo["kategori"])].copy()
    adaylar = adaylar[adaylar["fiyat_seviyesi"] <= senaryo["maks_fiyat"]]
    
    if adaylar.empty:
        adaylar = df.copy()
        
    hedef_kume = set(senaryo["etiketler"])
    adaylar["benzerlik"] = adaylar["etiketler"].apply(lambda x: jaccard_benzerligi(hedef_kume, set(x.split())))
    
    # Sıralama
    oneriler = adaylar.sort_values(by=["benzerlik", "puan"], ascending=[False, False]).head(top_k)
    
    gecen_sure = (time.time() - basla) * 1000  # ms
    toplam_sure += gecen_sure
    
    onerilen_etiketler = oneriler["etiketler"].tolist()
    p_k = precision_at_k(onerilen_etiketler, senaryo["etiketler"], k=top_k)
    hr_k = hit_rate_at_k(onerilen_etiketler, senaryo["etiketler"], k=top_k)
    
    sonuclar.append({
        "Senaryo ID": senaryo["id"],
        "Kullanıcı Profili": senaryo["kullanici"],
        "Precision@5": round(p_k, 2),
        "Hit_Rate@5": round(hr_k, 2),
        "Yanıt Süresi (ms)": round(gecen_sure, 2)
    })

df_metrikler = pd.DataFrame(sonuclar)

# Çıktıları Kaydet
csv_yolu = os.path.join(OUTPUT_DIR, "faz6_model_metrikleri.csv")
df_metrikler.to_csv(csv_yolu, index=False, encoding="utf-8")

ortalama_p = df_metrikler["Precision@5"].mean()
ortalama_hr = df_metrikler["Hit_Rate@5"].mean()
ortalama_sure = df_metrikler["Yanıt Süresi (ms)"].mean()

print("\n--- MODEL DEĞERLENDİRME SONUÇLARI ---")
print(df_metrikler.to_string(index=False))
print(f"\n--> Ortalama Precision@5: %{ortalama_p * 100:.1f}")
print(f"--> Ortalama Hit Rate@5  : %{ortalama_hr * 100:.1f}")
print(f"--> Ortalama Yanıt Süresi: {ortalama_sure:.2f} ms")

# 5. METRİK GRAFİĞİ ÇİZDİRME VE KAYDETME
plt.figure(figsize=(10, 5))
plt.bar(df_metrikler["Kullanıcı Profili"], df_metrikler["Precision@5"], color="#2b5c8f", edgecolor="black")
plt.axhline(y=ortalama_p, color="red", linestyle="--", label=f"Ortalama Precision ({ortalama_p:.2f})")
plt.xticks(rotation=45, ha="right")
plt.title("Faz 6: Test Senaryolarına Göre Precision@5 Başarımı")
plt.ylabel("Precision@5 Skoru")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()

grafik_yolu = os.path.join(OUTPUT_DIR, "faz6_precision_grafigi.png")
plt.savefig(grafik_yolu, dpi=300)
print(f"--> Başarı grafiği '{grafik_yolu}' olarak kaydedildi!\n")