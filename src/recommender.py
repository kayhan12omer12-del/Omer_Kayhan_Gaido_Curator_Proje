"""
Faz 3 ve Faz 4: TF-IDF, Kosinüs Benzerliği, Jaccard Etiket Eşleştirme,
Coğrafi Öklid Mesafe Analizi ve Hibrit Çok Kriterli Öneri Motoru.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_data(csv_path="places_cleaned.csv"):
    """Temiz veri setini yükler ve TF-IDF matrisini hesaplar."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' dosyası bulunamadı! Lütfen veri üretimini kontrol edin.")
    
    df = pd.read_csv(csv_path)
    df["metadata"] = df["kategori"].astype(str) + " " + df["etiketler"].astype(str)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["metadata"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, tfidf_matrix, cosine_sim

def oklid_mesafe_km(lat1, lon1, lat2, lon2):
    """İki koordinat arasındaki Öklid mesafesini km cinsinden hesaplar."""
    try:
        d_lat = (lat1 - lat2) * 111.0
        d_lon = (lon1 - lon2) * 85.0
        return np.sqrt(d_lat**2 + d_lon**2)
    except Exception as e:
        print(f"[HATA] Mesafe hesabı başarısız: {e}")
        return 0.0

def jaccard_benzerligi(kume_a, kume_b):
    """İki etiket kümesi arasındaki Jaccard Benzerliğini hesaplar."""
    if not kume_a or not kume_b:
        return 0.0
    kesisim = len(kume_a.intersection(kume_b))
    birlesim = len(kume_a.union(kume_b))
    return kesisim / birlesim if birlesim > 0 else 0.0

def onerileri_hesapla(
    df,
    cosine_sim,
    user_lat,
    user_lon,
    secilen_kategoriler=None,
    maks_fiyat=4,
    min_puan=3.5,
    maks_mesafe=25.0,
    arama_metni=""
):
    """Tüm kural tabanlı, mesafe tabanlı ve benzerlik tabanlı bileşik önerileri üretir."""
    try:
        filtrelenmis_df = df.copy()
        filtrelenmis_df["mesafe_km"] = oklid_mesafe_km(
            filtrelenmis_df["enlem"], filtrelenmis_df["boylam"], user_lat, user_lon
        ).round(2)
        
        # 1. Kural Tabanlı Filtreler
        if secilen_kategoriler:
            filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["kategori"].isin(secilen_kategoriler)]
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["fiyat_seviyesi"] <= maks_fiyat]
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["puan"] >= min_puan]
        filtrelenmis_df = filtrelenmis_df[filtrelenmis_df["mesafe_km"] <= maks_mesafe]
        
        # Fallback (Geri Çekilme): Filtreleme sonucu boş küme kalırsa esnet
        if filtrelenmis_df.empty:
            filtrelenmis_df = df.copy()
            filtrelenmis_df["mesafe_km"] = oklid_mesafe_km(
                filtrelenmis_df["enlem"], filtrelenmis_df["boylam"], user_lat, user_lon
            ).round(2)
            
        # 2. Benzerlik Hesabı (Arama varsa Jaccard, yoksa 0)
        if arama_metni.strip():
            kullanici_kume = set(arama_metni.lower().split())
            filtrelenmis_df["benzerlik"] = filtrelenmis_df["etiketler"].apply(
                lambda x: jaccard_benzerligi(kullanici_kume, set(str(x).split()))
            )
        else:
            filtrelenmis_df["benzerlik"] = 0.0
            
        # 3. Mesafe Normalizasyonu [0, 1]
        max_dist = filtrelenmis_df["mesafe_km"].max()
        min_dist = filtrelenmis_df["mesafe_km"].min()
        norm_mesafe = (filtrelenmis_df["mesafe_km"] - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0
        
        # 4. Bileşik Skorlama (Hybrid Multi-Criteria Scoring)
        if arama_metni.strip():
            filtrelenmis_df["bilesik_skor"] = (
                (filtrelenmis_df["benzerlik"] * 0.5) +
                ((filtrelenmis_df["puan"] / 5.0) * 0.3) +
                ((1 - norm_mesafe) * 0.2)
            )
        else:
            # Soğuk Başlangıç Formülü: Popülerlik (%70) + Yakınlık (%30)
            norm_yakinlik = 1 - (filtrelenmis_df["mesafe_km"] / (filtrelenmis_df["mesafe_km"].max() + 1e-5))
            filtrelenmis_df["bilesik_skor"] = (filtrelenmis_df["puan"] / 5.0) * 0.7 + norm_yakinlik * 0.3
            
        return filtrelenmis_df.sort_values(by="bilesik_skor", ascending=False)
        
    except Exception as e:
        print(f"[HATA] Öneri motoru istisnası: {e}")
        return df.head(5)