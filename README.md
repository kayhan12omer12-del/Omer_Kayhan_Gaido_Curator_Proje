# 📍 Gaido Curator - Akıllı Mekan Öneri Sistemi

Gaido Curator; kullanıcıların konum, kategori, bütçe, puan ve serbest metin tercihlerine göre İstanbul genelinde en uygun mekanları öneren hibrit bir tavsiye sistemidir.

## 🚀 Proje Mimarisi ve Algoritma

Sistem 3 temel aşamalı bir karar mimarisiyle çalışır:

1. **Kural Tabanlı Filtreleme:** Kategori, bütçe (<= 4) ve puan (>= 3.5) sınırlandırmaları.
2. **Coğrafi Filtreleme & Normalizasyon:** Kullanıcının seçtiği semt merkezi baz alınarak mekanların Öklid mesafesi hesaplanır ve [0, 1] aralığına normalize edilir.
3. **Bileşik Skorlama (Hybrid Multi-Criteria Scoring):**
   * **TF-IDF & Jaccard Benzerliği:** Metin/etiket eşleşmesi (%50 ağırlık).
   * **Normalize Puan:** Mekanın kullanıcı puanı (%30 ağırlık).
   * **Ters Mesafe Skoru:** Kullanıcıya yakınlık derecesi (%20 ağırlık).

## 📁 Proje Dosya Yapısı

- **src/**: Projenin modüler kaynak kodları (`config.py`, `recommender.py`).
- **outputs/**: Model metrik tabloları, test raporları ve grafikler.
- **app.py**: Streamlit ve Folium tabanlı interaktif web arayüzü.
- **evaluate.py**: Model başarı metriklerini (Precision@5, Hit Rate@5) hesaplayan test betiği.
- **main.py**: Konsol tabanlı test ve veri akış motoru.
- **places_cleaned.csv**: Gerçekçi mekan veri seti.
- **requirements.txt**: Proje bağımlılık listesi.

## 🛠 Kurulum ve Çalıştırma

### 1. Sanal Ortamı Oluşturun ve Aktifleştirin
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 2. Bağımlılıkları Yükleyin
```powershell
pip install -r requirements.txt
```
# Konsol tabanlı test motorunu çalıştırmak için:
python main.py

# Streamlit interaktif web arayüzünü açmak için:
streamlit run app.py
```
