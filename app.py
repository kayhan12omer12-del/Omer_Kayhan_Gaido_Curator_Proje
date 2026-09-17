import streamlit as st
import folium
from streamlit_folium import st_folium
from src.recommender import load_and_preprocess_data, onerileri_hesapla
from src.config import SEMT_KOORDINATLARI, KATEGORI_RENKLERI

st.set_page_config(page_title="Gaido Curator - Mekan Öneri Sistemi", page_icon="📍", layout="wide")

@st.cache_data
def get_data():
    return load_and_preprocess_data("places_cleaned.csv")

df, _, cosine_sim = get_data()

st.sidebar.title("🔍 Filtreleme ve Tercihler")
secilen_semt = st.sidebar.selectbox("Konumunuz (Semt):", list(SEMT_KOORDINATLARI.keys()), index=0)
user_lat, user_lon = SEMT_KOORDINATLARI[secilen_semt]

tum_kategoriler = list(df["kategori"].unique())
secilen_kategoriler = st.sidebar.multiselect("Kategori Seçin:", tum_kategoriler, default=tum_kategoriler)
maks_fiyat = st.sidebar.slider("Maksimum Fiyat Seviyesi (1-4):", 1, 4, 4)
min_puan = st.sidebar.slider("Minimum Mekan Puanı:", 3.5, 5.0, 4.0, step=0.1)
maks_mesafe = st.sidebar.slider("Maksimum Mesafe Yarıçapı (km):", 1.0, 25.0, 10.0, step=1.0)
arama_metni = st.sidebar.text_input("Aradığınız Özellikler (Örn: sessiz kahve deniz):", "")

sonuclar_df = onerileri_hesapla(
    df=df,
    cosine_sim=cosine_sim,
    user_lat=user_lat,
    user_lon=user_lon,
    secilen_kategoriler=secilen_kategoriler,
    maks_fiyat=maks_fiyat,
    min_puan=min_puan,
    maks_mesafe=maks_mesafe,
    arama_metni=arama_metni
)
onerilenler = sonuclar_df.head(5)

st.title("📍 Gaido Curator - Akıllı Mekan Öneri Sistemi")
st.markdown(f"**Mevcut Konum:** {secilen_semt} | **Önerilen En İyi {len(onerilenler)} Mekan**")

col_harita, col_liste = st.columns([3, 2])

with col_harita:
    st.subheader("🗺️ Önerilen Mekanlar Haritası")
    harita = folium.Map(location=[user_lat, user_lon], zoom_start=13)
    
    folium.Marker(
        location=[user_lat, user_lon],
        popup="<b>Sizin Konumunuz</b>",
        tooltip="Buradasınız",
        icon=folium.Icon(color="blue", icon="user", prefix="fa")
    ).add_to(harita)
    
    for _, row in onerilenler.iterrows():
        folium.Marker(
            location=[row["enlem"], row["boylam"]],
            popup=f"<b>{row['ad']}</b><br>Kategori: {row['kategori']}<br>Puan: ⭐ {row['puan']}<br>Mesafe: {row['mesafe_km']} km",
            tooltip=row["ad"],
            icon=folium.Icon(color=KATEGORI_RENKLERI.get(row["kategori"], "gray"), icon="info-sign")
        ).add_to(harita)
        
    st_folium(harita, width="100%", height=500)

with col_liste:
    st.subheader("📋 Mekan Detayları")
    for _, row in onerilenler.iterrows():
        fiyat_isareti = "₺" * int(row["fiyat_seviyesi"])
        st.markdown(f"""
        <div style="padding:12px; margin-bottom:12px; border-radius:8px; border:1px solid #ddd; background-color:#1e1e1e; color:white;">
            <h4 style="margin:0; color:#4CAF50;">{row['ad']}</h4>
            <p style="margin:4px 0;"><b>Kategori:</b> {row['kategori'].capitalize()} | <b>Fiyat:</b> {fiyat_isareti} ({row['fiyat_seviyesi']}/4)</p>
            <p style="margin:4px 0;"><b>Puan:</b> ⭐ {row['puan']} / 5.0 | <b>Mesafe:</b> 🚗 {row['mesafe_km']} km</p>
            <p style="margin:4px 0; font-size:13px; color:#bbb;"><b>Etiketler:</b> {row['etiketler']}</p>
        </div>
        """, unsafe_allow_html=True)