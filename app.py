import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import os

# ==================== UTILS ====================

@st.cache_resource
def load_model_and_scaler(model_name):
    model_path = f"models/{model_name}_model.pkl"
    scaler_path = f"models/{model_name}_scaler.pkl"
    features_path = f"models/{model_name}_features.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Modèle introuvable : {model_path}")
        st.stop()
    model = joblib.load(model_path)
    
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        st.warning(f"Scaler non trouvé pour {model_name}. Entrées brutes.")
    
    features = None
    if os.path.exists(features_path):
        features = joblib.load(features_path)
    else:
        st.info(f"Liste des features non trouvée pour {model_name}.")
    
    return model, scaler, features

@st.cache_resource
def load_city_encoder():
    """Charge l'encodeur de villes (label encoder) s'il existe, sinon utilise un mapping par défaut."""
    encoder_path = "models/city_encoder.pkl"
    if os.path.exists(encoder_path):
        return joblib.load(encoder_path)
    else:
        # Liste des 42 villes du dataset (ordre alphabétique pour reproductibilité)
        cities = [
            "Abong-Mbang", "Akonolinga", "Ambam", "Bafoussam", "Bafia", "Bamenda",
            "Batouri", "Bertoua", "Buea", "Dschang", "Ebolowa", "Edea", "Foumban",
            "Garoua", "Guider", "Kousseri", "Kribi", "Kumba", "Kumbo", "Limbe",
            "Loum", "Mamfe", "Maroua", "Mbalmayo", "Mbengwi", "Mbouda", "Meiganga",
            "Mokolo", "Ngaoundere", "Nkongsamba", "Poli", "Sangmelima", "Tibati",
            "Tignere", "Touboro", "Wum", "Yagoua", "Yaounde", "Yokadouma", "Bafia",
            "Bertoua", "Douala"  # à compléter si nécessaire
        ]
        cities = sorted(set(cities))  # unique et trié
        city_to_code = {city: idx for idx, city in enumerate(cities)}
        return city_to_code  # dictionnaire simple

def compute_is_dry_season(month):
    return 1 if month in [11, 12, 1, 2, 3] else 0

# ==================== PAGE VRI ====================

def page_vri(model, scaler, features, model_key):
    st.header("🦟 Indice de Risque Vectoriel (VRI)")
    st.markdown("""
    Le **VRI** évalue le risque de prolifération des moustiques vecteurs de maladies.
    """)

    # Importance des features
    if hasattr(model, 'feature_importances_') and features is not None:
        if st.checkbox("Afficher l'importance des variables"):
            imp_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig = px.bar(imp_df, x='importance', y='feature', orientation='h')
            st.plotly_chart(fig)
    else:
        st.info("Importance des variables non disponible.")

    # Formulaire
    st.subheader("📝 Prédiction personnalisée")
    
    # Chargement du mapping des villes
    city_encoder = load_city_encoder()
    city_names = list(city_encoder.keys()) if isinstance(city_encoder, dict) else city_encoder.classes_
    
    default_features_order = [
        'temperature_2m_mean', 'relative_humidity_2m_mean',
        'precipitation_sum', 'precipitation_hours', 'is_dry_season', 'city_encoded'
    ]
    
    with st.form(key="vri_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("🌡️ Température moyenne (°C)", value=25.0, step=0.5)
            humidity = st.slider("💧 Humidité relative moyenne (%)", 0, 100, 70)
            precip = st.number_input("☔ Précipitations totales (mm)", value=5.0, step=1.0)
        with col2:
            precip_hours = st.number_input("⏱️ Heures de précipitations", value=2.0, step=0.5, min_value=0.0, max_value=24.0)
            month = st.selectbox("📅 Mois", list(range(1,13)), index=5)
            city = st.selectbox("🏙️ Ville", city_names)
        
        is_dry = compute_is_dry_season(month)
        st.write(f"Saison : {'Sèche' if is_dry else 'Pluvieuse'}")
        
        submitted = st.form_submit_button("Prédire le VRI")
        if submitted:
            # Encodage de la ville
            if isinstance(city_encoder, dict):
                city_code = city_encoder[city]
            else:
                city_code = city_encoder.transform([city])[0]
            
            input_dict = {
                'temperature_2m_mean': temp,
                'relative_humidity_2m_mean': humidity,
                'precipitation_sum': precip,
                'precipitation_hours': precip_hours,
                'is_dry_season': is_dry,
                'city_encoded': city_code
            }
            input_df = pd.DataFrame([input_dict])
            
            # Réorganisation selon les features du modèle
            if features is not None:
                missing = set(features) - set(input_df.columns)
                if missing:
                    st.error(f"Features manquantes : {missing}")
                    st.stop()
                input_df = input_df[features]
            else:
                input_df = input_df[default_features_order]
                st.info("Ordre par défaut utilisé.")
            
            # Scaling
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            
            pred = model.predict(input_scaled)[0]
            pred_clamped = max(0.0, min(1.0, pred))
            st.metric("VRI", f"{pred_clamped:.3f}")
            st.progress(pred_clamped, text="Niveau de risque")

# ==================== PAGE GÉNÉRIQUE (avec ville) ====================

def page_generic(model, scaler, features, model_name, description, input_fields):
    st.header(f"📊 {model_name}")
    st.markdown(description)
    
    city_encoder = load_city_encoder()
    city_names = list(city_encoder.keys()) if isinstance(city_encoder, dict) else city_encoder.classes_
    
    with st.form(key=f"form_{model_name}"):
        inputs = {}
        # Champ ville si 'city_encoded' est dans les features
        if features is not None and 'city_encoded' in features:
            city = st.selectbox("🏙️ Ville", city_names)
            if isinstance(city_encoder, dict):
                inputs['city_encoded'] = city_encoder[city]
            else:
                inputs['city_encoded'] = city_encoder.transform([city])[0]
        
        # Autres champs
        for field_name, field_config in input_fields.items():
            if field_config['type'] == 'number':
                inputs[field_name] = st.number_input(field_config['label'], 
                                                     value=field_config.get('value', 0.0),
                                                     step=field_config.get('step', 0.1))
            elif field_config['type'] == 'slider':
                inputs[field_name] = st.slider(field_config['label'],
                                               field_config['min'], field_config['max'],
                                               field_config.get('value', 50))
            elif field_config['type'] == 'select':
                inputs[field_name] = st.selectbox(field_config['label'], field_config['options'])
        
        submitted = st.form_submit_button("Prédire")
        if submitted:
            input_df = pd.DataFrame([inputs])
            if features is not None:
                input_df = input_df[features]
            if scaler is not None:
                input_df = scaler.transform(input_df)
            pred = model.predict(input_df)[0]
            st.metric("Prédiction", f"{pred:.2f}")

# ==================== CONFIGURATION DES MODÈLES ====================

PAGES_CONFIG = {
    # "Indice de Stress Thermique (HSI)": {
    #     "key": "hsi",
    #     "description": "Indice combinant température et humidité.",
    #     "fields": {
    #         "temperature_2m_max": {"type": "number", "label": "Température max (°C)", "value": 30.0, "step": 0.5},
    #         "relative_humidity_2m_mean": {"type": "slider", "label": "Humidité (%)", "min": 0, "max": 100, "value": 70},
    #     }
    # },
    # "Indice de Qualité de l'Air (IQA)": {
    #     "key": "iqa",
    #     "description": "Proxy qualité de l'air.",
    #     "fields": {
    #         "wind_speed_10m_max": {"type": "number", "label": "Vent max (m/s)", "value": 2.0, "step": 0.5},
    #         "precipitation_sum": {"type": "number", "label": "Pluie (mm)", "value": 0.0, "step": 1.0},
    #     }
    # },
    # "Risque d'Inondation (FRI)": {"key": "fri", "description": "...", "fields": {}},
    # "Indice de Sécheresse (SPEI)": {"key": "spei", "description": "...", "fields": {}},
    "Risque Vectoriel (VRI)": {"key": "vri", "description": "...", "fields": {}},  # géré par page_vri
    # "Potentiel Solaire (SEP)": {"key": "sep", "description": "...", "fields": {}},
    # "Indice Composite de Risque Sanitaire (CHRI)": {"key": "chri", "description": "...", "fields": {}},
    # "Évapotranspiration (ETO)": {"key": "eto", "description": "...", "fields": {}},
    # "Classification du code météo (Weather Code)": {"key": "weather_code", "description": "...", "fields": {}},
    # "Risque d'Incendie (Fire Risk)": {"key": "fire_risk", "description": "...", "fields": {}},
}

# ==================== APPLICATION PRINCIPALE ====================

st.set_page_config(page_title="IndabaX Climate Dashboard", layout="wide")
st.title("🌍 IA pour la résilience climatique et sanitaire – Cameroun")

st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choisissez un modèle", list(PAGES_CONFIG.keys()))

config = PAGES_CONFIG[option]
model_key = config["key"]
model, scaler, features = load_model_and_scaler(model_key)

if option == "Risque Vectoriel (VRI)":
    page_vri(model, scaler, features, model_key)
else:
    page_generic(model, scaler, features, option, config["description"], config.get("fields", {}))
