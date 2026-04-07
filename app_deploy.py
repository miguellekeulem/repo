# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# import joblib
# import os

# # ==================== UTILS ====================

# @st.cache_resource
# def load_model_and_scaler(model_name):
#     model_path = f"models/{model_name}_model.pkl"
#     scaler_path = f"models/{model_name}_scaler.pkl"
#     features_path = f"models/{model_name}_features.pkl"
    
#     if not os.path.exists(model_path):
#         st.error(f"Modèle introuvable : {model_path}")
#         st.stop()
#     model = joblib.load(model_path)
    
#     scaler = None
#     if os.path.exists(scaler_path):
#         scaler = joblib.load(scaler_path)
#     else:
#         st.warning(f"Scaler non trouvé pour {model_name}. Entrées brutes.")
    
#     features = None
#     if os.path.exists(features_path):
#         features = joblib.load(features_path)
#     else:
#         st.info(f"Liste des features non trouvée pour {model_name}.")
    
#     return model, scaler, features

# @st.cache_resource
# def load_city_encoder():
#     """Charge l'encodeur de villes (label encoder) s'il existe, sinon utilise un mapping par défaut."""
#     encoder_path = "models/city_encoder.pkl"
#     if os.path.exists(encoder_path):
#         return joblib.load(encoder_path)
#     else:
#         # Liste des 42 villes du dataset (ordre alphabétique pour reproductibilité)
#         cities = [
#             "Abong-Mbang", "Akonolinga", "Ambam", "Bafoussam", "Bafia", "Bamenda",
#             "Batouri", "Bertoua", "Buea", "Dschang", "Ebolowa", "Edea", "Foumban",
#             "Garoua", "Guider", "Kousseri", "Kribi", "Kumba", "Kumbo", "Limbe",
#             "Loum", "Mamfe", "Maroua", "Mbalmayo", "Mbengwi", "Mbouda", "Meiganga",
#             "Mokolo", "Ngaoundere", "Nkongsamba", "Poli", "Sangmelima", "Tibati",
#             "Tignere", "Touboro", "Wum", "Yagoua", "Yaounde", "Yokadouma", "Bafia",
#             "Bertoua", "Douala"  # à compléter si nécessaire
#         ]
#         cities = sorted(set(cities))  # unique et trié
#         city_to_code = {city: idx for idx, city in enumerate(cities)}
#         return city_to_code  # dictionnaire simple

# def compute_is_dry_season(month):
#     return 1 if month in [11, 12, 1, 2, 3] else 0

# # ==================== PAGE VRI ====================

# def page_vri(model, scaler, features, model_key):
#     st.header("🦟 Indice de Risque Vectoriel (VRI)")
#     st.markdown("""
#     Le **VRI** évalue le risque de prolifération des moustiques vecteurs de maladies.
#     """)

#     # Importance des features
#     if hasattr(model, 'feature_importances_') and features is not None:
#         if st.checkbox("Afficher l'importance des variables"):
#             imp_df = pd.DataFrame({
#                 'feature': features,
#                 'importance': model.feature_importances_
#             }).sort_values('importance', ascending=False)
#             fig = px.bar(imp_df, x='importance', y='feature', orientation='h')
#             st.plotly_chart(fig)
#     else:
#         st.info("Importance des variables non disponible.")

#     # Formulaire
#     st.subheader("📝 Prédiction personnalisée")
    
#     # Chargement du mapping des villes
#     city_encoder = load_city_encoder()
#     city_names = list(city_encoder.keys()) if isinstance(city_encoder, dict) else city_encoder.classes_
    
#     default_features_order = [
#         'temperature_2m_mean', 'relative_humidity_2m_mean',
#         'precipitation_sum', 'precipitation_hours', 'is_dry_season', 'city_encoded'
#     ]
    
#     with st.form(key="vri_form"):
#         col1, col2 = st.columns(2)
#         with col1:
#             temp = st.number_input("🌡️ Température moyenne (°C)", value=25.0, step=0.5)
#             humidity = st.slider("💧 Humidité relative moyenne (%)", 0, 100, 70)
#             precip = st.number_input("☔ Précipitations totales (mm)", value=5.0, step=1.0)
#         with col2:
#             precip_hours = st.number_input("⏱️ Heures de précipitations", value=2.0, step=0.5, min_value=0.0, max_value=24.0)
#             month = st.selectbox("📅 Mois", list(range(1,13)), index=5)
#             city = st.selectbox("🏙️ Ville", city_names)
        
#         is_dry = compute_is_dry_season(month)
#         st.write(f"Saison : {'Sèche' if is_dry else 'Pluvieuse'}")
        
#         submitted = st.form_submit_button("Prédire le VRI")
#         if submitted:
#             # Encodage de la ville
#             if isinstance(city_encoder, dict):
#                 city_code = city_encoder[city]
#             else:
#                 city_code = city_encoder.transform([city])[0]
            
#             input_dict = {
#                 'temperature_2m_mean': temp,
#                 'relative_humidity_2m_mean': humidity,
#                 'precipitation_sum': precip,
#                 'precipitation_hours': precip_hours,
#                 'is_dry_season': is_dry,
#                 'city_encoded': city_code
#             }
#             input_df = pd.DataFrame([input_dict])
            
#             # Réorganisation selon les features du modèle
#             if features is not None:
#                 missing = set(features) - set(input_df.columns)
#                 if missing:
#                     st.error(f"Features manquantes : {missing}")
#                     st.stop()
#                 input_df = input_df[features]
#             else:
#                 input_df = input_df[default_features_order]
#                 st.info("Ordre par défaut utilisé.")
            
#             # Scaling
#             if scaler is not None:
#                 input_scaled = scaler.transform(input_df)
#             else:
#                 input_scaled = input_df.values
            
#             pred = model.predict(input_scaled)[0]
#             pred_clamped = max(0.0, min(1.0, pred))
#             st.metric("VRI", f"{pred_clamped:.3f}")
#             st.progress(pred_clamped, text="Niveau de risque")

# # ==================== PAGE GÉNÉRIQUE (avec ville) ====================

# def page_generic(model, scaler, features, model_name, description, input_fields):
#     st.header(f"📊 {model_name}")
#     st.markdown(description)
    
#     city_encoder = load_city_encoder()
#     city_names = list(city_encoder.keys()) if isinstance(city_encoder, dict) else city_encoder.classes_
    
#     with st.form(key=f"form_{model_name}"):
#         inputs = {}
#         # Champ ville si 'city_encoded' est dans les features
#         if features is not None and 'city_encoded' in features:
#             city = st.selectbox("🏙️ Ville", city_names)
#             if isinstance(city_encoder, dict):
#                 inputs['city_encoded'] = city_encoder[city]
#             else:
#                 inputs['city_encoded'] = city_encoder.transform([city])[0]
        
#         # Autres champs
#         for field_name, field_config in input_fields.items():
#             if field_config['type'] == 'number':
#                 inputs[field_name] = st.number_input(field_config['label'], 
#                                                      value=field_config.get('value', 0.0),
#                                                      step=field_config.get('step', 0.1))
#             elif field_config['type'] == 'slider':
#                 inputs[field_name] = st.slider(field_config['label'],
#                                                field_config['min'], field_config['max'],
#                                                field_config.get('value', 50))
#             elif field_config['type'] == 'select':
#                 inputs[field_name] = st.selectbox(field_config['label'], field_config['options'])
        
#         submitted = st.form_submit_button("Prédire")
#         if submitted:
#             input_df = pd.DataFrame([inputs])
#             if features is not None:
#                 input_df = input_df[features]
#             if scaler is not None:
#                 input_df = scaler.transform(input_df)
#             pred = model.predict(input_df)[0]
#             st.metric("Prédiction", f"{pred:.2f}")

# # ==================== CONFIGURATION DES MODÈLES ====================

# PAGES_CONFIG = {
#     "Indice de Stress Thermique (HSI)": {
#         "key": "hsi",
#         "description": "Indice combinant température et humidité.",
#         "fields": {
#             "temperature_2m_max": {"type": "number", "label": "Température max (°C)", "value": 30.0, "step": 0.5},
#             "relative_humidity_2m_mean": {"type": "slider", "label": "Humidité (%)", "min": 0, "max": 100, "value": 70},
#         }
#     },
#     "Indice de Qualité de l'Air (IQA)": {
#         "key": "iqa",
#         "description": "Proxy qualité de l'air.",
#         "fields": {
#             "wind_speed_10m_max": {"type": "number", "label": "Vent max (m/s)", "value": 2.0, "step": 0.5},
#             "precipitation_sum": {"type": "number", "label": "Pluie (mm)", "value": 0.0, "step": 1.0},
#         }
#     },
#     "Risque d'Inondation (FRI)": {"key": "fri", "description": "...", "fields": {}},
#     "Indice de Sécheresse (SPEI)": {"key": "spei", "description": "...", "fields": {}},
#     "Risque Vectoriel (VRI)": {"key": "vri", "description": "...", "fields": {}},  # géré par page_vri
#     "Potentiel Solaire (SEP)": {"key": "sep", "description": "...", "fields": {}},
#     "Indice Composite de Risque Sanitaire (CHRI)": {"key": "chri", "description": "...", "fields": {}},
#     "Évapotranspiration (ETO)": {"key": "eto", "description": "...", "fields": {}},
#     "Classification du code météo (Weather Code)": {"key": "weather_code", "description": "...", "fields": {}},
#     "Risque d'Incendie (Fire Risk)": {"key": "fire_risk", "description": "...", "fields": {}},
# }

# # ==================== APPLICATION PRINCIPALE ====================

# st.set_page_config(page_title="IndabaX Climate Dashboard", layout="wide")
# st.title("🌍 IA pour la résilience climatique et sanitaire – Cameroun")

# st.sidebar.header("Navigation")
# option = st.sidebar.selectbox("Choisissez un modèle", list(PAGES_CONFIG.keys()))

# config = PAGES_CONFIG[option]
# model_key = config["key"]
# model, scaler, features = load_model_and_scaler(model_key)

# if option == "Risque Vectoriel (VRI)":
#     page_vri(model, scaler, features, model_key)
# else:
#     page_generic(model, scaler, features, option, config["description"], config.get("fields", {}))






import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utilitaires (remplacent utils.py)
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler(model_name):
    """Charge modèle, scaler (si existant) et liste des features (si existante)."""
    model_path = f"{model_name}_model.pkl"
    scaler_path = f"{model_name}_scaler.pkl"
    features_path = f"{model_name}_features.pkl"

    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        st.stop()
    model = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    features = None
    if os.path.exists(features_path):
        features = joblib.load(features_path)

    return model, scaler, features

def compute_is_dry_season(month):
    """1 si saison sèche (novembre à mars), 0 sinon."""
    return 1 if month in [11, 12, 1, 2, 3] else 0

def normalize_series(series):
    """Normalisation min-max d'une série pandas."""
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

# ------------------------------------------------------------
# Pages spécifiques (remplacent vri_page.py, iqa_page.py, ...)
# ------------------------------------------------------------
def show_hsi_page(model, scaler, features, model_key):
    st.header("🌡️ Indice de Stress Thermique (HSI)")
    st.markdown("""
    Le HSI combine température et humidité pour évaluer la sensation thermique.
    Plus l’indice est élevé, plus le risque sanitaire (coup de chaleur, déshydratation) est grand.
    """)

    # Importance des features (si disponible)
    if hasattr(model, 'feature_importances_') and features is not None:
        st.subheader("📊 Importance des variables")
        imp_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        fig = px.bar(imp_df, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig)

    # Formulaire de prédiction
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="hsi_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp_max = st.number_input("Température max (°C)", value=30.0, step=0.5)
            humidity = st.slider("Humidité relative (%)", 0, 100, 70)
            wind = st.number_input("Vitesse du vent (m/s)", value=2.0, step=0.5)
        with col2:
            precip = st.number_input("Précipitations (mm)", value=0.0, step=1.0)
            rad = st.number_input("Rayonnement solaire (W/m²)", value=200.0, step=10.0)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)

        submitted = st.form_submit_button("Prédire le HSI")
        if submitted:
            input_dict = {
                'temperature_2m_mean': temp_max,
                'relative_humidity_2m_mean': humidity,
                'wind_speed_10m_max': wind,
                'precipitation_sum': precip,
                'shortwave_radiation_sum': rad,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("HSI prédit", f"{pred:.1f} °C")
            # Interprétation simple
            if pred < 27:
                st.success("✅ Aucun stress thermique.")
            elif pred < 32:
                st.info("⚠️ Stress thermique modéré (prudence).")
            elif pred < 41:
                st.warning("🔥 Stress thermique fort (danger potentiel).")
            else:
                st.error("💀 Stress thermique extrême (danger immédiat).")

def show_iqa_page(model, scaler, features, model_key):
    st.header("🏭 Indice de Qualité de l'Air (IQA)")
    st.markdown("""
    L'IQA est un proxy composite basé sur la stagnation, le rayonnement et la saison.
    Plus l'indice est élevé, plus la qualité de l'air est dégradée.
    """)
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="iqa_form"):
        col1, col2 = st.columns(2)
        with col1:
            wind = st.number_input("Vitesse du vent max (m/s)", value=3.0, step=0.5)
            precip = st.number_input("Cumul pluie (mm)", value=5.0, step=1.0)
            rad = st.number_input("Rayonnement solaire (W/m²)", value=250.0, step=10.0)
        with col2:
            et0 = st.number_input("Évapotranspiration (mm)", value=3.5, step=0.5)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire l'IQA")
        if submitted:
            dry = compute_is_dry_season(month)
            input_dict = {
                'wind_speed_10m_max': wind,
                'precipitation_sum': precip,
                'shortwave_radiation_sum': rad,
                'et0_fao_evapotranspiration': et0,
                'is_dry_season': dry,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("IQA prédit", f"{pred:.2f}")
            if pred < 0.33:
                st.success("Bonne qualité de l'air")
            elif pred < 0.66:
                st.info("Qualité moyenne")
            else:
                st.error("Mauvaise qualité de l'air")

def show_fri_page(model, scaler, features, model_key):
    st.header("🌊 Risque d'Inondation (FRI)")
    st.markdown("Indice basé sur les précipitations cumulées et l'évapotranspiration.")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="fri_form"):
        col1, col2 = st.columns(2)
        with col1:
            precip_3d = st.number_input("Cumul pluie 3 jours (mm)", value=30.0, step=5.0)
            et0_3d = st.number_input("ET0 cumul 3 jours (mm)", value=10.0, step=2.0)
        with col2:
            precip_hours = st.number_input("Heures de pluie (sur 24h)", value=6, step=1, min_value=0)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire le FRI")
        if submitted:
            input_dict = {
                'precipitation_3d_sum': precip_3d,
                'et0_3d_sum': et0_3d,
                'precipitation_hours': precip_hours,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("Risque d'inondation", f"{pred:.2f}")
            if pred < 0.5:
                st.success("Risque faible")
            else:
                st.error("Risque élevé")

def show_spei_page(model, scaler, features, model_key):
    st.header("💧 Indice de Sécheresse (SPEI simplifié)")
    st.markdown("Évalue le déficit hydrique (P - ET0) standardisé.")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="spei_form"):
        col1, col2 = st.columns(2)
        with col1:
            precip = st.number_input("Précipitations cumul (mm)", value=50.0, step=5.0)
            et0 = st.number_input("ET0 cumul (mm)", value=60.0, step=5.0)
        with col2:
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire le SPEI")
        if submitted:
            input_dict = {'precipitation_sum': precip, 'et0_fao_evapotranspiration': et0, 'month': month}
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("SPEI", f"{pred:.2f}")
            if pred < -1:
                st.error("Sécheresse sévère")
            elif pred < 0:
                st.warning("Sécheresse modérée")
            else:
                st.success("Conditions normales ou humides")

def show_vri_page(model, scaler, features, model_key):
    st.header("🦟 Indice de Risque Vectoriel (VRI)")
    st.markdown("""
    Évalue le risque de prolifération des moustiques (paludisme, dengue) en fonction de la température,
    de l'humidité et des précipitations.
    """)
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="vri_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Température moyenne (°C)", value=25.0, step=0.5)
            humidity = st.slider("Humidité relative (%)", 0, 100, 70)
            precip = st.number_input("Précipitations cumul (mm)", value=10.0, step=2.0)
        with col2:
            precip_hours = st.number_input("Heures de pluie", value=5, step=1, min_value=0)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire le VRI")
        if submitted:
            dry = compute_is_dry_season(month)
            input_dict = {
                'temperature_2m_mean': temp,
                'relative_humidity_2m_mean': humidity,
                'precipitation_sum': precip,
                'precipitation_hours': precip_hours,
                'is_dry_season': dry,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("VRI", f"{pred:.2f}")
            if pred < 0.33:
                st.success("Risque faible")
            elif pred < 0.66:
                st.info("Risque modéré")
            else:
                st.error("Risque élevé")

def show_sep_page(model, scaler, features, model_key):
    st.header("☀️ Potentiel Solaire (SEP)")
    st.markdown("Prédiction du rendement énergétique solaire basée sur le rayonnement et la durée d'ensoleillement.")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="sep_form"):
        col1, col2 = st.columns(2)
        with col1:
            rad = st.number_input("Rayonnement solaire (W/m²)", value=300.0, step=10.0)
            sunshine = st.number_input("Durée d'ensoleillement (heures)", value=8.0, step=0.5)
        with col2:
            daylight = st.number_input("Durée du jour (heures)", value=12.0, step=0.5)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire le SEP")
        if submitted:
            input_dict = {
                'shortwave_radiation_sum': rad,
                'sunshine_duration': sunshine,
                'daylight_duration': daylight,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("Production solaire (indice)", f"{pred:.2f}")
            if pred < 0.33:
                st.warning("Potentiel faible")
            elif pred < 0.66:
                st.info("Potentiel moyen")
            else:
                st.success("Potentiel élevé")

def show_chri_page(model, scaler, features, model_key):
    st.header("🏥 Indice Composite de Risque Sanitaire (CHRI)")
    st.markdown("Combinaison des risques chaleur (HSI), pollution (IQA) et vectoriel (VRI).")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="chri_form"):
        col1, col2 = st.columns(2)
        with col1:
            hsi = st.number_input("HSI (°C)", value=30.0, step=1.0)
            iqa = st.number_input("IQA (0-1)", value=0.5, step=0.05)
        with col2:
            vri = st.number_input("VRI (0-1)", value=0.4, step=0.05)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire le CHRI")
        if submitted:
            input_dict = {'HSI_norm': hsi/50, 'IQA_norm': iqa, 'VRI_norm': vri, 'month': month}
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("Risque sanitaire global", f"{pred:.2f}")
            if pred < 0.33:
                st.success("Risque faible")
            elif pred < 0.66:
                st.info("Risque modéré")
            else:
                st.error("Risque élevé")

def show_eto_page(model, scaler, features, model_key):
    st.header("💨 Évapotranspiration (ET0)")
    st.markdown("Prédiction de l'évapotranspiration de référence (FAO) à partir des variables météo.")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="eto_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Température moyenne (°C)", value=25.0, step=0.5)
            rad = st.number_input("Rayonnement solaire (W/m²)", value=250.0, step=10.0)
        with col2:
            wind = st.number_input("Vitesse du vent (m/s)", value=2.5, step=0.5)
            humidity = st.slider("Humidité (%)", 0, 100, 60)
        submitted = st.form_submit_button("Prédire l'ET0")
        if submitted:
            input_dict = {
                'temperature_2m_mean': temp,
                'shortwave_radiation_sum': rad,
                'wind_speed_10m_max': wind,
                'relative_humidity_2m_mean': humidity
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("ET0 prédite", f"{pred:.2f} mm/jour")

def show_weather_code_page(model, scaler, features, model_key):
    st.header("☁️ Classification du code météo (WMO)")
    st.markdown("Prédit le code WMO (pluie, orage, brouillard, etc.) à partir des mesures continues.")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="wcode_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Température moyenne (°C)", value=25.0, step=0.5)
            humidity = st.slider("Humidité (%)", 0, 100, 70)
            precip = st.number_input("Précipitations (mm)", value=0.0, step=1.0)
        with col2:
            wind = st.number_input("Vitesse du vent (m/s)", value=3.0, step=0.5)
            rad = st.number_input("Rayonnement (W/m²)", value=200.0, step=10.0)
            month = st.selectbox("Mois", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Prédire le code météo")
        if submitted:
            input_dict = {
                'temperature_2m_mean': temp,
                'relative_humidity_2m_mean': humidity,
                'precipitation_sum': precip,
                'wind_speed_10m_max': wind,
                'shortwave_radiation_sum': rad,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred_class = model.predict(X)[0]
            st.metric("Code WMO", int(pred_class))
            # Dictionnaire simplifié des codes
            wmo_desc = {0: "Ciel clair", 1: "Peu nuageux", 2: "Nuageux", 3: "Couvert",
                        45: "Brouillard", 51: "Bruine légère", 61: "Pluie modérée", 80: "Averses"}
            desc = wmo_desc.get(int(pred_class), "Code inconnu")
            st.write(f"**Description :** {desc}")

def show_fire_risk_page(model, scaler, features, model_key):
    st.header("🔥 Risque d'Incendie")
    st.markdown("Indice basé sur température élevée, faible humidité, absence de pluie, vent et rayonnement.")
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="fire_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp_max = st.number_input("Température max (°C)", value=35.0, step=1.0)
            humidity_min = st.slider("Humidité min (%)", 0, 100, 20)
            precip = st.number_input("Précipitations (mm)", value=0.0, step=1.0)
        with col2:
            wind = st.number_input("Vitesse du vent max (m/s)", value=5.0, step=0.5)
            rad = st.number_input("Rayonnement (W/m²)", value=350.0, step=10.0)
            month = st.selectbox("Mois", list(range(1, 13)), index=2)  # mars par défaut
        submitted = st.form_submit_button("Prédire le risque")
        if submitted:
            input_dict = {
                'temperature_2m_max': temp_max,
                'relative_humidity_2m_min': humidity_min,
                'precipitation_sum': precip,
                'wind_speed_10m_max': wind,
                'shortwave_radiation_sum': rad,
                'month': month
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                input_df = input_df[features]
            X = scaler.transform(input_df) if scaler is not None else input_df.values
            pred = model.predict(X)[0]
            st.metric("Risque d'incendie", f"{pred:.2f}")
            if pred < 0.33:
                st.success("Risque faible")
            elif pred < 0.66:
                st.info("Risque modéré")
            else:
                st.error("Risque élevé")

# ------------------------------------------------------------
# Application principale
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="IndabaX Climate Dashboard", layout="wide")
    st.title("🌍 IA pour la résilience climatique et sanitaire – Cameroun")

    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox(
        "Choisissez un modèle",
        [
            "Indice de Stress Thermique (HSI)",
            "Indice de Qualité de l'Air (IQA)",
            "Risque d'Inondation (FRI)",
            "Indice de Sécheresse (SPEI)",
            "Risque Vectoriel (VRI)",
            "Potentiel Solaire (SEP)",
            "Indice Composite de Risque Sanitaire (CHRI)",
            "Évapotranspiration (ETO)",
            "Classification du code météo (Weather Code)",
            "Risque d'Incendie (Fire Risk)"
        ]
    )

    pages = {
        "Indice de Stress Thermique (HSI)": ("hsi", show_hsi_page),
        "Indice de Qualité de l'Air (IQA)": ("iqa", show_iqa_page),
        "Risque d'Inondation (FRI)": ("fri", show_fri_page),
        "Indice de Sécheresse (SPEI)": ("spei", show_spei_page),
        "Risque Vectoriel (VRI)": ("vri", show_vri_page),
        "Potentiel Solaire (SEP)": ("sep", show_sep_page),
        "Indice Composite de Risque Sanitaire (CHRI)": ("chri", show_chri_page),
        "Évapotranspiration (ETO)": ("eto", show_eto_page),
        "Classification du code météo (Weather Code)": ("weather_code", show_weather_code_page),
        "Risque d'Incendie (Fire Risk)": ("fire_risk", show_fire_risk_page),
    }

    model_key, page_func = pages[option]
    model, scaler, features = load_model_and_scaler(model_key)
    page_func(model, scaler, features, model_key)

if __name__ == "__main__":
    main()