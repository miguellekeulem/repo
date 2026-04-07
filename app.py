import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import matplotlib.pyplot as plt
import shap

# ========================= UTILITAIRES =========================

@st.cache_resource
def load_model_and_scaler(model_name):
    """
    Charge le modèle, le scaler (si présent) et la liste des features (si présente).
    """
    model_path = f"{model_name}_model.pkl"
    scaler_path = f"{model_name}_scaler.pkl"
    features_path = f"{model_name}_features.pkl"

    if not os.path.exists(model_path):
        st.error(f"Modèle introuvable : {model_path}")
        st.stop()
    model = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        st.warning(f"Scaler non trouvé pour {model_name}. Les entrées seront utilisées brutes.")

    features = None
    if os.path.exists(features_path):
        features = joblib.load(features_path)
    else:
        st.info(f"Liste des features non trouvée pour {model_name}. La page définira l'ordre.")

    return model, scaler, features

def compute_is_dry_season(month):
    """Retourne 1 si le mois est en saison sèche (novembre à mars)."""
    return 1 if month in [11, 12, 1, 2, 3] else 0

# ===================== PAGES DES MODÈLES =====================

def show_hsi_page(model, scaler, features, model_key):
    st.header("🌡️ Indice de Stress Thermique (HSI)")
    st.markdown("Page à implémenter selon le notebook HSI.")
    # À compléter avec le formulaire et les graphiques

def show_iqa_page(model, scaler, features, model_key):
    st.header("🌫️ Indice de Qualité de l'Air (IQA)")
    st.markdown("Page à implémenter selon le notebook IQA.")

def show_fri_page(model, scaler, features, model_key):
    st.header("🌊 Risque d'Inondation (FRI)")
    st.markdown("Page à implémenter selon le notebook FRI.")

def show_spei_page(model, scaler, features, model_key):
    st.header("💧 Indice de Sécheresse (SPEI)")
    st.markdown("Page à implémenter selon le notebook SPEI.")

def show_vri_page(model, scaler, features, model_key):
    st.header("🦟 Indice de Risque Vectoriel (VRI)")
    st.markdown("""
    Le **VRI** évalue le risque de prolifération des moustiques vecteurs de maladies (paludisme, dengue, etc.).  
    Il dépend de conditions optimales de température (autour de 25°C), d'humidité (autour de 70%) et de la présence d'eau stagnante (précipitations).  
    La saison sèche réduit le risque.

    **Formule utilisée** :  
    \[
    VRI = T_{\mathrm{opt}} \times HR_{\mathrm{opt}} \times P_{\mathrm{opt}} \times (1 - S)
    \]
    où \(T_{\mathrm{opt}}\) et \(HR_{\mathrm{opt}}\) sont des fonctions gaussiennes, \(P_{\mathrm{opt}}\) dépend des précipitations, et \(S\) vaut 1 en saison sèche.
    """)

    # Importance des features
    if hasattr(model, 'feature_importances_') and features is not None:
        if st.checkbox("Afficher l'importance des variables"):
            imp_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                         title="Importance des variables dans le modèle VRI")
            st.plotly_chart(fig)
    else:
        st.info("L'importance des variables n'est pas disponible pour ce modèle.")

    # SHAP
    st.subheader("🔍 Interprétabilité (SHAP)")
    if st.button("Générer un graphique SHAP (sur données exemple)"):
        with st.spinner("Calcul en cours..."):
            if features is None:
                st.error("Liste des features non disponible.")
            else:
                n_samples = 100
                X_sample = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
                if scaler is not None:
                    X_sample_scaled = scaler.transform(X_sample)
                else:
                    X_sample_scaled = X_sample.values
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample_scaled[:50])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample_scaled[:50], feature_names=features, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP non disponible : {e}")

    # Formulaire de prédiction
    st.subheader("📝 Prédiction personnalisée du VRI")
    default_features_order = [
        'temperature_2m_mean',
        'relative_humidity_2m_mean',
        'precipitation_sum',
        'precipitation_hours',
        'is_dry_season'
    ]

    with st.form(key="vri_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("🌡️ Température moyenne (°C)", value=25.0, step=0.5)
            humidity = st.slider("💧 Humidité relative moyenne (%)", 0, 100, 70)
            precip = st.number_input("☔ Précipitations totales (mm)", value=5.0, step=1.0)
        with col2:
            precip_hours = st.number_input("⏱️ Heures de précipitations (heures)", value=2.0, step=0.5, min_value=0.0, max_value=24.0)
            month = st.selectbox("📅 Mois", list(range(1,13)), index=5)
            is_dry = compute_is_dry_season(month)
            st.write(f"Saison : {'**Sèche** (risque réduit)' if is_dry else '**Pluvieuse** (risque accru)'}")

        submitted = st.form_submit_button("Prédire le VRI")
        if submitted:
            input_dict = {
                'temperature_2m_mean': temp,
                'relative_humidity_2m_mean': humidity,
                'precipitation_sum': precip,
                'precipitation_hours': precip_hours,
                'is_dry_season': is_dry
            }
            input_df = pd.DataFrame([input_dict])
            if features is not None:
                missing = set(features) - set(input_df.columns)
                if missing:
                    st.error(f"Features manquantes : {missing}")
                    st.stop()
                input_df = input_df[features]
            else:
                input_df = input_df[default_features_order]
                st.info("Ordre par défaut des features utilisé.")

            if scaler is not None:
                try:
                    input_scaled = scaler.transform(input_df)
                except Exception as e:
                    st.error(f"Erreur normalisation : {e}")
                    st.stop()
            else:
                input_scaled = input_df.values

            try:
                pred = model.predict(input_scaled)[0]
                pred_clamped = max(0.0, min(1.0, pred))
                st.metric("Indice de Risque Vectoriel (VRI)", f"{pred_clamped:.3f}",
                          delta="Élevé" if pred_clamped > 0.6 else "Modéré" if pred_clamped > 0.3 else "Faible")
                if pred_clamped < 0.2:
                    interpretation = "✅ **Risque très faible**"
                elif pred_clamped < 0.4:
                    interpretation = "⚠️ **Risque faible**"
                elif pred_clamped < 0.6:
                    interpretation = "⚠️ **Risque modéré**"
                elif pred_clamped < 0.8:
                    interpretation = "🔥 **Risque élevé**"
                else:
                    interpretation = "🚨 **Risque très élevé**"
                st.markdown(f"**Interprétation** : {interpretation}")
                st.progress(pred_clamped, text="Niveau de risque")
            except Exception as e:
                st.error(f"Erreur prédiction : {e}")

def show_sep_page(model, scaler, features, model_key):
    st.header("☀️ Potentiel Solaire (SEP)")
    st.markdown("Page à implémenter selon le notebook SEP.")

def show_chri_page(model, scaler, features, model_key):
    st.header("🏥 Indice Composite de Risque Sanitaire (CHRI)")
    st.markdown("Page à implémenter selon le notebook CHRI.")

def show_eto_page(model, scaler, features, model_key):
    st.header("💨 Évapotranspiration (ETO)")
    st.markdown("Page à implémenter selon le notebook ETO.")

def show_weather_code_page(model, scaler, features, model_key):
    st.header("☁️ Classification du code météo (Weather Code)")
    st.markdown("Page à implémenter selon le notebook Weather Code.")

def show_fire_risk_page(model, scaler, features, model_key):
    st.header("🔥 Risque d'Incendie (Fire Risk)")
    st.markdown("Page à implémenter selon le notebook Fire Risk.")

# ========================= APPLICATION PRINCIPALE =========================

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
