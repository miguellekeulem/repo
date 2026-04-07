import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt

def show_page(model, scaler, features):
    st.header("🌡️ Indice de Stress Thermique (HSI)")
    st.markdown("""
    Le HSI combine température et humidité pour évaluer la sensation thermique. 
    Plus l’indice est élevé, plus le risque sanitaire (coup de chaleur, déshydratation) est grand.
    """)
    
    # --- Visualisations (à pré-calculer ou générer à partir d'un échantillon) ---
    st.subheader("📊 Analyse exploratoire")
    # Exemple : importance des features (à stocker dans un fichier ou recalculer)
    if st.checkbox("Afficher l'importance des features"):
        # Si le modèle a l'attribut feature_importances_
        if hasattr(model, 'feature_importances_'):
            imp_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig = px.bar(imp_df, x='importance', y='feature', orientation='h', title="Importance des variables")
            st.plotly_chart(fig)
        else:
            st.info("Importance non disponible pour ce modèle.")
    
    # SHAP summary (optionnel)
    st.subheader("🔍 Interprétabilité (SHAP)")
    if st.button("Générer un graphique SHAP (sur données exemple)"):
        with st.spinner("Calcul en cours..."):
            # Ici on charge un petit échantillon de test (pré-enregistré)
            # Pour simplifier, on peut utiliser un jeu de données dummy représentatif
            X_sample = pd.DataFrame(np.random.randn(100, len(features)), columns=features)
            X_scaled = scaler.transform(X_sample)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled[:50])
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_scaled[:50], feature_names=features, show=False)
            st.pyplot(fig)
    
    # --- Formulaire de prédiction ---
    st.subheader("📝 Prédiction personnalisée")
    with st.form(key="hsi_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp_max = st.number_input("Température maximale (°C)", value=30.0, step=0.5)
            humidity_mean = st.slider("Humidité relative moyenne (%)", 0, 100, 70)
            wind_speed = st.number_input("Vitesse du vent (m/s)", value=2.0, step=0.5)
        with col2:
            precip = st.number_input("Précipitations (mm)", value=0.0, step=1.0)
            radiation = st.number_input("Rayonnement solaire (W/m²)", value=200.0, step=10.0)
            month = st.selectbox("Mois", list(range(1,13)), index=5)  # juin par défaut
        
        # Construction du DataFrame d'entrée dans l'ordre exact des features
        input_dict = {
            'temperature_2m_mean': temp_max,  # selon les features du modèle
            'relative_humidity_2m_mean': humidity_mean,
            'wind_speed_10m_max': wind_speed,
            'precipitation_sum': precip,
            'shortwave_radiation_sum': radiation,
            'month': month,
            # Ajouter d'autres features comme sin/cos jour, rolling, etc. si nécessaires
        }
        # Compléter avec des valeurs par défaut pour les features manquantes (ex: lags)
        # Idéalement, les features sont exactement celles attendues.
        input_df = pd.DataFrame([input_dict])[features]  # réordonne
        input_scaled = scaler.transform(input_df)
        
        submitted = st.form_submit_button("Prédire le HSI")
        if submitted:
            pred = model.predict(input_scaled)[0]
            st.metric("Indice de Stress Thermique (HSI)", f"{pred:.1f} °C", 
                      delta="Élevé" if pred > 32 else "Modéré")
            st.markdown(f"**Interprétation** : {interpret_hsi(pred)}")
    
    # Fonction d'interprétation
    def interpret_hsi(value):
        if value < 27:
            return "Aucun stress thermique."
        elif value < 32:
            return "Stress thermique modéré (prudence)."
        elif value < 41:
            return "Stress thermique fort (danger potentiel)."
        else:
            return "Stress thermique extrême (danger immédiat)."