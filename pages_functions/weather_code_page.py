import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix

# Dictionnaire des principaux codes WMO (0-99) avec leur signification
WMO_DESCRIPTION = {
    0: "Ciel dégagé",
    1: "Principalement dégagé",
    2: "Partiellement nuageux",
    3: "Nuageux",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine fine",
    53: "Bruine modérée",
    55: "Bruine dense",
    56: "Bruine glaciale fine",
    57: "Bruine glaciale dense",
    61: "Pluie faible",
    63: "Pluie modérée",
    65: "Pluie forte",
    66: "Pluie glaciale faible",
    67: "Pluie glaciale forte",
    71: "Neige faible",
    73: "Neige modérée",
    75: "Neige forte",
    77: "Grains de neige",
    80: "Averse de pluie faible",
    81: "Averse de pluie modérée",
    82: "Averse de pluie forte",
    85: "Averse de neige faible",
    86: "Averse de neige forte",
    95: "Orage",
    96: "Orage avec grêle faible",
    99: "Orage avec grêle forte"
}

def show_page(model, scaler, features, model_key):
    st.header("☁️ Classification du code météo (Weather Code)")
    st.markdown("""
    Ce modèle prédit le code WMO (Organisation Météorologique Mondiale) à partir des conditions météorologiques observées.
    Le code décrit le phénomène météo dominant (pluie, orage, brouillard, etc.). 
    Cette prédiction est utile pour les systèmes d'alerte automatique.
    """)

    # --- Visualisations et interprétabilité ---
    st.subheader("📊 Analyse du modèle")

    # Importance des features (si disponible)
    if hasattr(model, 'feature_importances_') and features is not None:
        with st.expander("📈 Importance des variables", expanded=False):
            imp_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig = px.bar(imp_df.head(15), x='importance', y='feature', orientation='h',
                         title="Top 15 des variables influentes")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("L'importance des features n'est pas disponible pour ce modèle.")

    # SHAP (optionnel, un peu lourd)
    st.subheader("🔍 Interprétabilité locale (SHAP)")
    if st.button("Afficher un exemple SHAP (sur des données synthétiques)"):
        with st.spinner("Calcul SHAP en cours... (peut prendre quelques secondes)"):
            try:
                # Génération d'un échantillon aléatoire représentatif
                if features is not None:
                    X_sample = pd.DataFrame(np.random.randn(100, len(features)), columns=features)
                else:
                    # Si pas de liste de features, on utilise 5 features fictives
                    X_sample = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
                
                # Appliquer le scaler si disponible
                if scaler is not None:
                    X_sample_scaled = scaler.transform(X_sample)
                else:
                    X_sample_scaled = X_sample.values
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample_scaled[:50])
                # Pour un classifieur, shap_values est une liste par classe. On prend la première.
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_sample_scaled[:50],
                                  feature_names=features if features else [f"f{i}" for i in range(5)],
                                  show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur SHAP : {e}. Le modèle n'est peut-être pas compatible arbre.")

    # --- Formulaire de prédiction ---
    st.subheader("📝 Prédire le code météo")

    # Déterminer les features attendues. Si on a une liste, on construit le formulaire en conséquence.
    # Sinon, on propose des features génériques.
    if features is not None:
        # Construction dynamique du formulaire basé sur la liste des features
        with st.form(key="weather_code_form"):
            st.markdown("**Entrez les valeurs des paramètres météo :**")
            input_dict = {}
            cols = st.columns(3)
            for i, feat in enumerate(features):
                # Adapter le type de widget selon le nom de la feature
                with cols[i % 3]:
                    if 'temp' in feat.lower():
                        input_dict[feat] = st.number_input(f"{feat} (°C)", value=25.0, step=0.5)
                    elif 'humidity' in feat.lower() or 'rh' in feat.lower():
                        input_dict[feat] = st.slider(f"{feat} (%)", 0, 100, 65)
                    elif 'wind' in feat.lower():
                        input_dict[feat] = st.number_input(f"{feat} (m/s)", value=2.0, step=0.5)
                    elif 'precip' in feat.lower() or 'rain' in feat.lower():
                        input_dict[feat] = st.number_input(f"{feat} (mm)", value=0.0, step=1.0)
                    elif 'radiation' in feat.lower() or 'shortwave' in feat.lower():
                        input_dict[feat] = st.number_input(f"{feat} (W/m²)", value=200.0, step=10.0)
                    elif 'month' in feat.lower():
                        input_dict[feat] = st.selectbox(f"{feat}", list(range(1,13)), index=5)
                    elif 'season' in feat.lower() or 'dry' in feat.lower():
                        input_dict[feat] = st.selectbox(f"{feat}", [0,1], format_func=lambda x: "Saison humide" if x==0 else "Saison sèche")
                    else:
                        input_dict[feat] = st.number_input(f"{feat}", value=0.0)
            submitted = st.form_submit_button("Prédire le code météo")
    else:
        # Formulaire générique (basé sur des features typiques)
        with st.form(key="weather_code_form"):
            col1, col2 = st.columns(2)
            with col1:
                temp = st.number_input("Température moyenne (°C)", value=25.0)
                humidity = st.slider("Humidité relative (%)", 0, 100, 70)
                wind = st.number_input("Vitesse du vent (m/s)", value=2.0)
            with col2:
                precip = st.number_input("Précipitations (mm)", value=0.0)
                radiation = st.number_input("Rayonnement solaire (W/m²)", value=200.0)
                month = st.selectbox("Mois", list(range(1,13)), index=5)
            submitted = st.form_submit_button("Prédire le code météo")
            # Construction du dict générique
            input_dict = {
                'temperature_2m_mean': temp,
                'relative_humidity_2m_mean': humidity,
                'wind_speed_10m_max': wind,
                'precipitation_sum': precip,
                'shortwave_radiation_sum': radiation,
                'month': month
            }

    if submitted:
        # Création du DataFrame d'entrée
        input_df = pd.DataFrame([input_dict])
        # Réordonner selon les features si disponibles
        if features is not None:
            try:
                input_df = input_df[features]
            except KeyError as e:
                st.error(f"Feature manquante dans le formulaire : {e}. Vérifiez la liste des features.")
                st.stop()
        # Normaliser si scaler existe
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Prédiction
        pred_class = model.predict(input_scaled)[0]
        # Probabilités si disponible
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            confidence = np.max(proba) * 100
        else:
            confidence = None
        
        # Affichage du résultat
        st.success(f"**Code météo prédit :** {pred_class}")
        description = WMO_DESCRIPTION.get(pred_class, "Phénomène non documenté")
        st.metric("Description", description)
        if confidence is not None:
            st.metric("Confiance", f"{confidence:.1f}%")
        
        # Interprétation supplémentaire
        with st.expander("🔎 Que signifie ce code ?"):
            if pred_class in WMO_DESCRIPTION:
                st.write(f"Code {pred_class} : {WMO_DESCRIPTION[pred_class]}")
                # Recommandations simplifiées
                if pred_class in [95,96,99]:
                    st.warning("⚠️ Risque d'orage et de grêle. Prudence à l'extérieur.")
                elif pred_class in [61,63,65,80,81,82]:
                    st.info("🌧️ Pluie annoncée. Prévoyez un parapluie.")
                elif pred_class in [71,73,75,85,86]:
                    st.info("❄️ Neige possible. Attention aux routes glissantes.")
                elif pred_class in [45,48]:
                    st.info("🌫️ Brouillard. Visibilité réduite sur les routes.")
            else:
                st.write("Code non standard. Consultez la documentation WMO.")
        
        # Afficher les features importantes pour cette prédiction (SHAP local)
        if st.button("Expliquer cette prédiction (SHAP local)"):
            with st.spinner("Calcul de l'explication locale..."):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values_local = explainer.shap_values(input_scaled)
                    # Pour la classe prédite, extraire les shap values correspondants
                    # shap_values_local est un tableau (n_classes, n_samples, n_features)
                    if len(shap_values_local.shape) == 3:
                        class_idx = list(model.classes_).index(pred_class)
                        shap_vals_class = shap_values_local[class_idx][0]
                        # Créer un graphique en barres
                        if features is not None:
                            shap_df = pd.DataFrame({'feature': features, 'SHAP': shap_vals_class})
                        else:
                            shap_df = pd.DataFrame({'feature': input_df.columns, 'SHAP': shap_vals_class})
                        shap_df = shap_df.sort_values('SHAP', key=abs, ascending=False).head(10)
                        fig = px.bar(shap_df, x='SHAP', y='feature', orientation='h',
                                     title=f"Contribution des variables pour la classe {pred_class}")
                        st.plotly_chart(fig)
                    else:
                        st.write("Format SHAP non pris en charge pour ce modèle.")
                except Exception as e:
                    st.error(f"SHAP local a échoué : {e}")

    # --- Information supplémentaire sur les codes WMO ---
    with st.expander("📖 Liste des codes WMO courants"):
        # Afficher un tableau des codes
        wmo_df = pd.DataFrame(list(WMO_DESCRIPTION.items()), columns=["Code", "Description"])
        st.dataframe(wmo_df, use_container_width=True)

    # Note sur les limitations
    st.caption("Note : La précision du modèle dépend des données d'entraînement. Les prédictions sont à usage indicatif.")