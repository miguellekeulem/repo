import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap

def show_page(model, scaler, features, model_key):
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

    # --- Visualisations (si le modèle fournit des importances) ---
    st.subheader("📊 Analyse exploratoire")
    
    # Importance des features (si disponible)
    if hasattr(model, 'feature_importances_') and features is not None:
        if st.checkbox("Afficher l'importance des variables"):
            imp_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                         title="Importance des variables dans le modèle VRI",
                         labels={'importance': 'Importance', 'feature': 'Variable'})
            st.plotly_chart(fig)
    else:
        st.info("L'importance des variables n'est pas disponible pour ce modèle.")

    # SHAP (optionnel, peut être long)
    st.subheader("🔍 Interprétabilité (SHAP)")
    if st.button("Générer un graphique SHAP (sur données exemple)"):
        with st.spinner("Calcul en cours (peut prendre quelques secondes)..."):
            # Création d'un petit échantillon aléatoire représentatif
            if features is None:
                st.error("Liste des features non disponible. Impossible de générer SHAP.")
            else:
                n_samples = 100
                X_sample = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
                # Appliquer le scaler si disponible
                if scaler is not None:
                    X_sample_scaled = scaler.transform(X_sample)
                else:
                    X_sample_scaled = X_sample.values
                # SHAP nécessite un modèle avec predict (TreeExplainer pour arbres)
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample_scaled[:50])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample_scaled[:50], feature_names=features, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Impossible de générer SHAP : {e}")

    # --- Formulaire de prédiction ---
    st.subheader("📝 Prédiction personnalisée du VRI")
    st.markdown("Renseignez les conditions météorologiques pour obtenir l'indice de risque vectoriel (0 = risque nul, 1 = risque maximal).")

    # Définition des features attendues par le modèle (d'après la documentation)
    # On suppose que le modèle a été entraîné avec ces colonnes (dans l'ordre si features est None)
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
            temp = st.number_input("🌡️ Température moyenne (°C)", value=25.0, step=0.5,
                                   help="Température idéale autour de 25°C pour les moustiques.")
            humidity = st.slider("💧 Humidité relative moyenne (%)", 0, 100, 70,
                                 help="Humidité optimale ~70%.")
            precip = st.number_input("☔ Précipitations totales (mm)", value=5.0, step=1.0,
                                     help="Cumul de pluie sur la période.")
        with col2:
            precip_hours = st.number_input("⏱️ Heures de précipitations (heures)", value=2.0, step=0.5,
                                           min_value=0.0, max_value=24.0,
                                           help="Nombre d'heures avec précipitations dans la journée.")
            month = st.selectbox("📅 Mois", list(range(1,13)), index=5,
                                 help="La saison sèche (novembre à mars) réduit le risque.")
            # Calcul automatique de is_dry_season
            is_dry = 1 if month in [11,12,1,2,3] else 0
            st.write(f"Saison : {'**Sèche** (risque réduit)' if is_dry else '**Pluvieuse** (risque accru)'}")

        submitted = st.form_submit_button("Prédire le VRI")

        if submitted:
            # Construction du dictionnaire d'entrée
            input_dict = {
                'temperature_2m_mean': temp,
                'relative_humidity_2m_mean': humidity,
                'precipitation_sum': precip,
                'precipitation_hours': precip_hours,
                'is_dry_season': is_dry
            }
            # Conversion en DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Réordonner les colonnes si la liste des features est fournie
            if features is not None:
                # Vérifier que toutes les features sont présentes
                missing = set(features) - set(input_df.columns)
                if missing:
                    st.error(f"Features manquantes dans le formulaire : {missing}")
                    st.stop()
                input_df = input_df[features]
            else:
                # Utiliser l'ordre par défaut (attention : doit correspondre à l'entraînement)
                input_df = input_df[default_features_order]
                st.info("Utilisation de l'ordre par défaut des features (modèle non configuré avec une liste explicite).")
            
            # Appliquer le scaler si disponible
            if scaler is not None:
                try:
                    input_scaled = scaler.transform(input_df)
                except Exception as e:
                    st.error(f"Erreur lors de la normalisation : {e}. Vérifiez les colonnes.")
                    st.stop()
            else:
                input_scaled = input_df.values
            
            # Prédiction
            try:
                pred = model.predict(input_scaled)[0]
                # Le VRI est normalement entre 0 et 1, on le clamp pour l'affichage
                pred_clamped = max(0.0, min(1.0, pred))
                st.metric("Indice de Risque Vectoriel (VRI)", f"{pred_clamped:.3f}",
                          delta="Élevé" if pred_clamped > 0.6 else "Modéré" if pred_clamped > 0.3 else "Faible")
                
                # Interprétation qualitative
                if pred_clamped < 0.2:
                    interpretation = "✅ **Risque très faible** – Conditions défavorables aux moustiques."
                elif pred_clamped < 0.4:
                    interpretation = "⚠️ **Risque faible** – Surveillance recommandée."
                elif pred_clamped < 0.6:
                    interpretation = "⚠️ **Risque modéré** – Mesures de prévention (moustiquaires, répulsifs)."
                elif pred_clamped < 0.8:
                    interpretation = "🔥 **Risque élevé** – Forte probabilité de prolifération."
                else:
                    interpretation = "🚨 **Risque très élevé** – Épidémie potentielle, actions urgentes."
                
                st.markdown(f"**Interprétation** : {interpretation}")
                
                # Affichage d'une jauge simple
                st.progress(pred_clamped, text="Niveau de risque")
                
                # Explication basée sur les entrées
                st.markdown("#### 🔎 Facteurs contributifs :")
                if temp < 20 or temp > 30:
                    st.write("- Température éloignée de l'optimum (25°C) → risque réduit.")
                else:
                    st.write("- Température proche de l'optimum → favorable aux moustiques.")
                if humidity < 55 or humidity > 85:
                    st.write("- Humidité éloignée de 70% → moins favorable.")
                else:
                    st.write("- Humidité dans la zone optimale → favorable.")
                if precip < 1 and precip_hours < 1:
                    st.write("- Très peu de précipitations → réduction des sites de ponte.")
                else:
                    st.write("- Présence d'eau stagnante (précipitations) → augmentation du risque.")
                if is_dry:
                    st.write("- Saison sèche → diminution naturelle du risque.")
                else:
                    st.write("- Saison des pluies → risque accru.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")