# import joblib
# import streamlit as st
# import pandas as pd
# import numpy as np

# @st.cache_resource
# def load_model_and_scaler(model_name):
#     """Charge le modèle et le scaler pour un nom donné (ex: 'hsi')"""
#     model = joblib.load(f"models/{model_name}_model.pkl")
#     scaler = joblib.load(f"models/{model_name}_scaler.pkl")
#     features = joblib.load(f"models/{model_name}_features.pkl")  # liste des features
#     return model, scaler, features

# def compute_is_dry_season(month):
#     """Fonction utilitaire (à adapter selon la définition)"""
#     return 1 if month in [11,12,1,2,3] else 0 




import joblib
import streamlit as st
import os

@st.cache_resource
def load_model_and_scaler(model_name):
    """
    Charge le modèle, le scaler (si présent) et la liste des features (si présente).
    
    Args:
        model_name (str): Nom technique du modèle (ex: 'hsi', 'iqa')
    
    Returns:
        tuple: (model, scaler, features)
            - model: objet modèle chargé
            - scaler: scaler ou None si fichier absent
            - features: liste des features ou None si fichier absent
    """
    model_path = f"models/{model_name}_model.pkl"
    scaler_path = f"models/{model_name}_scaler.pkl"
    features_path = f"models/{model_name}_features.pkl"
    
    # Chargement du modèle (obligatoire)
    if not os.path.exists(model_path):
        st.error(f"Modèle introuvable : {model_path}")
        st.stop()
    model = joblib.load(model_path)
    
    # Chargement du scaler (optionnel)
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        st.warning(f"Scaler non trouvé pour {model_name}. Les entrées seront utilisées brutes.")
    
    # Chargement des features (optionnel)
    features = None
    if os.path.exists(features_path):
        features = joblib.load(features_path)
    else:
        st.info(f"Liste des features non trouvée pour {model_name}. La page devra définir l'ordre des colonnes.")
    
    return model, scaler, features

def compute_is_dry_season(month):
    """Retourne 1 si le mois est en saison sèche (novembre à mars)."""
    return 1 if month in [11, 12, 1, 2, 3] else 0