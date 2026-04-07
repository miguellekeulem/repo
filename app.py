# import streamlit as st
# from pages_functions import (
#     hsi_page, iqa_page, fri_page, spei_page, vri_page,
#     sep_page, chri_page, eto_page, weather_code_page, fire_risk_page
# )
# from utils import load_model_and_scaler

# st.set_page_config(page_title="IndabaX Climate Dashboard", layout="wide")
# st.title("🌍 IA pour la résilience climatique et sanitaire – Cameroun")

# # Menu déroulant dans la sidebar
# st.sidebar.header("Navigation")
# option = st.sidebar.selectbox(
#     "Choisissez un modèle",
#     [
#         "Indice de Stress Thermique (HSI)",
#         "Indice de Qualité de l'Air (IQA)",
#         "Risque d'Inondation (FRI)",
#         "Indice de Sécheresse (SPEI)",
#         "Risque Vectoriel (VRI)",
#         "Potentiel Solaire (SEP)",
#         "Indice Composite de Risque Sanitaire (CHRI)",
#         "Évapotranspiration (ETO)",
#         "Classification du code météo (Weather Code)",
#         "Risque d'Incendie (Fire Risk)"
#     ]
# )

# # Dictionnaire associant le libellé au nom technique et à la fonction d'affichage
# pages = {
#     "Indice de Stress Thermique (HSI)": ("hsi", hsi_page.show_page),
#     "Indice de Qualité de l'Air (IQA)": ("iqa", iqa_page.show_page),
#     "Risque d'Inondation (FRI)": ("fri", fri_page.show_page),
#     "Indice de Sécheresse (SPEI)": ("spei", spei_page.show_page),
#     "Risque Vectoriel (VRI)": ("vri", vri_page.show_page),
#     "Potentiel Solaire (SEP)": ("sep", sep_page.show_page),
#     "Indice Composite de Risque Sanitaire (CHRI)": ("chri", chri_page.show_page),
#     "Évapotranspiration (ETO)": ("eto", eto_page.show_page),
#     "Classification du code météo (Weather Code)": ("weather_code", weather_code_page.show_page),
#     "Risque d'Incendie (Fire Risk)": ("fire_risk", fire_risk_page.show_page),
# }

# model_key, page_func = pages[option]
# # Chargement des artefacts (mise en cache automatique)
# model, scaler, features = load_model_and_scaler(model_key)

# # Appel de la fonction d'affichage spécifique
# page_func(model, scaler, features)




import streamlit as st
from pages_functions import (
    hsi_page, iqa_page, fri_page, spei_page, vri_page,
    sep_page, chri_page, eto_page, weather_code_page, fire_risk_page
)
from utils import load_model_and_scaler

st.set_page_config(page_title="IndabaX Climate Dashboard", layout="wide")
st.title("🌍 IA pour la résilience climatique et sanitaire – Cameroun")

# Menu déroulant dans la sidebar
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Choisissez un modèle",
    [
        # "Indice de Stress Thermique (HSI)",
        # "Indice de Qualité de l'Air (IQA)",
        # "Risque d'Inondation (FRI)",
        # "Indice de Sécheresse (SPEI)",
        "Risque Vectoriel (VRI)",
        # "Potentiel Solaire (SEP)",
        # "Indice Composite de Risque Sanitaire (CHRI)",
        # "Évapotranspiration (ETO)",
        #"Classification du code météo (Weather Code)",
        # "Risque d'Incendie (Fire Risk)"
    ]
)

# Dictionnaire associant le libellé au nom technique et à la fonction d'affichage
pages = {
    # "Indice de Stress Thermique (HSI)": ("hsi", hsi_page.show_page),
    # "Indice de Qualité de l'Air (IQA)": ("iqa", iqa_page.show_page),
    # "Risque d'Inondation (FRI)": ("fri", fri_page.show_page),
    # "Indice de Sécheresse (SPEI)": ("spei", spei_page.show_page),
    "Risque Vectoriel (VRI)": ("vri", vri_page.show_page),
    # "Potentiel Solaire (SEP)": ("sep", sep_page.show_page),
    # "Indice Composite de Risque Sanitaire (CHRI)": ("chri", chri_page.show_page),
    # "Évapotranspiration (ETO)": ("eto", eto_page.show_page),
    #"Classification du code météo (Weather Code)": ("weather_code", weather_code_page.show_page),
    # "Risque d'Incendie (Fire Risk)": ("fire_risk", fire_risk_page.show_page),
}

model_key, page_func = pages[option]

# Chargement des artefacts (le scaler et les features peuvent être None)
model, scaler, features = load_model_and_scaler(model_key)

# Appel de la fonction d'affichage spécifique
# On passe également le nom technique (model_key) pour permettre à la page de connaître son identifiant
page_func(model, scaler, features, model_key)