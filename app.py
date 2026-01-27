import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="🏦 Churn Prediction", layout="centered")
st.title("🏦 Prédiction de Churn Bancaire")

@st.cache_resource
def load_model():
    return joblib.load(Path("models") / "best_model.pkl")

@st.cache_data
def get_preprocessing_stats():
    """Récupère les statistiques nécessaires pour Relative_Salary et Low_Balance_Active"""
    df = pd.read_csv(Path("data") / "preprocessed_data.csv")
    salary_by_geo = df.groupby('Geography')['EstimatedSalary'].mean().to_dict()
    balance_mean = df['Balance'].mean()
    return salary_by_geo, balance_mean

model = load_model()
salary_by_geo, balance_mean = get_preprocessing_stats()

# === Formulaire utilisateur ===
st.subheader("👤 Informations du client")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Score de crédit", 300, 850, 650)
    age = st.number_input("Âge", 18, 100, 35)
    tenure = st.number_input("Ancienneté (ans)", 0, 40, 5)
    balance = st.number_input("Solde ($)", 0.0, 300000.0, 100000.0)

with col2:
    salary = st.number_input("Salaire annuel ($)", 0.0, 500000.0, 50000.0)
    num_products = st.number_input("Nb produits", 1, 4, 2)
    has_credit_card = st.selectbox("Carte de crédit", ["Oui", "Non"])
    is_active = st.selectbox("Compte actif", ["Oui", "Non"])
    gender = st.selectbox("Genre", ["Male", "Female"])
    geography = st.selectbox("Pays", ["France", "Spain", "Germany"])

# === Préparation des features (exactement comme dans le notebook) ===
if st.button("🚀 Prédire", type="primary", use_container_width=True):
    # Valeurs binaires
    HasCrCard = 1 if has_credit_card == "Oui" else 0
    IsActiveMember = 1 if is_active == "Oui" else 0
    
    # Éviter division par zéro
    safe_age = max(age, 1)
    
    # Calcul des features dérivées (exactement comme dans votre notebook)
    Ratio_Balance_Salary = balance / max(salary, 1)
    Ratio_Balance_Age = balance / safe_age
    Ratio_Salary_Age = salary / safe_age
    Engagement_Score = IsActiveMember + num_products + HasCrCard  # CORRIGÉ
    Ratio_Products_Age = num_products / safe_age
    geo_mean_salary = salary_by_geo.get(geography, salary)
    Relative_Salary = salary / max(geo_mean_salary, 1)
    Ratio_CreditScore_Age = credit_score / safe_age
    Zero_Balance_HasCrCard = int((balance == 0) and (HasCrCard == 1))
    Low_Balance_Active = int((balance < balance_mean) and (IsActiveMember == 1))
    Active_HasCrCard = IsActiveMember * HasCrCard
    Log_Salary = np.log1p(salary)  # log1p = log(1+x)

    # 🔑 CORRECTION 1 : Noms de colonnes EN ANGLAIS (exactement comme le modèle)
    features = {
        'CreditScore': credit_score,
        'Geography': geography,          # ← Pas 'Géographie'
        'Gender': gender,                # ← Pas 'Genre'
        'Age': age,
        'Tenure': tenure,                # ← Pas 'Titularisation'
        'Balance': balance,              # ← Pas 'Équilibre'
        'NumOfProducts': num_products,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,  # ← Pas 'IsMembreActif'
        'EstimatedSalary': salary,       # ← Pas 'Salaire estimé'
        'Ratio_Balance_Salary': Ratio_Balance_Salary,
        'Ratio_Balance_Age': Ratio_Balance_Age,
        'Ratio_Salary_Age': Ratio_Salary_Age,
        'Engagement_Score': Engagement_Score,
        'Ratio_Products_Age': Ratio_Products_Age,
        'Relative_Salary': Relative_Salary,
        'Ratio_CreditScore_Age': Ratio_CreditScore_Age,
        'Zero_Balance_HasCrCard': Zero_Balance_HasCrCard,
        'Low_Balance_Active': Low_Balance_Active,
        'Active_HasCrCard': Active_HasCrCard,
        'Log_Salary': Log_Salary
    }

    input_df = pd.DataFrame([features])
    
    try:
        # 🔑 CORRECTION 2 : Conversion en float Python standard (pas numpy.float32)
        proba = float(model.predict_proba(input_df)[0][1])
        
        risk = "🔴 ÉLEVÉ" if proba > 0.7 else "🟡 MOYEN" if proba > 0.3 else "🟢 FAIBLE"
        
        st.markdown(f"### Résultat : {risk}")
        st.progress(min(proba, 1.0))  # ✅ Compatible avec float standard
        st.metric("Probabilité de churn", f"{proba:.1%}")
        
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.write("Colonnes envoyées :", list(input_df.columns))
        st.write("Colonnes attendues :", model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else "Non disponibles")