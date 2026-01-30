import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Prédiction de Churn Bancaire", layout="wide")

# === Chargement des ressources ===
@st.cache_resource
def load_model():
    return joblib.load(Path("models") / "best_model.pkl")

@st.cache_data
def get_preprocessing_stats():
    df = pd.read_csv(Path("data") / "preprocessed_data.csv")
    salary_by_geo = df.groupby('Geography')['EstimatedSalary'].mean().to_dict()
    balance_mean = df['Balance'].mean()
    return salary_by_geo, balance_mean

@st.cache_data
def load_raw_data():
    """Charger les données originales pour l'analyse"""
    try:
        return pd.read_csv(Path("data") / "preprocessed_data.csv")
    except:
        return None

try:
    model = load_model()
    salary_by_geo, balance_mean = get_preprocessing_stats()
    df_analysis = load_raw_data()
except Exception as e:
    st.error(f"Erreur de chargement : {str(e)}")
    st.stop()

# === Interface avec onglets ===
tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Analyse"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 : PRÉDICTION
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.title("Prédiction de Churn Bancaire")
    st.markdown("Entrez les informations du client pour prédire le risque de départ.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("👤 Informations client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Score de crédit", 300, 850, 650, help="Entre 300 et 850")
        age = st.number_input("Âge", 18, 100, 35)
        tenure = st.number_input("Ancienneté (ans)", 0, 40, 5, help="Nombre d'années avec la banque")
        balance = st.number_input("Solde compte ($)", 0.0, 300000.0, 100000.0)
    
    with col2:
        salary = st.number_input("Salaire annuel ($)", 0.0, 500000.0, 50000.0)
        num_products = st.number_input("Nombre de produits", 1, 4, 2, help="Comptes, cartes, assurances...")
        has_credit_card = st.selectbox("Carte de crédit", ["Oui", "Non"])
        is_active = st.selectbox("Membre actif", ["Oui", "Non"], help="Utilise régulièrement ses services")
        gender = st.selectbox("Genre", ["Male", "Female"])
        geography = st.selectbox("Pays", ["France", "Spain", "Germany"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bouton parfaitement centré
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        predict_btn = st.button("🚀 Prédire le risque de churn", type="primary", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if predict_btn:
        with st.spinner("Analyse en cours..."):
            # Conversion valeurs
            HasCrCard = 1 if has_credit_card == "Oui" else 0
            IsActiveMember = 1 if is_active == "Oui" else 0
            safe_age = max(age, 1)
            
            # Features dérivées
            Ratio_Balance_Salary = balance / max(salary, 1)
            Ratio_Balance_Age = balance / safe_age
            Ratio_Salary_Age = salary / safe_age
            Engagement_Score = IsActiveMember + num_products + HasCrCard
            Ratio_Products_Age = num_products / safe_age
            geo_mean_salary = salary_by_geo.get(geography, salary)
            Relative_Salary = salary / max(geo_mean_salary, 1)
            Ratio_CreditScore_Age = credit_score / safe_age
            Zero_Balance_HasCrCard = int((balance == 0) and HasCrCard)
            Low_Balance_Active = int((balance < balance_mean) and IsActiveMember)
            Active_HasCrCard = IsActiveMember * HasCrCard
            Log_Salary = np.log1p(salary)
            
            # Préparation données
            features = {
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': HasCrCard,
                'IsActiveMember': IsActiveMember,
                'EstimatedSalary': salary,
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
                proba = float(model.predict_proba(input_df)[0][1])
                risk_level = (
                    "🔴 ÉLEVÉ" if proba > 0.7 
                    else "🟠 MOYEN" if proba > 0.3 
                    else "🟢 FAIBLE"
                )
                
                # Affichage résultat
                st.success("✅ Prédiction effectuée avec succès !")
                st.markdown(f"### Résultat : {risk_level}")
                st.progress(min(proba, 1.0))
                st.metric("Probabilité de churn", f"{proba:.1%}")
                st.caption("Seuils : 🔴 >70% | 🟠 30-70% | 🟢 <30%")
                
            except Exception as e:
                st.error("❌ Erreur lors de la prédiction")
                with st.expander("Détails techniques"):
                    st.code(str(e))
                    st.write("**Colonnes reçues :**", list(input_df.columns))
                    if hasattr(model, 'feature_names_in_'):
                        st.write("**Colonnes attendues :**", model.feature_names_in_.tolist())

# ═══════════════════════════════════════════════════════════════
# TAB 2 : ANALYSE
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Analyse des Données Clients")
    
    if df_analysis is None:
        st.warning("⚠️ Données d'analyse non disponibles.")
    else:
        # Filtres rapides
        st.markdown("### 🔍 Filtres")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            geo_filter = st.multiselect(
                "Pays",
                options=df_analysis['Geography'].unique(),
                default=df_analysis['Geography'].unique().tolist()
            )
        
        with col_f2:
            gender_filter = st.multiselect(
                "Genre",
                options=df_analysis['Gender'].unique(),
                default=df_analysis['Gender'].unique().tolist()
            )
        
        with col_f3:
            products_filter = st.multiselect(
                "Nombre de produits",
                options=sorted(df_analysis['NumOfProducts'].unique()),
                default=sorted(df_analysis['NumOfProducts'].unique())
            )
        
        # Appliquer les filtres
        df_filtered = df_analysis[
            (df_analysis['Geography'].isin(geo_filter)) &
            (df_analysis['Gender'].isin(gender_filter)) &
            (df_analysis['NumOfProducts'].isin(products_filter))
        ]
        
        st.markdown(f"**Données filtrées :** {len(df_filtered):,} clients sur {len(df_analysis):,}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Métriques principales
        st.markdown("### 📈 Métriques Clés")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            churn_rate = df_filtered['Exited'].mean() * 100 if 'Exited' in df_filtered.columns else 0
            st.metric("Taux de churn", f"{churn_rate:.1f}%")
        
        with col_m2:
            avg_balance = df_filtered['Balance'].mean()
            st.metric("Solde moyen", f"${avg_balance:,.0f}")
        
        with col_m3:
            avg_salary = df_filtered['EstimatedSalary'].mean()
            st.metric("Salaire moyen", f"${avg_salary:,.0f}")
        
        with col_m4:
            avg_age = df_filtered['Age'].mean()
            st.metric("Âge moyen", f"{avg_age:.0f} ans")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Graphiques sans matplotlib (Streamlit natif)
        st.markdown("### 📊 Visualisations")
        
        # Graphique 1 : Churn par pays
        if 'Exited' in df_filtered.columns:
            st.markdown("**Taux de churn par pays**")
            churn_by_geo = df_filtered.groupby('Geography')['Exited'].mean() * 100
            st.bar_chart(churn_by_geo, color="#FF6B6B")
        
        # Graphique 2 : Distribution par âge
        st.markdown("**Distribution des clients par âge**")
        age_hist = df_filtered['Age'].value_counts().sort_index()
        st.bar_chart(age_hist, color="#4ECDC4")
        
        # Graphique 3 : Solde vs Salaire (scatter)
        st.markdown("**Solde en fonction du salaire**")
        scatter_data = df_filtered[['EstimatedSalary', 'Balance']].rename(
            columns={'EstimatedSalary': 'Salaire', 'Balance': 'Solde'}
        ).head(500)  # Limite pour performance
        st.scatter_chart(scatter_data, x='Salaire', y='Solde', size=50, color='#95E1D3')
        
        # Graphique 4 : Churn par nombre de produits
        if 'Exited' in df_filtered.columns:
            st.markdown("**Taux de churn par nombre de produits**")
            churn_by_products = df_filtered.groupby('NumOfProducts')['Exited'].mean() * 100
            st.bar_chart(churn_by_products, color="#FFA07A")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Données brutes optionnelles
        with st.expander("📋 Voir un échantillon des données"):
            st.dataframe(df_filtered.head(15), use_container_width=True)