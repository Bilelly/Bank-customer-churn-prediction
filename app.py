import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Churn Prediction Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-high-risk {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #2ecc71;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal avec style
st.markdown('<h1 class="main-header">🏦 Bank Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar améliorée
with st.sidebar:
    st.title("🧭 Navigation")
    page = st.radio("Sélectionnez une page:", 
        ["📊 Dashboard", "🔮 Prédiction", "📈 Analyse EDA", "📑 Analyse Batch", "ℹ️ À propos"],
        label_visibility="collapsed")
    
    st.markdown("---")
    
    st.info("""
        ### Fonctionnalités
        - 📊 Visualisez les KPIs
        - 🔮 Prédisez le churn individuel
        - 📈 Explorez les données (EDA)
        - 📑 Prédictions en masse (CSV)
        - 💡 Comprenez les insights
    """)
    
    st.markdown("---")
    st.markdown("**👨‍💻 Développé par:** Bilal SAYOUD")
    st.markdown("**📅 Version:** 1.0.0")

# Charger les données avec gestion d'erreurs améliorée
@st.cache_data
def load_data():
    """Charge les données prétraitées avec gestion d'erreurs robuste"""
    script_dir = Path(__file__).parent.resolve()
    
    data_paths = [
        script_dir / "data" / "preprocessed_data.csv",
        script_dir / "data" / "cleaned_data.csv",
        script_dir / "data" / "brut_data.csv"
    ]
    
    for path in data_paths:
        try:
            if path.exists():
                data = pd.read_csv(path)
                st.sidebar.success(f"✅ Données chargées: {path.name}")
                return data, str(path)
        except Exception as e:
            st.sidebar.error(f"Erreur avec {path.name}: {str(e)}")
    
    st.sidebar.error("❌ Aucun fichier de données trouvé")
    return None, None

# Charger le modèle avec meilleure gestion
def load_model():
    """Charge le modèle entraîné avec validation - sans cache pour debug"""
    # Déterminer le chemin du répertoire de travail
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / "best_model.pkl"
    
    st.sidebar.info(f"🔍 Recherche du modèle à: {model_path}")
    
    try:
        if not model_path.exists():
            st.sidebar.error(f"❌ Fichier non trouvé: {model_path}")
            st.sidebar.error(f"Répertoire courant: {script_dir}")
            return None
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Validation du modèle - vérifier les méthodes requises
        if not hasattr(model, 'predict'):
            st.sidebar.error("⚠️ Le modèle chargé n'a pas de méthode predict")
            return None
        
        st.sidebar.success(f"✅ Modèle chargé avec succès! Type: {type(model).__name__}")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        import traceback
        st.sidebar.error(f"Détails: {traceback.format_exc()}")
        return None

# Fonction utilitaire pour les prédictions
def make_prediction(model, input_data):
    """Effectue une prédiction avec gestion d'erreurs robuste"""
    try:
        # Essayer predict_proba d'abord (pour probabilités)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data)
            # Vérifier le format de la sortie
            if isinstance(proba, np.ndarray):
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    # Retourner la probabilité de la classe positive (classe 1)
                    return float(proba[0][1])
                elif proba.ndim == 1:
                    return float(proba[0])
        
        # Fallback sur predict si predict_proba n'existe pas
        if hasattr(model, 'predict'):
            pred = model.predict(input_data)
            if isinstance(pred, (np.ndarray, list)):
                return float(pred[0])
            return float(pred)
        
        return None
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.header("📊 Dashboard Analytique")
    
    data, data_source = load_data()
    
    if data is not None:
        # Vérifier si la colonne Churn existe
        has_churn = 'Churn' in data.columns or 'Exited' in data.columns
        churn_col = 'Churn' if 'Churn' in data.columns else 'Exited' if 'Exited' in data.columns else None
        
        # KPIs principaux
        st.subheader("📈 Indicateurs Clés de Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Clients", f"{len(data):,}")
        
        with col2:
            if has_churn:
                churn_count = (data[churn_col] == 1).sum()
                st.metric("📉 Clients en Churn", f"{churn_count:,}")
            else:
                st.metric("📁 Colonnes", len(data.columns))
        
        with col3:
            if has_churn:
                churn_rate = (data[churn_col].sum() / len(data) * 100)
                st.metric("📊 Taux de Churn", f"{churn_rate:.2f}%", 
                         delta=f"{churn_rate - 20:.2f}%" if churn_rate > 20 else None,
                         delta_color="inverse")
            else:
                st.metric("📋 Lignes", f"{len(data):,}")
        
        with col4:
            if has_churn:
                retention_rate = 100 - (data[churn_col].sum() / len(data) * 100)
                st.metric("✅ Taux de Rétention", f"{retention_rate:.2f}%")
            else:
                st.metric("🗂️ Variables", len(data.columns))
        
        st.markdown("---")
        
        # Aperçu des données avec filtres
        st.subheader("🔍 Aperçu des Données")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Source: `{data_source}`")
        with col2:
            n_rows = st.selectbox("Lignes à afficher:", [5, 10, 20, 50, 100], index=1)
        
        st.dataframe(data.head(n_rows), use_container_width=True, height=300)
        
        # Statistiques descriptives
        with st.expander("📊 Statistiques Descriptives", expanded=False):
            st.dataframe(data.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Visualisations du churn
        if has_churn:
            st.subheader("📉 Analyse du Churn")
            
            col1, col2 = st.columns(2)
            
            with col1:
                churn_counts = data[churn_col].value_counts()
                fig = px.pie(
                    values=churn_counts.values,
                    names=['Rétention', 'Churn'],
                    title="Distribution Churn vs Rétention",
                    color_discrete_sequence=['#2ecc71', '#e74c3c'],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Rétention', 'Churn'],
                    y=churn_counts.values,
                    marker_color=['#2ecc71', '#e74c3c'],
                    text=churn_counts.values,
                    textposition='auto',
                ))
                fig.update_layout(
                    title="Nombre de Clients par Catégorie",
                    xaxis_title="Catégorie",
                    yaxis_title="Nombre de clients",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse par variables catégorielles
            st.markdown("---")
            st.subheader("🔬 Analyse Segmentée du Churn")
            
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                selected_cat = st.selectbox("Variable d'analyse:", categorical_cols)
                
                churn_by_cat = data.groupby(selected_cat)[churn_col].agg(['sum', 'count'])
                churn_by_cat['rate'] = (churn_by_cat['sum'] / churn_by_cat['count'] * 100)
                
                fig = px.bar(
                    churn_by_cat.reset_index(),
                    x=selected_cat,
                    y='rate',
                    title=f"Taux de Churn par {selected_cat}",
                    labels={'rate': 'Taux de Churn (%)', selected_cat: selected_cat},
                    color='rate',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: PRÉDICTION
# ============================================================================
elif page == "🔮 Prédiction":
    st.header("🔮 Prédiction Individuelle du Churn")
    
    st.info("📝 Remplissez le formulaire ci-dessous pour obtenir une prédiction de risque de churn")
    
    model = load_model()
    
    if model is not None:
        with st.form("prediction_form"):
            st.subheader("Informations du Client")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Données Démographiques**")
                age = st.number_input("Âge", min_value=18, max_value=100, value=35, help="Âge du client en années")
                gender = st.selectbox("Genre", ["Male", "Female"], index=0)
                geography = st.selectbox("Pays", ["France", "Spain", "Germany"], index=0)
            
            with col2:
                st.markdown("**💰 Données Financières**")
                credit_score = st.number_input("Score de Crédit", min_value=300, max_value=850, value=650, 
                                              help="Score de crédit du client (300-850)")
                balance = st.number_input("Solde du Compte ($)", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)
                salary = st.number_input("Salaire Annuel ($)", min_value=0.0, max_value=500000.0, value=50000.0, step=1000.0)
            
            with col3:
                st.markdown("**📊 Relation Bancaire**")
                tenure = st.number_input("Ancienneté (années)", min_value=0, max_value=40, value=5, 
                                        help="Nombre d'années en tant que client")
                num_products = st.number_input("Nombre de Produits", min_value=1, max_value=4, value=2,
                                             help="Nombre de produits bancaires détenus")
                is_active = st.selectbox("Statut du Compte", ["Actif", "Inactif"], index=0)
                has_credit_card = st.selectbox("Carte de Crédit", ["Oui", "Non"], index=0)
            
            submitted = st.form_submit_button("🚀 Lancer la Prédiction", use_container_width=True, type="primary")
        
        if submitted:
            with st.spinner("🔄 Analyse en cours..."):
                # Préparation des données
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'NumOfProducts': [num_products],
                    'HasCrCard': [1 if has_credit_card == "Oui" else 0],
                    'IsActiveMember': [1 if is_active == "Actif" else 0],
                    'EstimatedSalary': [salary],
                    'Geography_Germany': [1 if geography == "Germany" else 0],
                    'Geography_Spain': [1 if geography == "Spain" else 0],
                    'Gender_Male': [1 if gender == "Male" else 0]
                })
                
                prediction_proba = make_prediction(model, input_data)
                
                if prediction_proba is not None:
                    st.success("✅ Prédiction effectuée avec succès!")
                    
                    # Résultat de la prédiction
                    st.markdown("---")
                    st.subheader("🎯 Résultat de la Prédiction")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Probabilité de Churn", 
                            f"{prediction_proba*100:.1f}%",
                            delta=f"{(prediction_proba - 0.5)*100:.1f}% vs seuil" if prediction_proba != 0.5 else None,
                            delta_color="inverse"
                        )
                    
                    with col2:
                        risk_level = "🔴 ÉLEVÉ" if prediction_proba > 0.7 else "🟡 MOYEN" if prediction_proba > 0.3 else "🟢 FAIBLE"
                        st.metric("Niveau de Risque", risk_level)
                    
                    with col3:
                        recommendation = "Action Immédiate" if prediction_proba > 0.7 else "Surveillance" if prediction_proba > 0.3 else "Aucune Action"
                        st.metric("Recommandation", recommendation)
                    
                    # Jauge visuelle
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction_proba * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risque de Churn (%)"},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "#d4edda"},
                                {'range': [30, 70], 'color': "#fff3cd"},
                                {'range': [70, 100], 'color': "#f8d7da"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Détails contextuels
                    st.markdown("---")
                    st.subheader("📋 Profil du Client")
                    
                    profile_cols = st.columns(4)
                    profile_cols[0].metric("👤 Âge", f"{age} ans")
                    profile_cols[1].metric("⏱️ Ancienneté", f"{tenure} ans")
                    profile_cols[2].metric("💵 Solde", f"${balance:,.0f}")
                    profile_cols[3].metric("💰 Salaire", f"${salary:,.0f}")
                    
                    # Recommandations d'action
                    if prediction_proba > 0.5:
                        st.markdown("""
                        <div class="prediction-high-risk">
                        <h4>⚠️ Actions Recommandées pour ce Client à Risque:</h4>
                        <ul>
                            <li>📞 Contact proactif par un conseiller dédié</li>
                            <li>🎁 Offre personnalisée de fidélisation</li>
                            <li>💬 Enquête de satisfaction pour identifier les problèmes</li>
                            <li>📊 Analyse détaillée de l'historique de transactions</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-low-risk">
                        <h4>✅ Client à Faible Risque:</h4>
                        <ul>
                            <li>👍 Maintenir la qualité de service actuelle</li>
                            <li>📧 Communication régulière et personnalisée</li>
                            <li>🎯 Opportunités de cross-selling</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Impossible d'effectuer la prédiction")
    else:
        st.warning("⚠️ Le modèle n'est pas disponible. Vérifiez le fichier `models/best_model.pkl`")

# ============================================================================
# PAGE 3: ANALYSE EDA
# ============================================================================
elif page == "📈 Analyse EDA":
    st.header("📈 Exploratory Data Analysis")
    
    data, _ = load_data()
    
    if data is not None:
        # Filtres de sélection
        col1, col2 = st.columns([2, 1])
        with col1:
            analysis_type = st.selectbox(
                "Type d'analyse:",
                ["Distribution Univariée", "Matrice de Corrélation", "Analyse Bivariée"]
            )
        
        st.markdown("---")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if analysis_type == "Distribution Univariée":
            if numeric_cols:
                selected_col = st.selectbox("Variable à analyser:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogramme
                    fig = px.histogram(
                        data,
                        x=selected_col,
                        title=f"Distribution de {selected_col}",
                        nbins=50,
                        color_discrete_sequence=['#3498db'],
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(
                        data,
                        y=selected_col,
                        title=f"Box Plot - {selected_col}",
                        color_discrete_sequence=['#e74c3c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                st.subheader(f"📊 Statistiques pour {selected_col}")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                stats_col1.metric("Moyenne", f"{data[selected_col].mean():.2f}")
                stats_col2.metric("Médiane", f"{data[selected_col].median():.2f}")
                stats_col3.metric("Écart-type", f"{data[selected_col].std():.2f}")
                stats_col4.metric("Valeurs manquantes", f"{data[selected_col].isna().sum()}")
        
        elif analysis_type == "Matrice de Corrélation":
            if len(numeric_cols) > 1:
                # Sélection des variables
                selected_vars = st.multiselect(
                    "Variables à inclure (laisser vide pour toutes):",
                    numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))]
                )
                
                if selected_vars and len(selected_vars) > 1:
                    corr_matrix = data[selected_vars].corr()
                    
                    # Heatmap
                    fig = px.imshow(
                        corr_matrix,
                        title="Matrice de Corrélation",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        text_auto='.2f',
                        aspect="auto"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top corrélations
                    st.subheader("🔝 Top 10 Corrélations")
                    corr_pairs = corr_matrix.unstack()
                    corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False)
                    top_corr = corr_pairs.head(10).reset_index()
                    top_corr.columns = ['Variable 1', 'Variable 2', 'Corrélation']
                    st.dataframe(top_corr, use_container_width=True)
        
        elif analysis_type == "Analyse Bivariée":
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variable X:", numeric_cols, index=0)
            with col2:
                y_var = st.selectbox("Variable Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            if x_var != y_var:
                # Scatter plot
                churn_col = 'Churn' if 'Churn' in data.columns else 'Exited' if 'Exited' in data.columns else None
                
                if churn_col:
                    fig = px.scatter(
                        data,
                        x=x_var,
                        y=y_var,
                        color=churn_col,
                        title=f"{y_var} vs {x_var} (coloré par Churn)",
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        opacity=0.6
                    )
                else:
                    fig = px.scatter(
                        data,
                        x=x_var,
                        y=y_var,
                        title=f"{y_var} vs {x_var}",
                        color_discrete_sequence=['#3498db']
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Corrélation
                corr_value = data[[x_var, y_var]].corr().iloc[0, 1]
                st.metric("Coefficient de Corrélation", f"{corr_value:.3f}")

# ============================================================================
# PAGE 4: ANALYSE BATCH
# ============================================================================
elif page == "📑 Analyse Batch":
    st.header("📑 Prédictions en Masse")
    
    st.info("📤 Uploadez un fichier CSV contenant les données de plusieurs clients pour obtenir des prédictions groupées")
    
    model = load_model()
    
    if model is not None:
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                
                st.subheader("📄 Aperçu des Données")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                st.metric("Nombre de clients", len(batch_data))
                
                if st.button("🚀 Lancer les Prédictions", use_container_width=True, type="primary"):
                    with st.spinner("Prédictions en cours..."):
                        predictions = []
                        
                        for idx, row in batch_data.iterrows():
                            try:
                                proba = make_prediction(model, pd.DataFrame([row]))
                                predictions.append(proba if proba is not None else np.nan)
                            except:
                                predictions.append(np.nan)
                        
                        batch_data['Churn_Probability'] = predictions
                        batch_data['Churn_Prediction'] = (batch_data['Churn_Probability'] > 0.5).astype(int)
                        batch_data['Risk_Level'] = pd.cut(
                            batch_data['Churn_Probability'],
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Faible', 'Moyen', 'Élevé']
                        )
                        
                        st.success("✅ Prédictions terminées!")
                        
                        # Résumé
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Clients à Risque Élevé", (batch_data['Risk_Level'] == 'Élevé').sum())
                        col2.metric("Clients à Risque Moyen", (batch_data['Risk_Level'] == 'Moyen').sum())
                        col3.metric("Clients à Risque Faible", (batch_data['Risk_Level'] == 'Faible').sum())
                        
                        # Visualisation
                        fig = px.histogram(
                            batch_data,
                            x='Churn_Probability',
                            nbins=30,
                            title="Distribution des Probabilités de Churn",
                            color_discrete_sequence=['#3498db']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export
                        st.subheader("📥 Télécharger les Résultats")
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Télécharger le CSV avec prédictions",
                            data=csv,
                            file_name="predictions_churn.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.dataframe(batch_data, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {str(e)}")

# ============================================================================
# PAGE 5: À PROPOS
# ============================================================================
elif page == "ℹ️ À propos":
    st.header("ℹ️ À propos du Projet")
    
    st.markdown("""
    ### 🎯 Objectif du Projet
    
    Cette application de **prédiction de churn bancaire** utilise des techniques de machine learning
    pour identifier les clients à risque de quitter leur banque. L'objectif est de permettre aux 
    équipes bancaires de prendre des actions préventives et personnalisées.
    
    ---
    
    ### 🔧 Technologies Utilisées
    
    - **Python 3.x** - Langage de programmation
    - **Streamlit** - Framework d'application web
    - **Scikit-learn** - Modèles de machine learning
    - **Plotly** - Visualisations interactives
    - **Pandas & NumPy** - Manipulation de données
    
    ---
    
    ### 📊 Structure du Projet
    """)
    
    st.code("""
    CHURN-PREDICTION-BANKING/
    ├── data/
    │   ├── brut_data.csv
    │   ├── cleaned_data.csv
    │   └── preprocessed_data.csv
    ├── models/
    │   └── best_model.pkl
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_data_preprocessing.ipynb
    │   ├── 03_data_modeling.ipynb
    │   ├── 04_model_optimisation.ipynb
    │   └── 05_model_explainability.ipynb
    ├── src/
    ├── app.py
    └── requirements.txt
    """, language="bash")

 
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'><small>🏦 Bank Churn Prediction | "
    "Bilal SAYOUD</small></div>",
    unsafe_allow_html=True
)
