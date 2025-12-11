# 🏦 Prédiction de l’Attrition Client en Banque  
*Par Bilal Sayoud – Data Scientist*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

---

## 🎯 Objectif du projet

L’attrition client est un enjeu stratégique pour les banques. Identifier en amont les clients susceptibles de quitter permet de :
- **Anticiper les pertes de revenus**
- **Améliorer la fidélisation**
- **Augmenter la satisfaction client**

Ce projet propose un pipeline complet de **détection proactive du churn** basé sur des données comportementales, démographiques et financières.

---

## 📊 Données

- **Dataset** : 15 000 clients bancaires simulés
- **Target** : `Exited` (1 = client parti, 0 = client actif)
- **Déséquilibre** : ~20 % de churn → problème de classification binaire déséquilibrée
- **Fichiers** :
  - `brut_data.csv` : données brutes
  - `cleaned_data.csv` : après suppression des colonnes inutiles
  - `preprocessed_data.csv` : après feature engineering

---

## 🔧 Pipeline de Data Science

Le projet suit une **démarche structurée en 5 notebooks** :

| Notebook | Objectif |
|--------|--------|
| `01_data_exploration.ipynb` | Analyse exploratoire (EDA), visualisations, tests statistiques (Chi², ANOVA, KS) |
| `02_data_preprocessing.ipynb` | Nettoyage, gestion des outliers, création de 11 features métier |
| `03_data_modeling.ipynb` | Entraînement de Logistic Regression, Random Forest et XGBoost |
| `04_model_optimisation.ipynb` | Hyperparameter tuning avec Optuna |
| `05_model_evaluation_export.ipynb` | Évaluation finale, SHAP, export du modèle (`best_model.pkl`) |

---

## 📈 Performances du modèle final (XGBoost)

| Métrique | Valeur |
|--------|--------|
| **F1-Score** | 0.729 |
| **Recall** | 79.4 % |
| **AUC** | 0.931 |

✅ **Pourquoi ces métriques ?**  
En contexte de churn, il est **plus critique de ne pas manquer un client à risque** (haut recall) que d’avoir quelques faux positifs.

### 🔍 Top 5 des features les plus influentes (SHAP)
1. `Ratio_Products_Age`
2. `IsActiveMember`
3. `Age`
4. `Geography_Germany`
5. `Gender_Female`

→ Ces insights sont **actionnables** : par exemple, les clients allemands ou inactifs méritent une attention particulière.

---

## 🗂️ Structure du projet

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
│   └── 05_model_evaluation_export.ipynb
├── report/
│   ├── figure/
│   └── eda_report.html
├── src/
│   ├── __pycache__/
│   ├── EvaluationFunction.py
│   ├── helperfunction.py
│   ├── OptunaXGB.py
│   ├── RemoveOutliers.py
│   └── ValidationCross.py
├── .gitignore
├── README.md
└── requirements.txt

---

## ▶️ Comment reproduire le projet

```bash
# 1. Cloner le dépôt
git clone https://github.com/Bilelly/churn-prediction-banking.git

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Explorer les notebooks dans l'ordre
jupyter notebook



