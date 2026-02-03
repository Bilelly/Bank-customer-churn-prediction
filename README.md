#  Prédiction de l’Attrition Client en Banque  
*Par Bilal Sayoud – Data Scientist*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

---

## 🚀 Démo 

*Prédisez le risque de churn en temps réel avec l'interface utilisateur*

👉 [**Tester l'application Streamlit**](https://bank-customer-churn-prediction-by-bilal-sayoud.streamlit.app/)  
---

##  Objectif du projet

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


## Dictionnaire des Significations

| Variable | Type | Signification |
|----------|------|---------------|
| **id** | int | Identifiant unique du client  |
| **CustomerId** | int | ID client |
| **Surname** | string | Nom de famille du client |
| **CreditScore** | int | Score de crédit du client (0-850) - indicateur de solvabilité |
| **Geography** | string | Pays de résidence du client (France, Allemagne, Espagne) |
| **Gender** | string | Genre du client (Male, Female) |
| **Age** | int | Âge du client en années |
| **Tenure** | int | Durée d'adhésion à la banque en années |
| **Balance** | float | Solde du compte bancaire en euros |
| **NumOfProducts** | int | Nombre de produits bancaires utilisés (1-4) |
| **HasCrCard** | binary | Possède une carte de crédit (0=Non, 1=Oui) |
| **IsActiveMember** | binary | Membre actif (0=Inactif, 1=Actif) - activité financière régulière |
| **EstimatedSalary** | float | Salaire annuel estimé en euros |
| **Exited** | binary | **TARGET** - Churn (0=Client reste, 1=Client part) |
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


<img width="1089" height="590" alt="image" src="https://github.com/user-attachments/assets/8e02fd13-c6ef-4440-8deb-5e8e78746d3b" />


✅ **Pourquoi ces métriques ?**  
En contexte de churn, il est **plus critique de ne pas manquer un client à risque** (haut recall) que d’avoir quelques faux positifs.

###  Top  des features les plus influentes (SHAP)
1. `Ratio_Products_Age`
2. `IsActiveMember`
3. `Age`
4. `Geography_Germany`
5. `Gender_Female`

<img width="787" height="940" alt="image" src="https://github.com/user-attachments/assets/6ece875a-bf32-491b-a63f-dacad2f52a13" />


→ Ces insights sont **actionnables** : par exemple, les clients allemands ou inactifs méritent une attention particulière.

---

## 🗂️ Structure du projet

```text
CHURN-PREDICTION-BANKING/
├── data/
│   ├── brut_data.csv        # Données brutes.
│   ├── cleaned_data.csv     # Données après nettoyage.
│   └── preprocessed_data.csv# Données prêtes pour le modèle.
├── models/
│   └── best_model.pkl       
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Exploration des données et analyse descriptive.
│   ├── 02_data_preprocessing.ipynb  # Transformations des features et sélection des modeles.
│   ├── 03_data_modeling.ipynb       # Entraînement des modèles de base.
│   ├── 04_model_optimisation.ipynb  # Tuning des hyperparamètres Optuna et GridsearchCV, faire une analyse sur l'équilibration.
│   └── 05_model_explainability.ipynb # comprendre les feautures avec shap .
├── report/              
│   └── eda_report.html      
├── src/                     
│   ├── EvaluationFunction.py# Fonction standardisée pour calculer les métriques.
│   ├── OptunaXGB.py         # Script pour l'optimisation des hyperparamètres d'XGBoost via Optuna.
│   ├── RemoveOutliers.py    # Fonction pour gérer les valeurs aberrantes.
│   └── ValidationCross.py   # Logique de validation croisée.
├── .gitignore
├── app.py
├── README.md
└── requirements.txt         
```
---

## ▶️ Comment reproduire le projet

```bash
# 1. Cloner le dépôt
git clone https://github.com/Bilelly/churn-prediction-banking.git

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Explorer les notebooks dans l'ordre
jupyter notebook










