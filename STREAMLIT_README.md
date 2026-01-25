# 🚀 Lancer l'Application Streamlit

## Installation des dépendances

```bash
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse:
```
http://localhost:8501
```

## Fonctionnalités

### 📊 Dashboard
- Affichage des statistiques globales
- Aperçu des données
- Distribution du churn
- Statistiques descriptives

### 🔮 Prédiction
- Formulaire interactif pour entrer les données d'un client
- Prédiction du risque de churn
- Indicateur visuel du niveau de risque

### 📈 Analyse EDA
- Distribution des variables numériques
- Matrice de corrélation
- Visualisations interactives

### ℹ️ À propos
- Description du projet
- Technologies utilisées
- Structure du projet

## Notes

- Les données sont cachées en mémoire pour optimiser les performances
- Les prédictions sont actuellement simulées (remplacer par votre modèle réel)
- Pour intégrer votre modèle, modifiez la section prédiction dans `app.py`

