Dépot de mon travail sur le projet 7 de la formation Data Scientist d'OpenClassrooms :

#Implémentez un modèle de scoring.

## Présentation des livrables :

- Dossier **.github/worflows** : Contient le fichier de configuration du déploiement automatique via Github actions.
- Dossier **model** : Contient le modèle Lightgbm pour la classification.
- **AnalyseExploratoire.ipynb** : Notebook de l'analyse exploratoire récupéré de Kaggle (https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook) et légèrement modifié.
- **Dashboard.py** : Dashboard Streamlit faisant appel à l'API.
- **Datadrift.ipynb** : Notebook du datadrift.
- **FeatureEngineering.ipynb** : Notebook du feature engineering récupéré de Kaggle (https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script) et légèrement modifié.
- **Modelisation.ipynb** : Notebook principal, allant du test des différents modèles à l'analyse de la feature importance.
- **app.py** : API hébergée à cette adresse : https://openclassrooms-projet7-oc.azurewebsites.net/.
- **data_drift_report.html** : Rapport sur le Datadrift généré via le notebook Datadrift.
- **dataframe_final_test_10first.csv** : Contient les 10 premières lignes du dataframe final généré après le feature engineering et toutes autres modifications. Il permet simplement au tests Pytest de fontionner. Le fichier complet est trop volumineux pour être stocké ici.
- **requirements.txt** : Liste des packages nécessaires au bon fonctionnement du projet.
- **test_API.py** : Fichier de tests unitaires fait avec Pytest.
