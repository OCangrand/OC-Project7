import pandas as pd
import pytest
from app import app, model

#Dataframe de test avec seulement les 10 premières lignes pour pouvoir tester l'API
df = pd.read_csv('dataframe_final_test_10first.csv')

def test_model():
    assert model is not None, "Le modèle n'a pas pu être récupéré"
    row1 = df[df['SK_ID_CURR']==100001]
    row1 = row1.drop(columns=['SK_ID_CURR'])
    res = model.predict(row1)
    assert res==1 or res==0
    res_proba = model.predict_proba(row1)
    assert 0 <= res_proba[0][1] <= 1


# Créer un client de test pour l'application Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Teste la fonction de prédiction de l'API
def test_prediction(client):
    row = df[df['SK_ID_CURR'] == 100001]
    row_json = row.to_json()
    with app.test_client() as client:
        response = client.post('/prediction', json=row_json)
        resultat = response.json
        pred_proba = round(resultat["Proba_Faillite"], 2)
        assert 0 <= pred_proba <= 100, "La probabilité de faillite devrait être entre 0 et 100%."
        assert len(resultat['Shap_Values']) == 795, "Le nombre de features ou le calcul des features importances avec Shap n'est pas bon."

