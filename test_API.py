import pandas as pd
import numpy as np
import pytest
import requests
from app import app, model
from Dashboard import df

def test_csv():
    nb_col = df.shape[1]
    assert nb_col == 796, "Le nombre de colonnes devrait être égal à 796"


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
        assert 0 <= pred_proba <= 100, "La probilité de faillite devrait être entre 0 et 100%."
        assert len(resultat['Shap_Values']) == 795

