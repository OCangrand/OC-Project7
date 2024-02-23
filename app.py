import os
import pickle
import pandas as pd
import shap
from io import StringIO
from flask import Flask, request, jsonify

### Cette API est hébergée sur le cloud à cette adresse : https://openclassrooms-projet7-oc.azurewebsites.net/

app = Flask(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "model", "model.pkl")

with open(model_path, 'rb') as pickle_file:
    model = pickle.load(pickle_file)


@app.route('/', methods=['GET'])
def accueil():
    return "Bienvenue !"

@app.route('/prediction', methods=['POST'])
def prediction():
    row_json = request.json
    row = pd.read_json(StringIO(row_json))
    row = row.drop(columns=['SK_ID_CURR'])
    
    resultat = model.predict_proba(row)
    proba_faillite = resultat[0][1]*100

    explainer = shap.TreeExplainer(model[1])
    shap_values = explainer.shap_values(row)
    
    return jsonify({
        'Proba_Faillite': proba_faillite,
        'Shap_Values': shap_values[1][0].tolist(),
        'Feature_Names': row.columns.tolist(),
        'Feature_Values': row.values[0].tolist()
    })

if __name__ == '__main__':
    app.run()