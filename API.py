import mlflow
import pandas as pd
import shap
from io import StringIO
from flask import Flask, request, jsonify

# To start the mlflow server in local : mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name = "BestModel"
model_version = 1
#Récupération du modèle
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")


app = Flask(__name__)

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
    app.run(debug=False, host="0.0.0.0", port=5000)