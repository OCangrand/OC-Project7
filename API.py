import mlflow
import pandas as pd
import shap
import flask
from flask import request, jsonify

# To start the mlflow server in local : mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name = "LGBM"
model_version = 1
#Récupération du modèle
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

data = pd.read_csv("dataframe_final_test.csv")
test_id = 100001

app = flask.Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    #data = request.json
    user_id = data['SK_ID_CURR']
    row = data[data['SK_ID_CURR'] == test_id]
    row = row.drop(columns=['SK_ID_CURR'])
    
    resultat = model.predict_proba(row)
    proba_faillite = resultat[0][1]*100

    explainer = shap.TreeExplainer(model[1])
    shap_values = explainer.shap_values(row)
    
    return jsonify({
        'Proba faillite': proba_faillite,
        'Shap Values': shap_values[1][0].tolist(),
        'Feature Names': row.columns.tolist(),
        'Feature Values': row.values[0].tolist()
    })


app.run(debug=False, host="0.0.0.0", port=5000)