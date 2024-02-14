import mlflow
import pandas as pd
import requests
import streamlit as st

st.title("Calcul de probabilité du remboursement d'un prêt")

df = pd.read_csv("dataframe_final_test.csv")

id_user = st.text_input("Entrez l'id du client")



if id_user:
    try :
        int(id_user)
    except:
        st.write("L'id client doit être numérique")
    else:
        if int(id_user) in df['SK_ID_CURR'].tolist():
            st.write("Id client actuel :", id_user)
            row = df[df['SK_ID_CURR'] == int(id_user)]
            row_json = row.to_json()
            req = requests.post("http://127.0.0.1:5000/prediction", json=row_json)
            st.write("Passage API OK")
            resultat = req.json()
            st.write("Probabilité de faillite du client :", resultat["Proba_Faillite"])
            if resultat["Proba_Faillite"]>0.3:
                st.write("Prêt refusé...")
            else:
                st.write("Prêt accepté !")
            st.write("Shap Values :", resultat["Shap_Values"])
            st.write("Feature Names :", resultat["Feature_Names"])
            st.write("Feature Values :", resultat["Feature_Values"])
        else:
            st.write("Id non trouvé dans la base de donnée. Essayez en un autre (Exemple : 100001, 100028, 456250)")
