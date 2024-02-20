import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

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
            #req = requests.post("http://127.0.0.1:5000/prediction", json=row_json)
            req = requests.post("https://openclassrooms-projet7-oc.azurewebsites.net/prediction", json=row_json)            
            st.write("Passage API OK")
            resultat = req.json()
            st.write("Probabilité de faillite du client :", round(resultat["Proba_Faillite"], 2),"%")
            #Résultat du prêt en fonction du threshold précédemment optimisé :
            if resultat["Proba_Faillite"]>24: 
                st.write("Prêt refusé...")
            else:
                st.write("Prêt accepté !")

            dfFeatName = pd.DataFrame(resultat['Feature_Names'])
            dfFeatValue = pd.DataFrame(resultat['Feature_Values'])
            dfShapValue = pd.DataFrame(resultat['Shap_Values'])
            feat_df = pd.concat([dfFeatName, dfFeatValue, dfShapValue], axis=1)
            feat_df.columns = ['Feature_Names', 'Feature_Values', 'Shap_Values']
            feat_df['Abs_Shap'] = feat_df['Shap_Values'].abs()
            feat_df.sort_values(by='Abs_Shap', ascending=False, inplace=True, ignore_index=True)
            feat_df.drop('Abs_Shap', axis=1, inplace=True)

            st.write(feat_df)

            
            c1, c2 = st.columns(2)

            c1.title("Analyse univariée")
            feat_selected_hist = c1.selectbox(label="Selectionner une feature :", options=feat_df['Feature_Names'], index=None)
            if feat_selected_hist:
                fig_hist = px.histogram(df[feat_selected_hist], x=feat_selected_hist)
                feat_selected_hist_serie = feat_df.loc[feat_df['Feature_Names'] == feat_selected_hist, 'Feature_Values']
                feat_selected_hist = feat_selected_hist_serie.iloc[0]

                fig_hist.add_annotation(x=feat_selected_hist, y=0, text="Client", showarrow=True, arrowhead=1, arrowcolor="Red", bgcolor="Red")
                c1.plotly_chart(fig_hist)


            c2.title("Analyse bivariée")
            feat_select_scatter_1 = c2.selectbox(label="Selectionner une 1ère feature :", options=feat_df['Feature_Names'], index=None)
            feat_select_scatter_2 = c2.selectbox(label="Selectionner une 2ème feature :", options=feat_df['Feature_Names'], index=None)
            if feat_select_scatter_1 and feat_select_scatter_2:
                fig_scatter = px.scatter(df, x=feat_select_scatter_1, y=feat_select_scatter_2)
                feat_selected_scatter_serie_1 = feat_df.loc[feat_df['Feature_Names'] == feat_select_scatter_1, 'Feature_Values']
                feat_selected_scatter_1 = feat_selected_scatter_serie_1.iloc[0]
                feat_selected_scatter_serie_2 = feat_df.loc[feat_df['Feature_Names'] == feat_select_scatter_2, 'Feature_Values']
                feat_selected_scatter_2 = feat_selected_scatter_serie_2.iloc[0]
                fig_scatter.add_annotation(x=feat_selected_scatter_1, y=feat_selected_scatter_2, text="Client", showarrow=True, arrowhead=1, arrowcolor="Red", bgcolor="Red")
                c2.plotly_chart(fig_scatter)
        else:
            st.write("Id non trouvé dans la base de donnée. Essayez-en un autre (Exemple : 100001, 100005, 100028, 456250).")
