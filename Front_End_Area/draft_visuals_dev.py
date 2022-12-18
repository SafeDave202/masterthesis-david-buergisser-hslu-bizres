# from curses import raw
import pandas as pd
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
import cv2
from st_aggrid import AgGrid
import io


def main():
    with st.sidebar:
        choose = option_menu("App Gallery", ["About", "Polar Plots", "Analysis", "Contact"],
                             icons=['house', 'kanban',
                                    'book', 'person lines fill'],

                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
        )


# About Page
    logo = Image.open(
        r'G:\Meine Ablage\GitHub\masterthesis-david-buergisser-hslu-bizres\Front_End_Area\Images\noun-design-thinking-2406269.png')
    profile = Image.open(
        r'G:\Meine Ablage\GitHub\masterthesis-david-buergisser-hslu-bizres\Front_End_Area\Images\noun-touring-ski-692819.png')
    if choose == "About":
        col1, col2 = st.columns([0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">About the Creator</p>',
                        unsafe_allow_html=True)
        with col2:               # To display brand log
            st.image(logo, width=130)

        st.write("Dies ist die Masterarbeit von David Bürgisser. Diese Streamlit-Website soll einen ersten Versuch der Visualisierung der Ergebnisse liefern.")
        # st.image(profile, width=700)

# second page

    elif choose == "Polar Plots":
        col1, col2 = st.columns([0.8, 0.1])
        with col1:               # To display the header text using css style

            path = "https://github.com/SafeDave202/masterthesis-david-buergisser-hslu-bizres/blob/754c92be5ec0e18d128609e435f549cd10c42b99/Data/Resultate/combined_results.csv?raw=true"

            data = pd.read_csv(path, index_col=0)
            st.title("Report Analysis")

            menu = data["Company"].tolist()
            menu = set(menu)

            choice = st.sidebar.selectbox(
                "Wähle einen Nachhaltigkeitsbericht", menu)
            # st.write(type(choice))
            st.write(
                f"""Du hast den Bericht von {choice} gewählt. Falls Du dich über die lustigen Namen des Berichts wunders: dieser stellt sich aus den 3 meistgenannten
                Worten des Berichts zusammen. So siehst du gleich den Grad der Selbstverliebtheit der Firma. ;)""")

            choice_data = data[data["Company"] ==
                               choice].sort_values(by=["Label"])
            st.title("Sentence-BERT satzweise Analyse")
            st.write(
                """Hier werden die Scores Deiner Auswahl zu den fünf Label gezeigt. Jeder Satz wurde einzeln mit den Zieltexten von Wikipedia verglichen und bewertet.
                Die Scores repräsentieren die Mittelwerte de Bewertungen. Negative Bewertungen wurden automatisch auf 0 gesetzt.""")
            st.dataframe(
                choice_data[["Label", "Sentence Cosine Similarity Wiki Summarized Mean Threshold 0"]].sort_values(by=["Label"]))
            st.write("Dieser Polar-Plot zeigt die Verteilung visuell.")
            fig1 = px.line_polar(choice_data, r='Sentence Cosine Similarity Wiki Summarized Mean Threshold 0', theta='Label',
                                 range_r=[0, 0.3], line_close=True, template="seaborn")

            st.plotly_chart(fig1, use_container_width=True)

            st.title("Zero-Shot Learning satzweise")
            st.write(
                """Dieser Abschnitt zeigt die Daten des Zero-Shot Learning satzweise Ansatzes mit dem gewählten Bericht. Im Gegensatz zum ersten Ansatz braucht es
                hierzu keinen Zieltext. Der Klassifizierungsalgorithmus beruht auf dem Model 'bart-large-mnli' und versucht jeden einzelnen Satz den fünf verschiedenen
                Labeln prozentual zuzuweisen. Nachdem jeder Satz gescored wurde, wird der Durchschnitt berechnet. Auch hier gibt es wieder einen Threshold. Sobald ein Score
                kleiner als 0.05 ist, wird er nicht miteinbezogen, da sonst die Resultate zu verzerrt sind.""")

            st.dataframe(
                choice_data[["Label", "Zero Shot Learning Sentence Mean Threshold"]].sort_values(by=["Label"]))

            fig2 = px.line_polar(choice_data, r='Zero Shot Learning Sentence Mean Threshold', theta='Label',
                                 range_r=[0, 0.7], line_close=True, template="seaborn")

            st.plotly_chart(fig2, use_container_width=True)

            st.title("Zero-Shot Learning ganzer Text")
            st.write(
                """Nachdem die vorhergehenden Ansätze sich jeweils um einzelne Sätze kümmerten, wurde als Vergleich noch ein Ansatz gewählt, der den ganzen Text betrachtet.
                Vom Modell her handels es sich um den selben Algorithmus wie im vorherigen Plot. Nur wurde diesmal nicht satzweise die Scores verteilt und anschliessend
                Durchschnittswerte berechnet, sondern der ganze Text ergab einen Score für jedes Label.""")

            st.dataframe(
                choice_data[["Label", "Zero Shot Fulltext"]].sort_values(by=["Label"]))

            fig3 = px.line_polar(choice_data, r='Zero Shot Fulltext', theta='Label',
                                 range_r=[0, 0.6], line_close=True, template="seaborn")

            st.plotly_chart(fig3, use_container_width=True)

            tab1, tab2, tab3 = st.tabs(
                ["Sentence-BERT satzweise Analyse", "Zero-Shot Learning satzweise", "Zero-Shot Learning ganzer Text"])
            with tab1:
                st.plotly_chart(fig1, theme="streamlit",
                                use_conatiner_width=True)
            with tab2:
                st.plotly_chart(fig2, theme="streamlit",
                                use_conatiner_width=True)
            with tab3:
                st.plotly_chart(fig3, theme="streamlit",
                                use_conatiner_width=True)

    elif choose == "Analysis":
        col1, col2 = st.columns([0.8, 0.1])
        with col1:               # To display the header text using css style
            path = "https://github.com/SafeDave202/masterthesis-david-buergisser-hslu-bizres/blob/754c92be5ec0e18d128609e435f549cd10c42b99/Data/Resultate/combined_results.csv?raw=true"

            data = pd.read_csv(path, index_col=0)
            st.title("Analysis")

            x_axis = ["Zero Shot Fulltext", "Sentence Cosine Similarity Wiki Summarized Mean Threshold 0",
                      "Sentence by Sentence Cosine Similarity human rights Mean Threshold 0", "Zero Shot Learning Sentence Mean Threshold"]
            y_axis = ["Zero Shot Fulltext", "Sentence Cosine Similarity Wiki Summarized Mean Threshold 0",
                      "Sentence by Sentence Cosine Similarity human rights Mean Threshold 0", "Zero Shot Learning Sentence Mean Threshold"]
            menu_x = set(x_axis)
            menu_y = set(y_axis)

            choice_x = st.sidebar.selectbox(
                "Wähle einen Nachhaltigkeitsbericht x-Achse", menu_x)
            choice_y = st.sidebar.selectbox(
                "Wähle einen Nachhaltigkeitsbericht y-Achse", menu_y)

            # st.write(type(choice))

            # choice_data = data[data["Report Type"] == "Normal"]
            choice_data = data

            result_prep = choice_data[[f"{choice_x}", f"{choice_y}"]]
            # result_prep = choice_data[[
            #     f"Zero Shot Fulltext", f"Sentence Cosine Similarity Wiki Summarized Mean Threshold 0"]]

            result_prep_melt = result_prep.melt(var_name="Data")
            fig = px.box(result_prep_melt, y="value",
                         points="all", color="Data", width=900)
            fig.update_layout(legend=dict(
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0.01
            ))

            st.plotly_chart(fig, use_container_width=True)

            # fig = px.scatter(choice_data, x=f"Zero Shot Fulltext",
            #                  y=f"Sentence Cosine Similarity Wiki Summarized Mean Threshold 0")
            fig = px.scatter(choice_data, x=f"{choice_x}", y=f"{choice_y}")

            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
