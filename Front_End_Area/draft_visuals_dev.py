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
from st_aggrid import AgGrid
import io
import webbrowser


def main():
    with st.sidebar:
        choose = option_menu("App Gallery", ["About", "Polar Plots", "Analysis", "Contact"],
                             icons=['house', 'kanban',
                                    'book', 'person lines fill'],

                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#93D099"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#D093BB"},
            "nav-link-selected": {"background-color": "#D1BF94"},
        }
        )


# About Page
    if choose == "About":
        # col1 = st.columns([0.8])
        # with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Arial'; color: #D1BF94;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Ein Versuch der Anwendung von Natural Language Processing auf Nachhaltigkeitsberichte</p>',
                    unsafe_allow_html=True)
        st.write(
            "Prototyp der visuellen Darstellung der Masterarbeit von David Bürgisser")

        st.write("Dies ist die Masterarbeit von David Bürgisser. Diese Streamlit-Website soll einen ersten Versuch der Visualisierung der Ergebnisse liefern.")
        st.header("""Einleitung""")
        st.write("""
        Selten hatte eine Abstimmung so viel Aufmerksamkeit auf sich gezogen, wie jene vom 29. November 2020. Die Konzernverantwortungsinitiative «Für verantwortungsvolle Unter-nehmen – zum Schutz von Mensch und Umwelt» hat es geschafft die Menschen dazu zu bewegen Balkone mit orangenen Fahnen zu tapezieren während die Gegnerschaft viel Geld in die grosse Anti-Kampagne gesteckt hat. Allen war bewusst; es steht viel auf dem Spiel. Sollen Schweizer Unternehmen für die Arbeitsweisen ihrer Subsidiären in Drittweltländern verantwortlich sein? Genau dies wäre der Plan der Initiative gewesen. Doch das Unterfan-gen wurde durch das Ständemehr trotz Stimmmehrheit der Bevölkerung vereitelt. Der indi-rekte Gegenvorschlag veranlasst immerhin Schweizer Unternehmen ab einer gewissen Grösse dazu, per Geschäftsjahr 2023 jährlich über fünf nicht-finanzielle Belange zu berich-ten. Diese Themen umfassen die Bereiche Umwelt, Sozialbelange, Arbeitnehmerbelange, Menschenrechte und Korruption. 
        Dies bedeutet in Zukunft zwischen 200 und 1000 Berichte im Jahr, welche gelesen und bewertet werden müssten. Eine Vernachlässigung des Einreichens sowie eine ungenügende Berichterstattung kann mit bis zu 100'000 CHF bestraft. Doch wie wird die Qualität der Prüfung sichergestellt? Wer kümmert sich um die Bewertung dieser Berichte? Diese Fra-gen sind weitgehendst noch offen. Auch wenn sich einzelne Nichtregierungsorganisatio-nen gewissen Berichten annehmen würden, wäre eine objektive Bewertung immer noch ausstehend. An diesem Punkt setzt die vorliegende Arbeit an. Mit dem Auftraggeber Bizres wird versucht, Möglichkeiten der Data Science Domäne Natural Language Processing (NLP) zu erkunden, um eine objektive und einheitliche Bewertung von Nachhaltigkeitsbe-richten durchzuführen. 
        """)
        st.header("Erkenntnisse")
        st.write("""Während der Arbeit wurden verschiedene Ansätze getestet und ausprobiert. Zum einen wurden verschiedene Algorithmen in verschiedenen Techniken an den Texten getestet. Zum anderen wurden verschiedene Eingabetexte sowie Zieltexte verwendet. 
        Erste Erkenntnisse der Vergleichsanalysen konnten Korrelationen zwischen den Ansätzen Sentence-BERT-satzweise und Zero-Shot-Learning-satzweise aufzeigen. Dies sind zwei Verfahren, welche auf verschiedene Weisen jeden Satz einzeln analysieren und bewerten. Die Korrelation ist mit einem R2 von rund 57.7 % moderat mit einem P-Value unter 0.05. 
        Eine weitere Erkenntnis ist, dass die Algorithmen gewisse Belangen grundsätzlich höher bewerten als andere. Das Belangen Nachhaltigkeit wird in allen Ansätzen durchschnittlich höher bewertet als die anderen Belangen. Die Frage der Ursache kann in dieser Arbeit nicht abschliessend geklärt werden. 
        Des Weiteren ergab ein erster Vergleichstest beim Sentence-BERT-satzweise-Ansatz eine signifikante Differenz bei den Ergebnissen zwischen «Normalen Reports», «Fake Reports» und «DE-Reports». Dieser Ansatz konnte eine klare Trennung des Report-Typen aufzeigen. 
        Als letzter Vergleich zeigte der Austausch des Zieltextes zum Thema Menschenrechte beim Sentence-BERT-Ansatz eine signifikante Abweichung auf. Dadurch wird verdeut-licht, wie wichtig die Zieltexte als Bewertungskriterium sind. 
        """)
        st.header("Handlungsempfehlungen")
        st.write("""Die Ergebnisse zeigen auf, dass die gewählten Ansätze durchaus Potential haben. Als Handlungsempfehlungen bzw. Ausblick sind zwei Punkte besonders hervorzuheben.
                Einerseits sollte die Report-Typ-Zuteilung vertieft werden. Wenn ein gewählter Ansatz fähig ist, Reportarten bei einem grösseren Sampling voneinander zu unterscheiden, wäre dies eine starke Metrik, um die Fähigkeiten eines Algorithmus zu bewerten.
            Zum anderen wird eine Fokussierung auf den Sentence-BERT-Ansatz empfohlen. Da eben dieser Ansatz als einziger fähig war, die Report-Typen auseinanderzuhalten, wäre eine Wei-terentwicklung und Vertiefung dieses Ansatz zu empfehlen. Dies kann auf zwei Ebenen passieren. Erstere wäre die Optimierung des Algorithmus durch finetuning. Dies wird als sehr arbeitsintensiv erachtet. Zweitere wären weitere Vergleiche verschiedener Zieltexte, welche als Basis für die Bewertung dienen, interessant. 
        """)
        # st.image(profile, width=700)

# second page

    elif choose == "Polar Plots":
        col1, col2 = st.columns([0.8, 0.1])
        with col1:               # To display the header text using css style

            path = "https://github.com/SafeDave202/masterthesis-david-buergisser-hslu-bizres/blob/754c92be5ec0e18d128609e435f549cd10c42b99/Data/Resultate/combined_results.csv?raw=true"

            data = pd.read_csv(path, index_col=0)
            st.title("Report Analyse")

            menu = data["Company"].tolist()
            menu = set(menu)

            st.write(
                "Ganz unten kannst Du die Ergebnisse der verschiedenen Ansätze nebeneinander auf einen Blick durchklicken.")

            choice = st.sidebar.selectbox(
                "Wähle einen Nachhaltigkeitsbericht", menu)
            # st.write(type(choice))
            st.write(
                f"""Du hast den Bericht von {choice} gewählt. Falls Du dich über die lustigen Namen des Berichts wunderst: dieser stellt sich aus den 3 meistgenannten
                Worten des Berichts zusammen. So siehst Du gleich den Grad der Selbstverliebtheit der Firma. ;)""")

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

    elif choose == "Contact":
        st.title("Kontakt")

        st.write(
            "Falls Du Fragen oder Anliegen zu dieser Arbeit hast; zögere nicht, mich zu kontaktieren. :)")

        url_website = 'www.worldofdave.ch'
        url_github = 'https://github.com/SafeDave202'
        url_linkedin = 'https://www.linkedin.com/in/david-b%C3%BCrgisser-96068414b/'
        url_mail = "mailto:dabuergi@protonmail.com"

        if st.button('Fotografie Website'):
            webbrowser.open_new_tab(url_website)

        if st.button('LinkedIn'):
            webbrowser.open_new_tab(url_linkedin)

        if st.button('GitHub'):
            webbrowser.open_new_tab(url_github)

        if st.button('Mail'):
            webbrowser.open_new_tab(url_mail)


if __name__ == '__main__':
    main()
