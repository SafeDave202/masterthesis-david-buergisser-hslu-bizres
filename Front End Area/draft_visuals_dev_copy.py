# from curses import raw
import pandas as pd
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    path = "https://github.com/SafeDave202/masterthesis-david-buergisser-hslu-bizres/blob/754c92be5ec0e18d128609e435f549cd10c42b99/Data/Resultate/combined_results.csv?raw=true"

    data = pd.read_csv(path, index_col=0)
    st.title("Report Analysis")

    menu = data["Company"].tolist()
    menu = set(menu)

    choice = st.sidebar.selectbox("Menu", menu)
    # st.write(type(choice))
    st.write(
        f"""Du hast den Bericht von {choice} gewählt. Falls Du dich über die lustigen Namen des Berichts wunders: dieser stellt sich aus den 3 meistgenannten
        Worten des Berichts zusammen. So siehst du gleich den Grad der Selbstverliebtheit der Firma. ;)""")

    # # if choice == "Home":
    # #     st.subheader("Home")
    # #     input_file = st.file_uploader("Upload Report", type=[
    # #         "pdf", "text", "docx"])
    # #     if st.button("Process"):
    # #         if input_file is not None:
    # #             file_detail = {"filename": input_file.name,
    # #                            "filetype": input_file.type, "filesize": input_file.size}
    # #             st.write(file_detail)
    # #             if file_detail["filetype"] == "application/pdf":
    # #                 # try:
    # #                 #     text = read_pdf(input_file)
    # #                 #     st.write(text)

    # #                 # except:
    # #                 #     st.write("Didn't work")
    # #                 raw_text = read_pdf(input_file)(raw_text)
    # #                 # processed_text = remove_strange_characters
    # #                 st.write()
    # #                 st.write(raw_text)

    choice_data = data[data["Company"] == choice]
    st.title("Sentence-BERT satzweise Analyse")
    st.write(
        """Hier werden die Scores Deiner Auswahl zu den fünf Label gezeigt. Jeder Satz wurde einzeln mit den Zieltexten von Wikipedia verglichen und bewertet.
        Die Scores repräsentieren die Mittelwerte de Bewertungen. Negative Bewertungen wurden automatisch auf 0 gesetzt.""")
    st.dataframe(
        choice_data["Sentence Cosine Similarity Wiki Summarized Mean Threshold 0"])
    st.write("Dieser Polar-Plot zeigt die Verteilung visuell.")
    fig = px.line_polar(choice_data, r='Sentence Cosine Similarity Wiki Summarized Mean Threshold 0', theta='Label',
                        range_r=[0, 0.3], line_close=True, template="seaborn")

    st.plotly_chart(fig, use_container_width=True)

    st.title("Zero-Shot Learning satzweise")
    st.write(
        """Dieser Abschnitt zeigt die Daten des Zero-Shot Learning satzweise Ansatzes mit dem gewählten Bericht. Im Gegensatz zum ersten Ansatz braucht es
        hierzu keinen Zieltext. Der Klassifizierungsalgorithmus beruht auf dem Model 'bart-large-mnli' und versucht jeden einzelnen Satz den fünf verschiedenen
        Labeln prozentual zuzuweisen. Nachdem jeder Satz gescored wurde, wird der Durchschnitt berechnet. Auch hier gibt es wieder einen Threshold. Sobald ein Score
        kleiner als 0.05 ist, wird er nicht miteinbezogen, da sonst die Resultate zu verzerrt sind.""")

    st.dataframe(choice_data["Zero Shot Learning Sentence Mean Threshold"])

    fig = px.line_polar(choice_data, r='Zero Shot Learning Sentence Mean Threshold', theta='Label',
                        range_r=[0, 0.6], line_close=True, template="seaborn")

    st.plotly_chart(fig, use_container_width=True)

    st.title("Zero-Shot Learning ganzer Text")
    st.write(
        """Nachdem die vorhergehenden Ansätze sich jeweils um einzelne Sätze kümmerten, wurde als Vergleich noch ein Ansatz gewählt, der den ganzen Text betrachtet.
        Vom Modell her handels es sich um den selben Algorithmus wie im vorherigen Plot. Nur wurde diesmal nicht satzweise die Scores verteilt und anschliessend
        Durchschnittswerte berechnet, sondern der ganze Text ergab einen Score für jedes Label.""")

    st.dataframe(choice_data["Zero Shot Fulltext"])

    fig = px.line_polar(choice_data, r='Zero Shot Fulltext', theta='Label',
                        range_r=[0, 0.6], line_close=True, template="seaborn")

    st.plotly_chart(fig, use_container_width=True)

    # st.title("Comparison of the different approaches")
    # st.write("""Since we have two times used the same similarity analysis; once with the full text and once with the topic rebuilt textes; we can see, if there is a difference
    # between these two approaches in the similarity to the wikipedia texts""")

    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}]*2]*2)

    fig.add_trace(go.Scatterpolar(
        r=[1, 2, 3],
        theta=[50, 100, 200],
        marker_symbol="square"
    ), 1, 1)
    fig.add_trace(go.Scatterpolar(
        r=[1, 2, 3],
        theta=[1, 2, 3],
        thetaunit="radians"
    ), 1, 1)
    fig.add_trace(go.Scatterpolar(
        r=["a", "b", "c", "b"],
        theta=["D", "C", "B", "A"],
        subplot="polar2"
    ), 1, 2)
    fig.add_trace(go.Scatterpolar(
        r=[50, 300, 900],
        theta=[0, 90, 180],
        subplot="polar3"
    ), 2, 1)
    fig.add_trace(go.Scatterpolar(
        mode="lines",
        r=[3, 3, 4, 3],
        theta=[0, 45, 90, 270],
        fill="toself",
        subplot="polar4"
    ), 2, 2)

    fig.update_layout(
        polar=dict(
            radialaxis_range=[1, 4],
            angularaxis_thetaunit="radians"
        ),
        polar3=dict(
            radialaxis=dict(type="log", tickangle=45),
            sector=[0, 180]
        ),
        polar4=dict(
            radialaxis=dict(visible=False, range=[0, 6])),
        showlegend=False
    )


if __name__ == '__main__':
    main()
