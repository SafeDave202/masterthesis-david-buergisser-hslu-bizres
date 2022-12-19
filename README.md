# Übersicht des Repositorys

Im Folgenden wird erläutert, wie das Repository aufgebaut ist sowie eine kleine Erklärung der jeweiligen Funktionen.

## Ordnerstruktur

Die Ordner beinhalten jeweils weiter Unterordner mit den jeweiligen Daten bzw. Funktionen. Im Ordner Analyse befinden sich die Scripts welche zur Erarbeitung der Unterkapitel aus Kapitel 5 verwendet wurden.
Der Ordner Data beinhaltet alle Daten, welche im Kontext dieser Arbeit verwendet wurden.Von den Nachhaltigkeitsberichten zu den Resultaten, bis hin zu den Zieltexten.
Die Front_End_Area beinhaltet den Code, welcher für den visuellen Prototypen der Analyse verwendet wurde. Die Underlines waren notwendig, damit das Deployment auf die Streamlit Cloud funktionierte.
Danach folgen die Ordner mit den jeweiligen Ansätzen. Im letzten Zielordner ist das Script, welches die Zieltexte vorbereitet.
Das Management Summary ist im so bennanten Markdown zu finden.
Die nlp_function.py Datei beinhaltet Funktionen, welche selbst geschrieben und für diese Arbeit verwendet wurden.
Das requirements.txt-File sollte alle nötigen Libraries mit den dazugehörigen Versionen beinhalten.

## Funktionsbeschreibung

### classifier(list)

Diese Funktion nimmt eine Liste entgegen.

def classifier(list):

    classifier_pipeline = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")

    input_sequence = list
    label_candidate = ['sustainability', 'human rights', 'fraud',
                       'social issues', 'labour law']
    output = classifier_pipeline(input_sequence, label_candidate)
    return output
