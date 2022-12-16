# from curses import raw
from PyPDF2 import PdfFileReader
import pdfplumber
import itertools
from nlp_functions import text_loader, classifier, remove_strange_characters, remove_n, remove_colons, remove_stripes, remove_redundant_whitespaces, remove_digits
import torchvision
import torch
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import gensim.downloader as api
from spacy.lang.en import English
from gensim.models import Word2Vec
import io
from pdfminer3.converter import TextConverter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
import contractions
import string
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import spacy
import unicodedata
import re
import os

import collections

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


def main():
    data = pd.read_clipboard(r"Data\Resultate\combined_results.csv")
    st.title("Report Analysis")

    menu = data["Company"]

    # menu = os.listdir(
    #     r"..\Data\Nachhaltigkeitsberichte\Alle")
    choice = st.sidebar.selectbox("Menu", menu)
    # st.write(type(choice))
    st.write(
        f"You have choosen the {choice} report. In the following you will see the different analysis of this company's report.")

    # if choice == "Home":
    #     st.subheader("Home")
    #     input_file = st.file_uploader("Upload Report", type=[
    #         "pdf", "text", "docx"])
    #     if st.button("Process"):
    #         if input_file is not None:
    #             file_detail = {"filename": input_file.name,
    #                            "filetype": input_file.type, "filesize": input_file.size}
    #             st.write(file_detail)
    #             if file_detail["filetype"] == "application/pdf":
    #                 # try:
    #                 #     text = read_pdf(input_file)
    #                 #     st.write(text)

    #                 # except:
    #                 #     st.write("Didn't work")
    #                 raw_text = read_pdf(input_file)(raw_text)
    #                 # processed_text = remove_strange_characters
    #                 st.write()
    #                 st.write(raw_text)

    path = fr"..\Data\Resultate\Sentence by Sentence Cosine Similarity raw\{choice[:-4]}abb_sustainability_performance_cosine_raw.csv"
    # st.write(path)
    df_fulltext = pd.read_csv(path
                              )
    st.title("Full Text Cosine Similarity Analysis")
    st.write(
        "Here you can see the Scores of the 5 different areas the company should write about.")
    df_fulltext
    st.write("This is the radar Map of the scores. It shows the Similarity to the target texts from Wikipedia.")
    fig = px.line_polar(df_fulltext, r='Score', theta='Label',
                        range_r=[0, 1.0], line_close=True, template="seaborn")

    st.plotly_chart(fig, use_container_width=True)

    st.title("Full Text Topic Modeling")
    st.write(
        "This shows how BERT distributes the 5 different topics to the text")

    path = fr"G:\Meine Ablage\HSLU\Master\Master Thesis\Coding Area\Data\Results\{choice[:-4]}_FullText_TopicModeling.csv"
    df_fulltext_tm = pd.read_csv(path)

    df_fulltext_tm

    fig = px.line_polar(df_fulltext_tm, r='Score', theta='Label',
                        range_r=[0, 0.6], line_close=True, template="seaborn")

    st.plotly_chart(fig, use_container_width=True)

    st.title("Topic Modeled Cosine Similarity")
    st.write(
        """This shows the similarity of 5 different rebuilt texts. Each sentences was scored to a label and then rebuilt together with its closest sentences. This than was compared to the 
        wikipedia textes.""")

    path = fr"G:\Meine Ablage\HSLU\Master\Master Thesis\Coding Area\Data\Results\{choice[:-4]}_Paragraphed_Cosine_Scoring.csv"
    df_paragraphed_scoring = pd.read_csv(path)

    df_paragraphed_scoring

    fig = px.line_polar(df_paragraphed_scoring, r='Score', theta='Label',
                        range_r=[0, 1.0], line_close=True, template="seaborn")

    st.plotly_chart(fig, use_container_width=True)

    st.title("Comparison of the different approaches")
    st.write("""Since we have two times used the same similarity analysis; once with the full text and once with the topic rebuilt textes; we can see, if there is a difference
    between these two approaches in the similarity to the wikipedia texts""")


if __name__ == '__main__':
    main()
