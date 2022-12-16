

import itertools
import torchvision
import torch
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import gensim.downloader as api
from spacy.lang.en import English
import pandas as pd
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
from transformers import pipeline
import spacy
import unicodedata
import re

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


def classifier(list):

    classifier_pipeline = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")

    input_sequence = list
    label_candidate = ['sustainability', 'human rights', 'fraud',
                       'social issues', 'labour law']
    output = classifier_pipeline(input_sequence, label_candidate)
    return output


def text_loader(path):

    from pdfminer3.layout import LAParams, LTTextBox
    from pdfminer3.pdfpage import PDFPage
    from pdfminer3.pdfinterp import PDFResourceManager
    from pdfminer3.pdfinterp import PDFPageInterpreter
    from pdfminer3.converter import PDFPageAggregator
    from pdfminer3.converter import TextConverter
    import io

    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(
        resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    # with open(r'Data\Sustainability Reports\soy-progress-mid-year-report-2021-en.pdf', 'rb') as fh:
    with open(fr"{path}", 'rb') as fh:

        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()
        return text
    # close open handles
    converter.close()
    fake_file_handle.close()


def remove_strange_characters(text):
    bad_chars = ["$", "’", "&", "|",
                 "(", ")", "[", "]", "/", "–", "- ", "™", "©"]
    text = ''.join(i for i in text if not i in bad_chars)
    return text


def remove_n(text):
    text = re.sub(r'\n+', " ", text)
    return text.strip()


def remove_colons(text):
    text = re.sub(r'\s,', ",", text)
    return text.strip()


def remove_stripes(text):
    text = re.sub(r'-', "", text)
    return text.strip()


def remove_redundant_whitespaces(text):
    text = re.sub(r'\s+', " ", text)
    return text.strip()


def remove_digits(text):
    mapping = str.maketrans('', '', string.digits)
    text = text.translate(mapping)
    return text
