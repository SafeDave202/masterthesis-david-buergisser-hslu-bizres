import io
from pdfminer3.converter import TextConverter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
import string
from transformers import pipeline
import re
import collections


def classifier(list):

    classifier_pipeline = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")

    input_sequence = list
    label_candidate = ['sustainability', 'human rights', 'fraud',
                       'social issues', 'labour law']
    output = classifier_pipeline(input_sequence, label_candidate)
    return output


def text_loader(path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(
        resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

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
