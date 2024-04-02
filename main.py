from tqdm.auto import tqdm
import random
import pandas as pd
import pytest
from spacy.lang.en import English

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    # print(cleaned_text)
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    import fitz
    doc = fitz.open(pdf_path)
    # print('Doc opened successfully!')
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        # print(text)
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 1,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text
                                })
    # print(pages_and_texts)
    return pages_and_texts


# def test_spacy():
#     nlp = English()
#     nlp.add_pipe("sentencizer")
#     doc = nlp("This is a sentence. Anotha sentence.")
#     assert len(list(doc.sents)) == 2


# Using spaCy splits pages_and_texts.text to multiple sentences
def split_texts_to_sentences(pages_and_texts) -> list[dict]:
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts

def main():
    pages_and_texts = open_and_read_pdf("data/oracle/March-12(Quarterly)-24.pdf")
    # The workflow which we are following:
    # Ingest text -> split it into groups/chunks -> embed the groups/chunks -> use the embeddings
    random_pages_and_texts = random.sample(pages_and_texts, k=3)
    # print(random_pages_and_texts)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    # df = pd.DataFrame(pages_and_texts)
    # print(df.head())
    # print(df.describe().round(2))
    pages_and_texts = split_texts_to_sentences(pages_and_texts)
    # print(random.sample(pages_and_texts, k=1))
    df = pd.DataFrame(pages_and_texts)
    # For our set of text, it looks like our raw sentence count
    # (e.g. splitting on ". ") is quite close to what spaCy came up with.
    print(df.describe().round(2))

    # Chunk our sentences together

if __name__ == '__main__':
    main()
