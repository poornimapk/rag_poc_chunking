from tqdm.auto import tqdm
import random
import pandas as pd
import pytest
from spacy.lang.en import English
import re

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
        item["sentences"] = [str(sentence).strip() for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts


def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    # tmp_list = []
    # for i in range(0, len(input_list), slice_size):
    #     tmp_list.append(input_list[i:i + slice_size])
    # print(tmp_list)
    # return tmp_list
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def generate_pages_and_chunks(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace(" ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.(A-Z)', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about chunks
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk)

            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks




def main():
    pages_and_texts = open_and_read_pdf("data/oracle/March-12(Quarterly)-24.pdf")
    # pages_and_texts = open_and_read_pdf("data/oracle/March-12(Quarterly)-22.pdf")
    # pages_and_texts = open_and_read_pdf("data/oracle/September-12(Quarterly)-22.pdf")
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
    item: list = random.sample(pages_and_texts, k=1)
    num_sentence_chunk_size = 10

    # Chunk our sentences together
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                             slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    # print(random.sample(pages_and_texts, k=1))

    pages_and_chunks = generate_pages_and_chunks(pages_and_texts)
    # print(len(pages_and_chunks))
    # print(random.sample(pages_and_chunks, k=1))
    # print(item)
    df = pd.DataFrame(pages_and_chunks)
    # Show random chunks with under 30 tokens in length
    min_token_length = 30

    # Below code is to check any chunks are there with <= 30 which we want to remove from our chunks list
    # They are mostly header footer details which does not make sense to add to the chunks
    # count_of_chunks = 1
    # for row in df[df["chunk_token_count"] <= min_token_length].iterrows():
    #     print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')
    #     count_of_chunks += 1
    # print(count_of_chunks)

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    print(pages_and_chunks_over_min_token_len)

    # print(df.head())
    # print(df.describe().round(2))






if __name__ == '__main__':
    main()
