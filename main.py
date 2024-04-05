import json
import time
import uuid

from tqdm.auto import tqdm
import random
import pandas as pd
import pytest
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI


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
        pages_and_texts.append({"file_name": pdf_path,
                                "page_number": page_number - 1,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text,
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
            chunk_dict["file_name"] = item["file_name"]

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


def chunk_and_embed(file_name_with_path):
    # pages_and_texts = open_and_read_pdf("data/oracle/March-12(Quarterly)-22.pdf")
    pages_and_texts = open_and_read_pdf(file_name_with_path)
    # pages_and_texts = open_and_read_pdf("data/oracle/March-12(Quarterly)-22.pdf")
    # pages_and_texts = open_and_read_pdf("data/oracle/September-12(Quarterly)-22.pdf")
    # The workflow which we are following:
    # Ingest text -> split it into groups/chunks -> embed the groups/chunks -> use the embeddings
    # random_pages_and_texts = random.sample(pages_and_texts, k=3)
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

    # for item in pages_and_chunks_over_min_token_len:
    #     print(item["sentence_chunk"])

    # print(df.head())
    # print(df.describe().round(2))

    # using all-mpnet-base-v2 model, which has embedding vector size of 768. Max token it can accept is 384
    # 109 million params Rank 67
    # embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
    #                                       device="cpu")

    # using all-MiniLM-L12-v2 model, which has max token (to be verified) with 33.4M params.
    # Rank: 75 embedding vector size of 384
    # whether smaller vector size encodes more information into embedding is currently questionable,
    # since smaller, better model are outperforming larger ones.
    embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/all-MiniLM-L12-v2",
                                          device="cpu")

    # This is the best model in the leaderboard, but takes a long time to load model since it has 7.11 billion params
    # embedding_model = SentenceTransformer(model_name_or_path="Salesforce/SFR-Embedding-Mistral",
    #                                       device="cpu")

    # This is the best model in the leaderboard, but takes a long time to load model since it has 7.11 billion params
    # Anything with billion params I can't test with CPU, GPU will work.
    # embedding_model = SentenceTransformer(model_name_or_path="wvprevue/e5-mistral-7b-instruct",
    #                                       device="cpu")

    # waited for 5 mins, no response while creating the model itself.
    # From their website I understand it's a priced model
    # 335 M params
    # embedding_model = SentenceTransformer(model_name_or_path="mixedbread-ai/mxbai-embed-large-v1",
    #                                       device="cpu")

    # Below code is to check the execution time of creating embeddings
    # start = time.time()
    for item in tqdm(pages_and_chunks_over_min_token_len):
        # item["embedding"] = embedding_model.encode(item["sentence_chunk"],
        #                                            batch_size=32,).tolist()
        item["embedding"] = embedding_model.encode(item["sentence_chunk"], ).tolist()

    # Turn text chunks into a single list
    # text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
    # start = time.time()
    # text_chunk_embeddings = embedding_model.encode(text_chunks,
    #                                                batch_size=32,
    #                                                show_progress_bar=True)
    # print(len(text_chunk_embeddings))

    # end = time.time()
    # print("Execution time: ", (end - start), "s")
    return pages_and_chunks_over_min_token_len


def main():
    pages_and_chunks = chunk_and_embed("data/oracle/March-12(Quarterly)-22.pdf")
    # print(pages_and_chunks)
    # print(pages_and_chunks[0])

    vector_store = MilvusVectorStore(dim=384,
                                     collection_name="chunked_store",
                                     overwrite=True,
                                     )

    # vector_store = MilvusVectorStore(collection_name="chunked_store",
    #                                  overwrite=True,
    #                                  )
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    milvus_records = []
    # print(type(pages_and_chunks), type(pages_and_chunks[0]))
    for item in tqdm(pages_and_chunks):
        # record = {"id_": uuid.uuid4(), "embedding": (None,), "metadata": item}
        record = TextNode(id_=str(uuid.uuid4()),
                          embedding=item["embedding"],
                          text=item["sentence_chunk"],
                          metadata={"filename": item["file_name"],
                                    "page_number": item["page_number"]})
        milvus_records.append(record)
        # print(item)
    # print(milvus_records)
    vector_store.add(nodes=milvus_records)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    print("success")
    # vector_store.add(nodes=pages_and_chunks)
    #
    # index = VectorStoreIndex.from_vector_store(vector_store)
    # llm = OpenAI(model="gpt-3.5-turbo")
    # query_engine = index.as_query_engine(llm=llm)
    # response = query_engine.query("What is the total revenues of Oracle corporation by February 29 2024?")
    # print(response)


if __name__ == '__main__':
    main()
