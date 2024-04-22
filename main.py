from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
import re
from llama_index.core import Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
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
                                "page_number": page_number + 1,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text,
                                })
    # print(pages_and_texts)
    return pages_and_texts


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


def chunk_pdf(file_name_with_path) -> list[dict]:
    pages_and_texts = open_and_read_pdf(file_name_with_path)
    # print(pages_and_texts)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)

    pages_and_texts = split_texts_to_sentences(pages_and_texts)

    num_sentence_chunk_size = 10

    # Chunk our sentences together
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                             slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    pages_and_chunks = generate_pages_and_chunks(pages_and_texts)
    # print(len(pages_and_chunks))
    df = pd.DataFrame(pages_and_chunks)
    # Show random chunks with under 30 tokens in length
    min_token_length = 30

    # Below code is to check any chunks are there with <= 30 which we want to remove from our chunks list
    # They are mostly header footer details which does not make sense to add to the chunks
    count_of_chunks = 1
    for row in df[df["chunk_token_count"] <= min_token_length].iterrows():
        count_of_chunks += 1

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

    return pages_and_chunks_over_min_token_len


def create_documents_from_chunks(pages_and_chunks) -> list[Document]:
    document_list = []
    for item in tqdm(pages_and_chunks):
        document = Document(
            text=item["sentence_chunk"],
            metadata={"page_number": item["page_number"],
                      "file_name": item["file_name"],
                      "file_type": 'application/pdf',
                      # "file_size": None, TODO
                      #                     # "creation_date": None, TODO
                      #                     # "last_modified_date": None, TODO
                      },
            excluded_llm_metadata_keys=[item["file_name"]],
        )
        document_list.append(document)
    return document_list


def main():
    load_dotenv()
    # pages_and_chunks = chunk_pdf("data/oracle/March-12(Quarterly)-24.pdf")
    pages_and_chunks = chunk_pdf("data/oracle/paul-graham-ideas.pdf")
    # print(pages_and_chunks[0])
    documents = create_documents_from_chunks(pages_and_chunks)
    print("Success so far")
    # print(len(documents))

    vector_store = MilvusVectorStore(dim=1536,
                                     collection_name="chunked_rag_store",
                                     overwrite=True)
    # vector_store = MilvusVectorStore(dim=1536,
    #                                  collection_name="chunked_rag_store",
    #                                  overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.llm = Ollama(model="llama3")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("More Success so far")

    query_engine = index.as_query_engine()
    # response = query_engine.query("What is Oracle's marketable securities in millions by end of February 29 2024?")
    response = query_engine.query("Who is Paul Graham?")
    print(response)


if __name__ == '__main__':
    main()
