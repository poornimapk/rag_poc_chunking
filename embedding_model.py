from sentence_transformers import SentenceTransformer


def main():
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                          device="cpu")

    # Create a list of sentences to turn into numbers
    sentences = [
        "My name is Mark Zuckerberg.",
        "I created Facebook.",
        "TheFacebook, Inc., is an American multinational technology conglomerate based in Menlo Park, California.",
        "I hated creating Facebook."
    ]

    # Sentences are encoded/embedded by calling model.encode()
    embeddings = embedding_model.encode(sentences)
    embeddings_dict = dict(zip(sentences, embeddings))

    # See the embeddings
    for sentence, embedding in embeddings_dict.items():
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")


if __name__ == '__main__':
    main()
