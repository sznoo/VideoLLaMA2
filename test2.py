from FlagEmbedding import FlagAutoModel
import torch


if __name__ == "__main__":
    model = FlagAutoModel.from_finetuned(
        "BAAI/bge-base-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=True,
    )
    # device = torch.device("cpu")
    # model.to(device)
    sentences_1 = ["I love NLP", "I love machine learning"]
    sentences_2 = ["I love BGE", "I love text retrieval"]
    embeddings_1 = model.encode(sentences_1)
    embeddings_2 = model.encode(sentences_2)

    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
