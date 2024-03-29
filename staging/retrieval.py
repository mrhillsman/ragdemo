import numpy as np
import pandas as pd
import textwrap
import torch
from sentence_transformers import util, SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_model = SentenceTransformer('f0xn0v4/redhatdoc-MiniLM-L12-v1', device=device)

chunks_and_embeddings_df = pd.read_csv("chunks_and_embeddings.csv")
chunks_and_embeddings_df["embedding"] = (chunks_and_embeddings_df["embedding"]
                                         .apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")))
chunks_and_embeddings_records = chunks_and_embeddings_df.to_dict(orient="records")
embeddings = torch.tensor(np.array(chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)


def retrieve_relevant_docs(query: str, embeddings: torch.tensor = embeddings,
                           model: SentenceTransformer = embedding_model,
                           n_resources_to_return: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    dotscores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dotscores, k=n_resources_to_return)
    return scores, indices


def wrap_text(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def debug_view(query: str, embeddings: torch.tensor = embeddings,
               chunks_and_embeddings: list[dict] = chunks_and_embeddings_records):
    scores, indices = retrieve_relevant_docs(query=query, embeddings=embeddings)
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Content:")
        wrap_text(chunks_and_embeddings[idx]["sentence_chunk"])
        print(f"Page number: {chunks_and_embeddings[idx]['page_number']}")
        print(f"Document: {chunks_and_embeddings[idx]['document']}")
        print("\n")
