import random

import torch
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

text_chunks_and_embeddings_df = pd.read_csv('text_chunks_and_embeddings_df.csv')
pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient='records')

single = text_chunks_and_embeddings_df.sample(1)
print(single["embedding"])
print(pages_and_chunks.pop())

# text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.frombuffer(bytes(x.strip('[]'), 'utf-8'), offset=8))

# embeddings = np.stack(text_chunks_and_embeddings_df["embedding"].tolist(), axis=0)
# print(embeddings[1])
# embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)
# embeddings = torch.tensor(np.stack(text_chunks_and_embeddings_df["embedding"].tolist(), axis=0))

# pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient='records')

# print(embeddings.shape)

