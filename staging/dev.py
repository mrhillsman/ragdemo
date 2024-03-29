import fitz
import chromadb
import torch
from chromadb.utils import embedding_functions
import os
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English

# Check if we have a GPU that supports cuda if not fallback to cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Number of sentences to chunk together
num_sentence_chunk_size = 10


# General manipulation of text, could be handled by a library later
def general_formatter(text: str) -> str:
    """Remove new lines to squash text into sequential sentences"""
    formatted_text = text.replace("\n", " ").strip()
    return formatted_text


def split_list(input_list: list[str], slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]


# Open and read the contents of a single PDF return a list of dicts containing raw information about each page of the
# PDF document: the name of the pdf read page_number: page number text was read from page_char_count: total number of
# characters on the page page_word_count: total number of words as obtained by splitting on space " "
# page_sentence_count_raw: total number of sentences as obtained by splitting on ". " page_token_count: total number
# of tokens (https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) text: the entire text
# found on the page
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    built_pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = general_formatter(text=text)
        built_pages_and_texts.append({"page_number": page_number,
                                      "page_char_count": len(text),
                                      "page_word_count": len(text.split(" ")),
                                      "page_sentence_count_raw": len(text.split(". ")),
                                      "page_token_count": len(text) / 4,
                                      "text": text})
    return built_pages_and_texts


# Open and read multiple PDFs returning a list containing each PDF as a list
# of dicts containing raw information about each page of the PDF as noted in the open_and_read_pdf function
def open_and_read_pdfs(pdfs_dir: str, pdfs_list: list) -> dict:
    docs = {}
    for pdf in pdfs_list:
        pdf_location = pdfs_dir + pdf
        docs[pdf_location.split("-")[-3]] = open_and_read_pdf(pdf_location)

    return docs


# Example of obtaining a single PDF
# pages_and_texts = open_and_read_pdf('ocp-4-15-pdfs/01-openshift_container_platform-4.15-getting_started-en-us.pdf')

# Get all PDFs from the directory and read them returning a list of PDFs
# that are lists of dicts containing raw information about each page of the PDFs
pdf_dir = 'ocp-4-15-pdfs/'
pdf_list = os.listdir(pdf_dir)
print("Opening and processing PDFs...")
pdfs_pages_and_texts = open_and_read_pdfs(pdf_dir, pdf_list)
pdfs = list(pdfs_pages_and_texts.keys())


# Use spaCy's sentencizer for sentence boundary detection
# https://spacy.io/api/sentencizer
# nltk and other similar tools could be used
# looking to invest in learning unstructured.io for ingestion and detection/pre-processing
nlp = English()
nlp.add_pipe('sentencizer')


# Loop over each PDF from the list of PDFs and use sentencizer to split text into sentences
# then convert each sentence into a string and store a count of sentences created by spaCy
for pdf in pdfs:
    for page in pdfs_pages_and_texts[pdf]:
        page["sentences"] = list(nlp(page["text"]).sents)
        # Convert sentences into strings from spaCy default datatype
        page["sentences"] = [str(sentence) for sentence in page["sentences"]]
        page["page_sentence_count_spacy"] = len(page["sentences"])


for pdf in pdfs:
    for page in pdfs_pages_and_texts[pdf]:
        page["sentence_chunks"] = split_list(input_list=page["sentences"],
                                             slice_size=num_sentence_chunk_size)
        page["num_chunks"] = len(page["sentence_chunks"])


pages_and_chunks = []
for pdf in pdfs:
    for page in pdfs_pages_and_texts[pdf]:
        for sentence_chunk in page["sentence_chunks"]:
            chunk_dict = {"document": pdf, "page_number": page["page_number"]}
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            # joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

            pages_and_chunks.append(chunk_dict)


df = pd.DataFrame(pages_and_chunks)
min_token_length = 30

# Sample 5 sentence chunks that are less than the min_token_length to determine if they are viable candidates
# for embedding and/or ensure our min_token_length is not causing us to exclude viable candidates for embedding
# for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
#     print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
print("Finished processing PDFs...")

# View a random pages_and_chunks record
# print(random.sample(pages_and_chunks_over_min_token_len, k=1))

# View the first 5 rows of pages_and_chunks via Pandas DataFrame
# df = pd.DataFrame(pages_and_chunks)
# print(df.head())
# print(df.describe().round(2))

# embedding_model = SentenceTransformer('f0xn0v4/redhatdoc-MiniLM-L12-v1')
# print("Creating embeddings...")
# for content in tqdm(pages_and_chunks_over_min_token_len):
#     content["embedding"] = embedding_model.encode(content["sentence_chunk"], batch_size=32, convert_to_tensor=True)

# An example of batching embeddings speeding up the process by utilizing more of the GPU
# text_chunks = [content["sentence_chunk"] for content in pages_and_chunks_over_min_token_len]
# embeddings = embedding_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)

print("Adding embeddings to ocp-4-15-embeddings collection...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="f0xn0v4/redhatdoc-MiniLM-L12-v1", device=device)

chromaClient = chromadb.PersistentClient("chromadb")
collection = chromadb.Collection

try:
    collection = chromaClient.get_collection("ocp-4-15-embeddings")
except:
    collection = chromaClient.create_collection("ocp-4-15-embeddings", embedding_function=sentence_transformer_ef)
else:
    for i, content in enumerate(pages_and_chunks_over_min_token_len):
        collection.add(
            ids=[f"embedding{i}"],
            metadatas=[{"document": content["document"],
                        "page_number": content["page_number"],
                        "chunk_char_count": content["chunk_char_count"],
                        "chunk_word_count": content["chunk_word_count"],
                        "chunk_token_count": content["chunk_token_count"]}],
            documents=content["sentence_chunk"]
        )

# chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
# print(random.sample(pages_and_chunks_over_min_token_len, k=1))
# embeddings_df_save_path = "chunks_and_embeddings_df.parquet"
# text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, escapechar="\\", index=False)
# chunks_and_embeddings_df.to_parquet(embeddings_df_save_path, index=False)

# text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path) text_chunks_and_embedding_df["embedding"] =
# text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x))

# print(random.sample(pages_and_chunks_over_min_token_len, k=1))
# text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
