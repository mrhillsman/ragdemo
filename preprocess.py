import fitz
import os
import re
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
import torch
from tqdm.auto import tqdm

# Check if we have a GPU that supports cuda if not fallback to cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Number of sentences to chunk together
num_sentence_chunk_size = 10
page_skip_mappings = json.load(open("pdf_page_skip_mappings.json"))
regex_mappings = json.load(open("pdf_regex_mappings.json"))


def general_formatter(text: str) -> str:
    # Perform any general text formatting here
    # Remove new lines to squash text into sequential sentences
    formatted_text = re.sub(r'([A-Z]+\s)*([0-9]\.)([0-9]\.)?', '', text)
    formatted_text = re.sub(r'([A-Z]{4,}\s)', '', formatted_text)
    formatted_text = formatted_text.replace("\n", " ").strip()
    # Additional code to perform any further formatting changes overall
    return formatted_text


# Split a list into chunks of a specified size returning a list of lists
def split_list(input_list: list[str], slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]


# Open and read the contents of a single PDF return a list of dicts containing raw information about each page of the
# PDF document: the name of the pdf read page_number: page number text was read from page_char_count: total number of
# characters on the page page_word_count: total number of words as obtained by splitting on space " "
# page_sentence_count_raw: total number of sentences as obtained by splitting on ". " page_token_count: total number
# of tokens (https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) text: the entire text
# found on the page
def open_and_read_pdf(pdf_path: str, page_skips: int) -> list[dict]:
    doc = fitz.open(pdf_path)
    built_pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        if page_number < page_skips:
            continue
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
        print(f"Processing {pdf}...")
        pdf_location = pdfs_dir + pdf
        page_skips = page_skip_mappings[pdf_location.split("-")[-3]]
        docs[pdf_location.split("-")[-3]] = open_and_read_pdf(pdf_location, page_skips)
    return docs


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
print("Using spaCy's sentencizer to split text into sentences...")
for pdf in pdfs:
    for page in tqdm(pdfs_pages_and_texts[pdf]):
        # Split text into sentences using spaCy's sentencizer
        page["sentences"] = list(nlp(page["text"]).sents)
        # Convert sentences into strings from spaCy's default datatype
        page["sentences"] = [str(sentence) for sentence in page["sentences"]]
        # Store the number of sentences created by spaCy
        page["page_sentence_count_spacy"] = len(page["sentences"])


# Loop over each PDF from the list of PDFs and split the sentences into chunks of a specified size
print(f"Splitting sentences into lists of sentence chunks of {num_sentence_chunk_size}...")
for pdf in pdfs:
    for page in pdfs_pages_and_texts[pdf]:
        # Split sentences into chunks of a specified size
        # Our purpose here is to chunk sentences together to create a more meaningful embedding
        page["sentence_chunks"] = split_list(input_list=page["sentences"],
                                             slice_size=num_sentence_chunk_size)
        # Store the number of chunks created
        page["num_chunks"] = len(page["sentence_chunks"])


# Create a list of dictionaries containing the document name, page number, and the sentence chunk
pages_and_chunks = []
for pdf in pdfs:
    for page in pdfs_pages_and_texts[pdf]:
        for sentence_chunk in page["sentence_chunks"]:
            # Create a dictionary containing the document name, page number, and the sentence chunk
            chunk_dict = {"document": pdf, "page_number": page["page_number"]}
            # Join the sentence chunk into a single string and remove any double spaces
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            # Store the joined sentence chunk, the character count, word count, and token count
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
            # Append the chunk dictionary to the pages_and_chunks list
            pages_and_chunks.append(chunk_dict)


# Create a Pandas DataFrame from the pages_and_chunks list of dictionaries
# This will allow us to easily filter and manipulate the data
df = pd.DataFrame(pages_and_chunks)
min_token_length = 30

# Sample 5 sentence chunks that are less than the min_token_length to determine if they are viable candidates
# for embedding and/or ensure our min_token_length is not causing us to exclude viable candidates for embedding
# for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
#     print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

# Filter out sentence chunks that are less than the min_token_length
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
print("Finished processing PDFs...")

print("Creating embeddings...")
# Setup the SentenceTransformer model for embedding
embedding_model = SentenceTransformer('f0xn0v4/redhatdoc-MiniLM-L12-v1', device=device)

# Batch process the sentence chunks to create embeddings
text_chunks = [content["sentence_chunk"] for content in pages_and_chunks_over_min_token_len]
embeddings = embedding_model.encode(text_chunks, batch_size=32, show_progress_bar=True, device=device)

# Add the embeddings to the pages_and_chunks_over_min_token_len list of dictionaries
for idx, content in enumerate(pages_and_chunks_over_min_token_len):
    content["embedding"] = embeddings[idx]

# Print total number of embeddings created
print(f"Total number of embeddings created: {len(embeddings)}")

print("Save pages and chunks to disk...")
# Save the pages_and_chunks_over_min_token_len list of dictionaries to disk
chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
chunks_and_embeddings_df.to_csv("chunks_and_embeddings.csv", escapechar='\\', index=False)


# Create a collection in ChromaDB to store the embeddings
# print("Adding embeddings to ocp-4-15-embeddings collection...")
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="f0xn0v4/redhatdoc-MiniLM-L12-v1", device=device)

# Create a ChromaDB client and collection
# chromaClient = chromadb.PersistentClient("chromadb")
# collection = chromadb.Collection

# try:
#     # Get the collection if it exists
#     collection = chromaClient.get_collection("ocp-4-15-embeddings", embedding_function=sentence_transformer_ef)
# except ValueError or KeyError or TypeError as e:
#     # Create the collection if it does not exist
#     collection = chromaClient.create_collection("ocp-4-15-embeddings", embedding_function=sentence_transformer_ef)
#     # Check if the collection count is the same as the number of embeddings we expect to add
#     if collection.count() != len(pages_and_chunks_over_min_token_len):
#         # Add the embeddings to the collection
#         for i, content in tqdm(enumerate(pages_and_chunks_over_min_token_len)):
#             print(f"Adding embedding {i} to collection...")
#             collection.add(
#                 # Add the document name and page number as the ID
#                 ids=[f"embedding{i}"],
#                 # Add the document name, page number, character count, word count, and token count as metadata
#                 metadatas=[{"document": content["document"],
#                             "page_number": content["page_number"],
#                             "chunk_char_count": content["chunk_char_count"],
#                             "chunk_word_count": content["chunk_word_count"],
#                             "chunk_token_count": content["chunk_token_count"]}],
#                 # Add the sentence chunk as the document
#                 documents=content["sentence_chunk"],
#             )
# else:
#     print(f"Collection already exists with expected count of {collection.count()} embeddings.")

