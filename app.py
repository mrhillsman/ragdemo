import chainlit as ct
import os
import numpy as np
import pandas as pd
import re
from sentence_transformers import util, SentenceTransformer
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers.utils import is_flash_attn_2_available

# Check if we have a GPU that supports cuda if not fallback to cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting the embedding model; should be the same used to create the initial embeddings in preprocess.py
embedding_model = SentenceTransformer('f0xn0v4/redhatdoc-MiniLM-L12-v1', device=device)

# Load the embeddings from disk
chunks_and_embeddings_df = pd.read_csv("chunks_and_embeddings.csv")
chunks_and_embeddings_df["embedding"] = (chunks_and_embeddings_df["embedding"]
                                         .apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")))
chunks_and_embeddings_records = chunks_and_embeddings_df.to_dict(orient="records")
# Load the embeddings into the GPU
embeddings = torch.tensor(np.array(chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)


# Using the query submitted by user get embeddings that are most similar based on dot product
def retrieve_relevant_docs(query: str, embeddings: torch.tensor = embeddings,
                           model: SentenceTransformer = embedding_model,
                           n_resources_to_return: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    dotscores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dotscores, k=n_resources_to_return)
    return scores, indices


# Wrap output after 80 characters
def wrap_text(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


# View embeddings pulled that are most similar to query submitted by user
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


# Setup quantization config using bitsandbytes library
# More about quantization and bitsandbytes can be found here:
# https://huggingface.co/docs/transformers/quantization
# https://huggingface.co/docs/bitsandbytes/main/en/explanations/resources
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    print("Flash attention is available.")
    # https://arxiv.org/abs/2307.08691 https://github.com/Dao-AILab/flash-attention
    attn_implementation = "flash_attention_2"
else:
    print("Flash attention is not available or the GPU has a compute capability of less than 8.0.")
    attn_implementation = "sdpa"  # Scaled Dot-Product Attention:  https://paperswithcode.com/method/scaled

use_quantization_config = False
model_id = "google/gemma-2b-it"

# HuggingFace read only API key
api_key = os.getenv("HF_TOKEN")

# Automatically select the tokenizer and model for the specified model_id
# https://huggingface.co/docs/transformers/en/main_classes/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=api_key)

# https://huggingface.co/docs/transformers/tasks/language_modeling
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                 config=quantization_config if use_quantization_config else None,
                                                 low_cpu_mem_usage=False,
                                                 attn_implementation=attn_implementation,
                                                 token=api_key).to(device)


# This is prompt engineering - https://www.promptingguide.ai/
# We are setting up the conversation template to have with our model
# This should be improved in order to attempt to get more relevant responses
@ct.step
async def engineer_prompt(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([context_item["sentence_chunk"] for context_item in context_items])
    instruction = """Based on the following context items, answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following context items to answer the user query:
{context}
User query: {query}
Answer:"""
    base_prompt = instruction.format(context=context, query=query)
    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False)
    await ct.sleep(2)
    return prompt


print("Model loaded and ready for use.")


# This function takes care of sending the prompt with instruction+context+query to the LLM
@ct.step
async def ask_llm(prompt: str) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**input_ids, max_new_tokens=2048, temperature=0.7, do_sample=True)
    outputs_decoded = tokenizer.decode(outputs[0])
    await ct.sleep(2)
    return outputs_decoded.strip()


# This is our chainlit main function which handles interactions with the chat frontend
@ct.on_message
async def main(message: ct.Message):
    scores, indices = retrieve_relevant_docs(message.content)
    context_items = [chunks_and_embeddings_records[i] for i in indices]
    prompt = await engineer_prompt(query=message.content, context_items=context_items)
    response = await ask_llm(prompt)
    pattern = re.compile(r'(?<=<end_of_turn>)([\s\S]*?)(?=<eos>)')
    msg = pattern.findall(response)

    await ct.Message(
        content=''.join(msg)
    ).send()
