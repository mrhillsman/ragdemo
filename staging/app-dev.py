import chainlit as ct
import torch
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available


device = 'cuda' if torch.cuda.is_available() else 'cpu'

chromaClient = chromadb.PersistentClient("../chromadb")
collection = chromaClient.get_collection("ocp-4-15-embeddings")

embedding_model = SentenceTransformer('f0xn0v4/redhatdoc-MiniLM-L12-v1', device=device)

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)


def retrieve_relevant_docs(query: str,
                           model: SentenceTransformer = embedding_model,
                           n_resources_to_return: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_resources_to_return)

    return results


if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    print("Flash attention is available.")
    attn_implementation = "flash_attention_2"
else:
    print("Flash attention is not available or the GPU has a compute capability of less than 8.0.")
    attn_implementation = "sdpa"  # Scaled Dot-Product Attention


gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
model_id = ""


# Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for
# local use.
if gpu_memory_gb < 5.1:
    print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally "
          f"without quantization.")
elif gpu_memory_gb < 8.1:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
    use_quantization_config = True 
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
    use_quantization_config = False 
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb > 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
    use_quantization_config = False 
    model_id = "google/gemma-7b-it"

print(f"use_quantization_config set to: {use_quantization_config}")
print(f"model_id set to: {model_id}")

use_quantization_config = True
# model_id = "google/gemma-7b-it"

api_key = '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'

tokenizer = AutoTokenizer.from_pretrained(model_id, token=api_key)

llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                 config=quantization_config if use_quantization_config else None,
                                                 low_cpu_mem_usage=False,
                                                 attn_implementation=attn_implementation,
                                                 token=api_key)
llm_model.to("cuda")

print("Model loaded and ready for use.")


@ct.on_message
async def main(message: ct.Message):
    # results = retrieve_relevant_docs(message)
    
    dialogue_template = [{"role": "user", "content": message.content}]

    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**input_ids, max_new_tokens=256)
    outputs_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    await ct.Message(
        content=outputs_decoded,
    ).send()
