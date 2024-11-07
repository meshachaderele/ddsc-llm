import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
import argparse
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from copy import copy
import torch 

def generate_prompt(language, positive, title):
    prompt = (
        f"Your task is to anticipate possible search queries by users in the form of a question for a given document.\n"
        f"- The question must be written in {language}\n"
        f"- The question should be formulated concretely and precisely and relate to the information from the given document\n"
        f"- The question must be coherent and should make sense without knowing the document\n"
        f"- The question must be answerable by the document\n"
        f"- The question should focus on one aspect and avoid using subclauses connected with 'and'\n"
        f"- The question should not be overly specific and should mimic a request of a user who is just starting to research the given topic\n"
        f"- It should be clear what named entity the question is about. Do not use anaphora\n"
        f"- Do not draw on your prior knowledge\n\n"
        f"Generate a question in {language} for the following document:\n"
        f"<document>\n{positive}\n</document>\n\n"
        f"The title of the document is:\n"
        f"<title>{title}</title>\n\n"
        f"Search query:"
    )
    return prompt


def format_example(example: dict) -> dict:
    # Helper function for making the "prompt" column in the dataset
    prompt = generate_prompt(language, example["positive"], example["title"])
    return {"prompt": [{"role": "user", "content": prompt}]}

# Create an argument parser
parser = argparse.ArgumentParser(description="Parse a Hugging Face token from the terminal.")

# Add an argument for the Hugging Face token
parser.add_argument('--token', type=str, default="none", required=False, help="Your Hugging Face token")

parser.add_argument('--save_to', type=str, default="DDSC/da-wikipedia-queries", required=True, help="What huggingface repo to push the generated dataset to")

parser.add_argument('--language', type=str, default="Danish", required=True, help="Language of the generated queries")

parser.add_argument(
    '--dataset_path', 
    type=str, 
    default="eshachaderele/negative-positive-wikipedia-2023-11-da", 
    required=True, 
    help="Huggingface dataset name, e.g. meshachaderele/negative-positive-wikipedia-2023-11-da"
    )

parser.add_argument(
    "--models",
    nargs="*",
    help="List of one or more models (HF path) to use to generate the queries",
    required=True
    )

parser.add_argument(
    "--n_samples",
    type=int,
    help="We will generate a query for the first n_samples. Set to -1 for all",
    required=False
)

# Parse the arguments
args = parser.parse_args()

# Access the token
token = args.token

login(token, add_to_git_credential=True)

models = args.models

language = args.language 

n_samples = args.n_samples
if not n_samples:
    n_samples = -1

dataset_path = args.dataset_path #"meshachaderele/negative-positive-wikipedia-2023-11-da"

save_to = args.save_to
#hf_model = "ThatsGroes/Llama-3-8B-instruct-AI-Sweden-SkoleGPT"
#hf_model = "CohereForAI/aya-expanse-32b"
#hf_model = "google/gemma-2-27b-it"

dataset = load_dataset(dataset_path, split = "train")

# Apply map with the improved function
dataset = dataset.map(format_example)

if n_samples > 0:
    dataset = dataset.select(range(n_samples))

all_results = []

energy_use = []

for model in models:

    results = copy(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)

    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512)

    llm = LLM(model=model, max_seq_len_to_capture=8000)

    # Log some GPU stats before we start inference
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(
        f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
        f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
    )

    tracker = EmissionsTracker()
    tracker.start()
    outputs = llm.chat(dataset["prompt"], sampling_params)
    emissions = tracker.stop()
    print(f"Emissions from generating queries with {model}:\n {emissions}")
    energy_consumption_kwh = tracker._total_energy.kWh  # Total energy in kWh
    print(f"Energy consumption from generating queries with {model}:\n {emissions}")

    responses = [output.outputs[0].text for output in outputs]

    results = results.add_column("query", responses)

    results = results.add_column("model", [model for _ in range(len(results))])
    
    # number of tokens in the prompt and response. Used for calculcating kwh/token
    results = results.add_column("num_tokens_query", [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]) # [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]

    # each element in results["prompt"] is a list with a dictionary with two keys: "content" and "role"
    results = results.add_column("num_tokens_prompt", [len(tokenizer.encode(text[0]["content"], add_special_tokens=False)) for text in results["prompt"]])

    all_results.append(results)

    # Print some post inference GPU stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_inference = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    inference_percentage = round(used_memory_inference / max_memory * 100, 3)

    energy_use.append({
        "model" : model, 
        "energy_use_kwh" : energy_consumption_kwh, 
        "num_tokens_query" : sum(results["num_tokens_query"]), 
        "num_tokens_prompt" : sum(results["num_tokens_prompt"]),
        "num_tokens_total" : sum(results["num_tokens_query"]) + sum(results["num_tokens_prompt"])
        })

    print(
        f"We ended up using {used_memory:.2f} GB GPU memory ({used_percentage:.2f}%), "
        f"of which {used_memory_inference:.2f} GB ({inference_percentage:.2f}%) "
        "was used for inference."
    )

    torch.cuda.empty_cache()

    # torch.cuda.empty_cache does not properly free up memory
    del llm 

energy_use = pd.DataFrame.from_records(energy_use)

energy_use.to_csv("energy_use_per_model.csv", index=False)

final_dataset = concatenate_datasets(all_results)

print(f"Final dataset: \n {final_dataset}")

#for row in final_dataset.select(range(5)):
#    print(row)
#    print("\n\n")

for prompt, response in zip(final_dataset.select(range(5))["prompt"], responses):
    print(f"Prompt:\n {prompt}\n\n Response: {response}\n")
    print("----------")

tokens = sum([len(tokenizer.encode(text, add_special_tokens=False)) for text in responses])

print(f"Num tokens in dataset: {tokens}")

final_dataset.push_to_hub(save_to)
