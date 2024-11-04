import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
import argparse
from huggingface_hub import login


def generate_prompt(language, positive):
    prompt = (
        f"Your task is to anticipate possible search queries by users in the form of a question for a given document.\n"
        f"- The question must be written in {language}\n"
        f"- The question should be formulated concretely and precisely and relate to the information from the given document\n"
        f"- The question must be coherent and should make sense without knowing the document\n"
        f"- The question must be answerable by the document\n"
        f"- The question should focus on one aspect and avoid using subclauses connected with 'and'\n"
        f"- The question should not be overly specific and should mimic a request of a user who is just starting to research the given topic\n"
        f"- Do not draw on your prior knowledge\n\n"
        f"Generate a question in {language} for the following document:\n"
        f"<document>\n{positive}\n</document>\n\n"
        f"Search query:"
    )
    return prompt

# Create an argument parser
parser = argparse.ArgumentParser(description="Parse a Hugging Face token from the terminal.")

# Add an argument for the Hugging Face token
parser.add_argument('--token', type=str, default="none", required=False, help="Your Hugging Face token")

# Parse the arguments
args = parser.parse_args()

# Access the token
token = args.token

login(token, add_to_git_credential=True)

#hf_model = "ThatsGroes/Llama-3-8B-instruct-AI-Sweden-SkoleGPT"
hf_model = "CohereForAI/aya-expanse-32b"

df = pd.read_parquet("processed_danish_wikipedia.parquet")
df = df[:10]

language = "Danish" 

# Make list of queries for VLLM to process
prompts = []
for _, row in df.iterrows():
    positive = row['positive']
    prompt = generate_prompt(language, positive)
    # for llm.generate()
    #prompts.append(prompt)

    # for llm.chat(), the input must be a list of conversations, where each conversation is a list of messages. Each message is a dictionary with "role" and "content"
    # https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
    prompts.append([{"role" : "user", "content" : prompt}])


tokenizer = AutoTokenizer.from_pretrained(hf_model, token=token)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

llm = LLM(model=hf_model)

tracker = EmissionsTracker()
tracker.start()
outputs = llm.chat(prompts, sampling_params)
emissions = tracker.stop()
print(emissions)



#print(outputs.__repr__())


# Print the outputs.
responses = [output.outputs[0].text for output in outputs]

for prompt, response in zip(prompts, responses):
    print(f"Prompt:\n {prompt}\n\n Response: {response}\n")
    print("----------")

tokens = sum([len(tokenizer.encode(text, add_special_tokens=False)) for text in responses])

print(f"Num tokens in dataset: {tokens}")