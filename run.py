import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker


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

hf_model = "AI-Sweden-Models/Llama-3-8B-instruct"

df = pd.read_parquet("processed_danish_wikipedia.parquet")
df = df[:10]

language = "Danish" 

# Make list of queries for VLLM to process
prompts = []
for _, row in df.iterrows():
    positive = row['positive']
    prompt = generate_prompt(language, positive)
    prompts.append(prompt)

# Load the appropriate tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/Llama-3-8B-instruct")

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=hf_model)

tracker = EmissionsTracker()
tracker.start()
outputs = llm.generate(prompts, sampling_params)
emissions = tracker.stop()
print(emissions)



print(outputs.__repr__())


# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    token_count = len(tokens)
    #token_count = len(output.outputs[0].token_ids)
    print(output.outputs[0].token_ids)
    print(f"Num tokens: {token_count}")
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

responses = [output.outputs[0].text for output in outputs]

for response in responses:
    print(response)


tokens = sum([len(tokenizer.encode(text, add_special_tokens=False)) for text in responses])

print(f"Num tokens in dataset: {tokens}")