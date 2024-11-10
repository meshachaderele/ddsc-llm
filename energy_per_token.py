

tokens = sum([len(tokenizer.encode(text, add_special_tokens=False)) for text in responses])
