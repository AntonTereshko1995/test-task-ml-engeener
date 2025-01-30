from types import SimpleNamespace

def tokenize_function(texts, tokenizer, params: SimpleNamespace):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=params.model.baseline.max_length)