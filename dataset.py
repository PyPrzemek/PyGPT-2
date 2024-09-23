# dataset.py
from datasets import load_dataset

def load_dataset(tokenizer):
    # Ładowanie przykładowego zbioru danych (można to dostosować)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Tokenizacja danych
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
