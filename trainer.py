# trainer.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from dataset import load_dataset
from utils import setup_logging

def train_model():
    setup_logging()  # Konfiguracja logowania
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Ładowanie zbioru danych
    train_dataset = load_dataset(tokenizer)

    # Ustawienia treningu
    training_args = TrainingArguments(
        output_dir='./results',         # Katalog na wyniki
        num_train_epochs=3,             # Liczba epok
        per_device_train_batch_size=2,  # Rozmiar partii
        save_steps=10_000,              # Co ile kroków zapisywać model
        save_total_limit=2,             # Ograniczenie liczby zapisanych modeli
    )

    # Inicjalizacja trenera
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Trening modelu
    trainer.train()

if __name__ == "__main__":
    train_model()
