import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pickle
import os
import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Paths
BART_MODEL_PATH = 'models/bart_model'
LSTM_VOCAB_PATH = 'models/vocab.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource  
def load_bart():
    """Load BART model and tokenizer — cached so it only loads once"""
    print("🔄 Loading BART model...")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_PATH)
    model     = BartForConditionalGeneration.from_pretrained(BART_MODEL_PATH)
    model     = model.to(device)
    model.eval()
    print(" BART model loaded")
    return tokenizer, model


def generate_summary(article, tokenizer, model, max_length=128):
    """Generate summary from article text"""
    inputs = tokenizer(
        article,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams            = 4,
            max_length           = max_length,
            min_length           = 20,
            length_penalty       = 2.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)