import streamlit as st
import sys
import os
import time
import torch
import pickle
import re
import nltk
from transformers import BartTokenizer, BartForConditionalGeneration

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text, Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Page Config 
st.set_page_config(
    page_title = "AI News Summarizer",
    page_icon  = "📰",
    layout     = "wide"
)

# Load Models 
@st.cache_resource
def load_bart():
    tokenizer = BartTokenizer.from_pretrained('models/bart_model')
    model     = BartForConditionalGeneration.from_pretrained('models/bart_model')
    model     = model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_lstm():
    from src.lstm_model import Encoder, Decoder, Seq2Seq

    with open('models/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    VOCAB_SIZE  = len(vocab)
    EMBED_DIM   = 256
    HIDDEN_DIM  = 512
    NUM_LAYERS  = 2
    DROPOUT     = 0.3

    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    model   = Seq2Seq(encoder, decoder, vocab).to(device)
    model.load_state_dict(torch.load('models/lstm_weights.pt', map_location=device))
    model.eval()
    return model, vocab

def generate_bart(article, tokenizer, model, max_length=128):
    inputs = tokenizer(
        article,
        max_length  = 512,
        truncation  = True,
        return_tensors = 'pt'
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

def generate_lstm(article, model, vocab, max_length=80):
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'

    encoded = vocab.encode(clean_text(article), 400)
    src     = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src)
        input        = torch.tensor([vocab.word2idx[SOS_TOKEN]]).to(device)
        generated    = []

        for _ in range(max_length):
            output, hidden, cell = model.decoder(input, hidden, cell)
            top_token = output.argmax(1)
            word      = vocab.idx2word.get(top_token.item(), UNK_TOKEN)
            if word == EOS_TOKEN:
                break
            generated.append(word)
            input = top_token

    return ' '.join(generated)


#  Header 
st.title("📰 AI News Summarizer")
st.caption("Compare RNN (LSTM) vs Transformer (BART) summarization side by side")
st.divider()

#  Sidebar 
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.radio(
        "Select Model",
        ["BART (Transformer)", "LSTM (RNN)", "⚡ Compare Both"],
        index = 0
    )

    max_length = st.slider("Summary Max Length", 50, 200, 128, step=10)

    st.divider()
    st.header("📊 Model Results")
    st.markdown("""
    | Model | ROUGE-1 | ROUGE-2 |
    |-------|---------|---------|
    | LSTM  | 0.0228  | 0.0012  |
    | BART  | 0.3599  | 0.1446  |
    """)

#  Load selected models 
if model_choice in ["BART (Transformer)", "⚡ Compare Both"]:
    with st.spinner("Loading BART..."):
        bart_tokenizer, bart_model = load_bart()

if model_choice in ["LSTM (RNN)", "⚡ Compare Both"]:
    with st.spinner("Loading LSTM..."):
        lstm_model, vocab = load_lstm()

#  Input 
st.subheader("📝 Paste your news article")

sample_article = """The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, 
the tenth increase in just over a year, as policymakers continued their fight against inflation 
while acknowledging risks to the banking system. The decision was unanimous. Fed Chair Jerome Powell 
said the banking system is sound and resilient but noted that recent developments are likely to result 
in tighter credit conditions for households and businesses. Economists had been divided over whether 
the Fed would pause its rate hiking campaign following recent bank failures."""

article_input = st.text_area(
    label       = "Article",
    value       = sample_article,
    height      = 200,
    placeholder = "Paste a news article here..."
)

col1, col2 = st.columns([1, 5])
with col1:
    summarize_btn = st.button("🚀 Summarize", type="primary", use_container_width=True)

#  Output 
if summarize_btn and article_input.strip():
    cleaned = clean_text(article_input)
    st.divider()

    # ── BART only ──
    if model_choice == "BART (Transformer)":
        with st.spinner("Generating summary..."):
            start   = time.time()
            summary = generate_bart(cleaned, bart_tokenizer, bart_model, max_length)
            elapsed = time.time() - start

        st.subheader("📋 BART Summary")
        st.success(summary)

        col1, col2, col3 = st.columns(3)
        col1.metric("Article Words",    len(article_input.split()))
        col2.metric("Summary Words",    len(summary.split()))
        col3.metric("Inference Time",   f"{elapsed:.2f}s")

    # ── LSTM only ──
    elif model_choice == "LSTM (RNN)":
        with st.spinner("Generating summary..."):
            start   = time.time()
            summary = generate_lstm(cleaned, lstm_model, vocab, max_length)
            elapsed = time.time() - start

        st.subheader("📋 LSTM Summary")
        st.warning(summary)

        col1, col2, col3 = st.columns(3)
        col1.metric("Article Words",  len(article_input.split()))
        col2.metric("Summary Words",  len(summary.split()))
        col3.metric("Inference Time", f"{elapsed:.2f}s")

    # ── Compare Both ──
    elif model_choice == "⚡ Compare Both":
        col_bart, col_lstm = st.columns(2)

        with col_bart:
            st.subheader("🤖 BART (Transformer)")
            with st.spinner("BART generating..."):
                start        = time.time()
                bart_summary = generate_bart(cleaned, bart_tokenizer, bart_model, max_length)
                bart_time    = time.time() - start
            st.success(bart_summary)
            st.metric("Words",          len(bart_summary.split()))
            st.metric("Inference Time", f"{bart_time:.2f}s")
            st.metric("ROUGE-1",        "0.3599")

        with col_lstm:
            st.subheader("🧠 LSTM (RNN)")
            with st.spinner("LSTM generating..."):
                start        = time.time()
                lstm_summary = generate_lstm(cleaned, lstm_model, vocab, max_length)
                lstm_time    = time.time() - start
            st.warning(lstm_summary)
            st.metric("Words",          len(lstm_summary.split()))
            st.metric("Inference Time", f"{lstm_time:.2f}s")
            st.metric("ROUGE-1",        "0.0228")

elif summarize_btn and not article_input.strip():
    st.warning("⚠️ Please paste an article first!")