import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

# Load the tokenizer and model
PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

# Define the sentence and word pairs
sample_vietnamese_sentence = 'Kẻ đánh bom đinh tồi tệ nhất nước Anh.'
vietnamese_gender_direction = [
    ["đàn_bà", "đàn_ông"],
    ["cô_ấy", "anh_ấy"],
    ["con gái", "con trai"],
    ["mẹ", "bố"],
    ["cô_gái", "chàng_trai"],
    ["u", "thầy"],
    ["nữ", "nam"],
    ["con", "thằng"],
    ["nữ_tính", "nam_tính"],
    ["Thúy", "Hùng"]
]

def vietnamese_sentence_embedding(input_sentence):
    inputs = PhobertTokenizer(input_sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
    num_tokens = last_hidden_state.shape[1]
    
    # Gather embeddings for each word except the first token ([CLS])
    word_embeddings = []
    for i in range(1, num_tokens):  # Start from the second token (index 1)
        token_embedding = last_hidden_state[0][i].detach().numpy()
        # Normalize the embedding
        normalized_embedding = token_embedding / np.linalg.norm(token_embedding)
        word_embeddings.append(normalized_embedding)
    
    return word_embeddings

def vietnamese_words_embedding(word_pairs):
    embeddings = []
    for pair in word_pairs:
        pair_embedding = []
        for word in pair:
            inputs = PhobertTokenizer(word, return_tensors="pt")
            outputs = model(**inputs)
            # Take embedding for the main token (excluding [CLS])
            embedding = outputs.last_hidden_state[0][1].detach().numpy()
            normalized_embedding = embedding / np.linalg.norm(embedding)
            pair_embedding.append(normalized_embedding)
        embeddings.append(pair_embedding)
    
    return embeddings

# input_vietnamese_sentence_embedding = vietnamese_sentence_embedding(sample_vietnamese_sentence)
# vietnamese_gender_embeddings = vietnamese_words_embedding(vietnamese_gender_direction)
# for i, (pair, embedding) in enumerate(zip(vietnamese_gender_direction, vietnamese_gender_embeddings)):
#     print(f"Pair: {pair}, Embedding: {embedding}")
