import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
model = SentenceTransformer('l3cube-pune/hindi-sentence-bert-nli')
sample_hindi_sentence = "नमस्ते, आप कैसे हैं?"
hindi_gender_direction = [
    ["महिला","आदमी"],
    ["बच्ची","बच्चा"],
    ["बेटी","बेटा"],
    ["माँ","पिता"],
    ["लड़की","लड़का"],
    ["स्त्री","मर्द"],
    ["उसकी","उसका"],
    ["स्त्रीलिंग","पुल्लिंग"],
    ["औरत","आदमी"],
    ["विद्या", "राम"]
]

def hindi_sentence_embedding(hindi_sentence):
    tokens = word_tokenize(hindi_sentence)
    # print("Tokens:", tokens)
    embeddings = model.encode(tokens)
    # print("Embeddings:")
    # for token, embedding in zip(tokens, embeddings):
    #     print(f"Token: {token}, Embedding: {embedding}")
    return embeddings

def hindi_words_embedding(hindi_word_pairs):
    embeddings = []
    for pair in hindi_word_pairs:
        # Tokenize both words in the pair
        pair_embedding=[]
        t1=model.encode(word_tokenize(pair[0]))
        t2=model.encode(word_tokenize(pair[1]))
        pair_embedding.append(t1)
        pair_embedding.append(t2)
        embeddings.append(pair_embedding)
    return embeddings


# input_hindi_sentence_embedding = hindi_sentence_embedding(sample_hindi_sentence)
# hindi_gender_embeddings = hindi_words_embedding(hindi_gender_direction)
# for i, (pair, embedding) in enumerate(zip(hindi_gender_direction, hindi_gender_embeddings)):
#     print(f"Pair: {pair}, Embedding: {embedding}")