# GenderBiasGPT

GenderBiasGPT is a tool designed to detect gender bias in text inputs, specifically for non-English languages like Vietnamese and Hindi. This model assesses whether a sentence is biased toward a male or female perspective and provides a bias score ranging from 0 to 1.

[1st Place Winner at Hack SoDA 2024](https://hack-soda-2024.devpost.com/)

[Demo Video Link](https://youtu.be/Y00wXz7UuC8) | [Presentation Link](https://docs.google.com/presentation/d/1gW6mgpVI6ElJAMt6ORozUrit6j601suvUXyPaN7WVYg/edit?usp=sharing) | [Devpost Link](https://devpost.com/software/genderbiasgpt-for-regional-languages)


![maxresdefault](https://github.com/user-attachments/assets/916948f9-54a2-42c3-be17-1c7fd9a35b07)


## Inspiration
- Word embedding libraries in English are often biased; for instance, the vector for "Man" is as close to "Programmer" as the vector for "Woman" is to "Homemaker."
- Our algorithm leverages the way language models reflect Western societal stereotypes to detect gender bias, using our own bias score.
- GenderBiasGPT was inspired by the need to detect gender bias not only in English but in regional languages, particularly Vietnamese and Hindi.
- While models to analyze bias in English text exist, few tools focus on regional languages. GenderBiasGPT addresses this gap by providing a model that identifies gender bias in Vietnamese and Hindi sentences.


## What it Does
GenderBiasGPT evaluates text inputs in Vietnamese or Hindi and calculates their cosine similarity with the gender_direction vector, indicating the level of s bias toward a male or female perspective. This score provides insight into how strongly a sentence may be biased in a particular direction.

## How We Built It
Our model leverages a regression architecture to detect gender bias within the input language. Key components include:
- **Tokenization and Embedding:** We tokenize the input text, generating embeddings that capture contextual meaning.
- **Gender Subspace Definition:** Using a gender subspace we defined, the model evaluates how the tokens in a sentence align with male or female bias, ultimately generating a bias score through cosine similarity.

- Building on insights from two key papers, [Estimating Gender Bias in Sentence Embeddings](https://www.politesi.polimi.it/retrieve/4e623e81-fd73-49f7-bc03-24482b923426/Estimating%20Gender%20Bias%20in%20Sentence%20Embeddings.pdf) and [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/pdf/1607.06520), which were conducted in English, we reverse-engineered the algorithm to apply it to Vietnamese and Hindi.
  
- Our metric, called `bias_score`, is based on the following four elements:
  - **Cosine similarity between two vectors x and y** (`cos(x, y)`): Measures the directional similarity between words.
  - **Gender direction in vector space** (`D`): Captures the primary axis along which gender information is encoded.
  - **List of gendered words** (`L`): A curated set of words associated with gender.
  - **Semantic importance of a word** (`I_w`): Represents each word's significance within the sentence.
  
- The formula for `bias_score` is the sum of the cosine similarity of each word with the gender direction, multiplied by its importance in the sentence.

## Challenges We Faced
- Finding an accurate language model for tokenization and creating embedding for Vietnamese and Hindi is hard, we have to try multiple models to get the best accuracy
- Each language requires a distinct approach to capture the nuances that indicate gender bias.

## Accomplishments We're Proud Of
- Successfully defining a **gender subspace** to analyze bias.
- Achieving an **accuracy of around 70%** in bias detection.
- Developing an **interactive output** that displays the bias score in a user-friendly way.

## What We Learned
We explored the nuances of implementing a gender bias model, focusing on adapting English-language concepts to regional languages. Experimenting with various language models revealed the complexities of detecting bias across multiple languages, highlighting the importance of diversity in developing fair, inclusive technologies.

## What's Next for GenderBiasGPT
- **Scaling to other languages** and adding support for additional regional dialects.
- Expanding detection to other forms of bias, including **social, political, and racial** biases.
- Working to **increase accuracy** and enhance the user experience.

GenderBiasGPT is an exciting first step toward making bias detection tools accessible for a wider range of languages and perspectives. We look forward to expanding and improving our model!

---

## Setup Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dheerajkallakuri/GenderBiasGPT
   cd GenderBiasGPT
   ```

2. **Install Dependencies**:
   Make sure you have `pip` installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Open two terminal windows:

   - **Terminal 1** (Frontend):
     ```bash
     streamlit run gui.py
     ```
   
   - **Terminal 2** (Backend):
     ```bash
     python3 app.py
     ```

4. **Information**:
   The backend is built with Flask, the frontend is built on streamlit while the bias score is generated by a regression model that uses cosine similarity to evaluate gender bias in the text.

Import the language embedder that you want 

```
from from BiasDetectionHelper import VietnameseEmbedder, HindiEmbedder
```
Initialize the embedder

```
vietnamese_embedder = VietnameseEmbedder()
```
Get the gender bias score of your sentence
Code:

```
sentence = "Một đàn ông xòe ra hai cái cánh"
bias_score = vietnamese_embedder.get_gender_bias_score_of_sentence(sentence)
print(bias_score)
```

Output:
```
{'female_bias_score': 0.13118663953125884,
 'male_bias_score': -0.08155740845883201,
 'bias_tokens': {
           'một': {'cosine_similarity': 0.13598315,      'word_importance': 0.20386893917965399},
           'đàn_ông': {'cosine_similarity': -0.4898948,    'word_importance': 0.16647943035338808},
            'x@@': {'cosine_similarity': 0.16068122,   'word_importance': 0.10408949010868583},
            'ò@@': {'cosine_similarity': 0.19100048,   'word_importance': 0.10127540317761463},
            'e': {'cosine_similarity': 0.17239621,   'word_importance': 0.07683502016518115},
            'ra': {'cosine_similarity': 0.1331513,   'word_importance': 0.07592466319248105},
            'hai': {'cosine_similarity': 0.08441967,   'word_importance': 0.10129592758936277},
            'cái': {'cosine_similarity': 0.21521486,   'word_importance': 0.09085246189998558},
            'cánh': {'cosine_similarity': 0.20075066,   'word_importance': 0.07937866433364689}
     }
}
```

## Example Sentences:

**Vietnamese**
- Cô ấy thích mặc váy và đá bóng
- Anh ta không thích nấu ăn nhưng thích may vá
- Vợ thích đấm nhau

**Hindi**
- महिलाएँ घर की सजावट और साफ-सफाई में अधिक रुचि लेती हैं।
- पुरुषों को समाज में सम्मान और समानता का अधिकार मिलना चाहिए।
- पानी का रंग पारदर्शी होता है।


