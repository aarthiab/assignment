from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

app = FastAPI()

# Dummy data for RAG
articles = [
    {"title": "Dealing with Anxiety", "content": "Anxiety is a feeling of worry, nervousness, or unease about something..."},
    {"title": "Coping with Depression", "content": "Depression is more than just feeling sad. It's a serious mental health condition..."},
    # Add more articles as needed
]

class RAGRequest(BaseModel):
    prompt: str

class ClassificationRequest(BaseModel):
    text: str

# Load an open-source LLM for text generation (e.g., GPT-2)
generator = pipeline('text-generation', model='gpt2')

# Dummy dataset for classification
data = pd.DataFrame({
    "text": ["I feel happy today", "I am very sad", "I am anxious about my exams", "I feel so depressed"],
    "category": ["happy", "sad", "anxious", "depressed"]
})

# TF-IDF Vectorizer for text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])

# Nearest Neighbors for RAG
nn_model = NearestNeighbors(n_neighbors=1, metric='cosine').fit(X)

# Load a pre-trained classification model (e.g., BERT for sequence classification)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Mapping from label indices to categories
label_map = {0: "happy", 1: "sad", 2: "anxious", 3: "depressed"}

@app.post("/rag")
async def rag_endpoint(request: RAGRequest):
    prompt = request.prompt

    # Generate response using the LLM
    generated = generator(prompt, max_length=50, num_return_sequences=1)
    generated_text = generated[0]['generated_text']

    # Find the most relevant article using Nearest Neighbors
    query_vec = vectorizer.transform([prompt])
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=1)
    article = articles[indices[0][0]]

    return {"generated_text": generated_text, "relevant_article": article}

@app.post("/classification")
async def classification_endpoint(request: ClassificationRequest):
    text = request.text

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    category = label_map[predicted_class]

    return {"text": text, "category": category}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)