import numpy as np
import tensorflow as tf
import pickle
from django.shortcuts import render
from .forms import ReviewForm
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка модели и токенизатора
model = tf.keras.models.load_model('model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(review):
    # Предобработка текста
    sequences = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequences, maxlen=100)  # Убедитесь, что maxlen соответствует вашему обучению
    prediction = model.predict(padded)
    sentiment = 'Положительный' if prediction[0] > 0.5 else 'Отрицательный'
    rating = int(prediction[0] * 9 + 1) if sentiment == 'Положительный' else int(prediction[0] * 4 + 1)
    return sentiment, rating

def review_view(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review_text']
            sentiment, rating = predict_sentiment(review_text)
            return render(request, 'main/result.html', {'sentiment': sentiment, 'rating': rating, 'review': review_text})
    else:
        form = ReviewForm()
    return render(request, 'main/review.html', {'form': form})
   