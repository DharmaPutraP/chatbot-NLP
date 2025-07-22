from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import random

# --- Load model dan pipeline ---

# Load tokenizers
with open('tokenizer/input_tokenizer.pkl', 'rb') as f:
    input_tokenizer = pickle.load(f)
with open('tokenizer/target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

# Load models
encoder_model = load_model('model_full/chatbot_encoder.h5')
decoder_model = load_model('model_full/chatbot_decoder.h5')

# Load max_input_len and max_target_len from training
df = pd.read_csv('dataset_augmented.csv')
df.drop_duplicates(inplace=True)
df = df[['pertanyaan', 'jawaban']].dropna()
input_tensor = input_tokenizer.texts_to_sequences(df['pertanyaan'])
target_tensor = target_tokenizer.texts_to_sequences(df['jawaban'])
max_input_len = max(len(t) for t in input_tensor)
max_target_len = max(len(t) for t in target_tensor)

reverse_target_word_index = target_tokenizer.index_word
target_word_index = target_tokenizer.word_index

# Preprocessing (harus sama persis dengan training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9' ]", '', text)
    return text

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

def chatbot_response(input_text):
    input_seq = input_tokenizer.texts_to_sequences([clean_text(input_text)])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    return decode_sequence(input_seq)

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = chatbot_response(user_input)
    return jsonify({'response': response})

@app.route('/suggestions', methods=['GET'])
def suggestions():
    df = pd.read_csv('dataset_augmented.csv')
    pertanyaan_list = df['pertanyaan'].dropna().unique().tolist()
    saran = random.sample(pertanyaan_list, min(3, len(pertanyaan_list)))
    return jsonify({'suggestions': saran})

if __name__ == '__main__':
    app.run(debug=True)
