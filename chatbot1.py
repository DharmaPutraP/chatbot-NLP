import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, Add, Attention, Concatenate # Import Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
import os

# --- DATA & PREPROCESSING ---
data = pd.read_csv('dataset.csv')

import string
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

def preprocess(text):
    text = text.lower()
    # Hapus hanya tanda baca yang tidak penting, biarkan koma dan titik
    allowed_punct = ',.'
    text = ''.join([c for c in text if c.isalnum() or c.isspace() or c in allowed_punct])
    return text

# Preprocessing pertanyaan dan jawaban
questions = data['pertanyaan'].apply(preprocess)
answers = data['jawaban'].apply(preprocess)
# Tambahkan token <start> dan <end> sebelum membuat list dan fit tokenizer
answers = ["<START> "+ ans + " <END>" for ans in answers]

tokenizer = Tokenizer(filters='') # Keep all characters, including \t and \n for target
tokenizer.fit_on_texts (questions + answers)

word_to_idx= tokenizer.word_index
num_encoder_tokens = len(word_to_idx) + 1
num_decoder_tokens = len(word_to_idx) + 1 # Use the same vocabulary for simplicity, we can also use different

#see the 10 first word in word dict, just to make sure
print(list(word_to_idx.items())[:10])

#Change to Sequence and Padding
#for encoder
encoder_input_data = tokenizer.texts_to_sequences (questions)
max_len_encoder = max([len(seq) for seq in encoder_input_data])
encoder_input_data = pad_sequences (encoder_input_data, maxlen=max_len_encoder, padding='post')

#for decoder
decoder_input_data = tokenizer.texts_to_sequences (answers)
max_len_decoder = max([len(seq) for seq in decoder_input_data])
decoder_input_data = pad_sequences (decoder_input_data, maxlen=max_len_decoder, padding='post')

encoder_input_data.shape

#preparing target for decoder, target is for decoder to predict the word to generate

decoder_target_data = np.zeros(
  (len(answers), max_len_decoder, num_decoder_tokens),
  dtype='float32'
)

for i, target_seq in enumerate (decoder_input_data):
  for t, word_idx in enumerate (target_seq):
    if t > 0:
      decoder_target_data[i, t-1, word_idx] = 1.0

decoder_target_data.shape

latent_dim = 128 # You can experiment with smaller values if needed, e.g., 64

# Encoder (ubah return_sequences=True)
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
enc_dropout = Dropout(0.3)(enc_emb)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_dropout)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
dec_dropout = Dropout(0.3)(dec_emb) # Added Dropout
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_dropout, initial_state=encoder_states)

# Attention layer
attn_layer = Attention()
attn_out = attn_layer([decoder_outputs, encoder_outputs])
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#Step 6: Compile and Train Model
model.compile(optimizer=Adam (learning_rate=0.001), loss='categorical_crossentropy')
batch_size = 32

epochs = 400 # Coba lebih banyak epoch jika data sudah besar
#fit the model

MODEL_PATH = 'seq2seq_chatbot_model.h5'

# Training dan/atau load model
if os.path.exists(MODEL_PATH):
    print('Memuat model dari file:', MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)
else:
    print('Training model baru...')
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size, epochs=epochs)
    model.save(MODEL_PATH)
    print('Model disimpan ke', MODEL_PATH)

#visualize the loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Encoder model untuk inference
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder model untuk inference
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_encoder, latent_dim))

dec_emb2 = dec_emb_layer(decoder_inputs)
dec_dropout2 = Dropout(0.3)(dec_emb2)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_dropout2, initial_state=[decoder_state_input_h, decoder_state_input_c])

attn_out_inf = attn_layer([decoder_outputs2, decoder_hidden_state_input])
decoder_concat_input_inf = Concatenate(axis=-1)([decoder_outputs2, attn_out_inf])
decoder_outputs2 = decoder_dense(decoder_concat_input_inf)

decoder_model = Model(
    [decoder_inputs, decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2, state_h2, state_c2]
)

#Reverse Tokenization
idx_to_word = {v: k for k, v in word_to_idx.items()}
# Tambahkan secara eksplisit indeks 0 untuk token padding
#Ini adalah hack sementara jika Anda ingin mengizinkan e diprediksi,
#tapi lebih baik mencegah diprediksi atau diabaikan

idx_to_word[0] = '<PAD>' # Atau string lain yang Anda inginkan untuk padding

# Fungsi decode_sequence dengan attention
def decode_sequence(input_seq):
    # Dapatkan encoder_outputs dan state awal
    encoder_outs, state_h, state_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_to_idx['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, encoder_outs, state_h, state_c]
        )
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx_to_word.get(sampled_token_index, '')
        if (sampled_word == '<end>' or sampled_word == '<PAD>' or
            len(decoded_sentence.split()) >= max_len_decoder):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        state_h, state_c = h, c
    # Post-processing: kapitalisasi dan tambahkan titik jika perlu
    result = decoded_sentence.strip()
    if result:
        result = result[0].upper() + result[1:]
        if result[-1] not in '.!?':
            result += '.'
    return result

#Chatting Function

def chat_with_bot(input_text):
    clean_input_text = preprocess(input_text)
    input_seq = tokenizer.texts_to_sequences([clean_input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len_encoder, padding='post')
    if not input_seq.any():
        print("Bot: Sorry, I don't understand. Could you please repeat that")
        return
    response = decode_sequence(input_seq)
    if not response.strip():
        print("Bot: Sorry, I can't provide a response at the moment. Please try a different question.")
    else:
        print("Bot:", response)


#Try Chatbot
print("Mulai chatting dengan bot. Ketik 'bye' untuk keluar.")
while True:
  user_input = input("Anda: ")
  if user_input.lower() == 'bye':
    print("Bot: Sampai jumpa!")
    break
  chat_with_bot(user_input)