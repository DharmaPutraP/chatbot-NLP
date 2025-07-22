import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import unicodedata
from nltk.corpus import stopwords
import string
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import os
import requests
from tqdm import tqdm

nltk.download('stopwords')

# Baca data
df = pd.read_csv('../dataset.csv')
df.head()

df.info()
print('---')
print('Missing values:')
print(df.isnull().sum())

print('Jumlah data sebelum drop duplikasi:', len(df))
df = df.drop_duplicates(subset=['pertanyaan', 'jawaban'])
print('Jumlah data setelah drop duplikasi:', len(df))

df['len_pertanyaan'] = df['pertanyaan'].astype(str).apply(lambda x: len(x.split()))
df['len_jawaban'] = df['jawaban'].astype(str).apply(lambda x: len(x.split()))
plt.figure(figsize=(12,5))
sns.histplot(df['len_pertanyaan'], bins=20, kde=True, color='blue', label='Pertanyaan')
sns.histplot(df['len_jawaban'], bins=20, kde=True, color='orange', label='Jawaban')
plt.legend()
plt.title('Distribusi Panjang Teks (dalam kata)')
plt.xlabel('Jumlah Kata')
plt.show()

stop_words = set(stopwords.words('indonesian'))

def clean_text_pro(text):
    # Lowercase
    text = str(text).lower()
    # Normalisasi unicode
    text = unicodedata.normalize('NFKC', text)
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    # Hapus stopwords
    tokens = text.strip().split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['pertanyaan_clean'] = df['pertanyaan'].apply(clean_text_pro)
df['jawaban_clean'] = df['jawaban'].apply(clean_text_pro)
df[['pertanyaan', 'pertanyaan_clean', 'jawaban', 'jawaban_clean']].head()

# Tokenisasi
df['pertanyaan_tokens'] = df['pertanyaan_clean'].apply(word_tokenize)
df['jawaban_tokens'] = df['jawaban_clean'].apply(word_tokenize)

# Analisis kata unik
all_q_tokens = [token for tokens in df['pertanyaan_tokens'] for token in tokens]
all_a_tokens = [token for tokens in df['jawaban_tokens'] for token in tokens]

print('Jumlah kata unik pertanyaan:', len(set(all_q_tokens)))
print('Jumlah kata unik jawaban:', len(set(all_a_tokens)))
print('10 kata paling sering di pertanyaan:', Counter(all_q_tokens).most_common(10))
print('10 kata paling sering di jawaban:', Counter(all_a_tokens).most_common(10))

# Visualisasi distribusi panjang token
plt.figure(figsize=(12,5))
sns.histplot([len(tokens) for tokens in df['pertanyaan_tokens']], bins=20, kde=True, color='blue', label='Pertanyaan')
sns.histplot([len(tokens) for tokens in df['jawaban_tokens']], bins=20, kde=True, color='orange', label='Jawaban')
plt.legend()
plt.title('Distribusi Panjang Token Setelah Preprocessing')
plt.xlabel('Jumlah Token')
plt.show()

# Tambahkan token <start> dan <end> pada jawaban

df['jawaban_seq2seq'] = df['jawaban_clean'].apply(lambda x: '<start> ' + x + ' <end>')

# Tokenizer Keras untuk pertanyaan dan jawaban
num_words = 5000
embedding_dim = 300
latent_dim = 256
q_tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
a_tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
q_tokenizer.fit_on_texts(df['pertanyaan_clean'])
a_tokenizer.fit_on_texts(df['jawaban_seq2seq'])

# Konversi ke sequence
q_sequences = q_tokenizer.texts_to_sequences(df['pertanyaan_clean'])
a_sequences = a_tokenizer.texts_to_sequences(df['jawaban_seq2seq'])

# Padding
max_q_len = max([len(seq) for seq in q_sequences])
max_a_len = max([len(seq) for seq in a_sequences])
q_padded = pad_sequences(q_sequences, maxlen=max_q_len, padding='post')
a_padded = pad_sequences(a_sequences, maxlen=max_a_len, padding='post')

# Split data
X_train, X_val, y_train, y_val = train_test_split(q_padded, a_padded, test_size=0.2, random_state=42)

# Download dan load FastText setelah padding

fasttext_path = 'cc.id.300.vec'
if not os.path.exists(fasttext_path):
    print('Mengunduh FastText Bahasa Indonesia...')
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz'
    r = requests.get(url, stream=True)
    with open('cc.id.300.vec.gz', 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)
    import gzip
    with gzip.open('cc.id.300.vec.gz', 'rb') as f_in:
        with open(fasttext_path, 'wb') as f_out:
            f_out.write(f_in.read())

def load_fasttext_matrix(tokenizer, path, num_words, embedding_dim):
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f, desc='Load FastText'):
            values = line.rstrip().split(' ')
            if len(values) < embedding_dim + 1:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

q_embedding_matrix = load_fasttext_matrix(q_tokenizer, fasttext_path, num_words, embedding_dim)
a_embedding_matrix = load_fasttext_matrix(a_tokenizer, fasttext_path, num_words, embedding_dim)

# Encoder (stacked LSTM + dropout)
encoder_inputs = Input(shape=(max_q_len,))
enc_emb = Embedding(input_dim=num_words, output_dim=embedding_dim, mask_zero=True,
                   weights=[q_embedding_matrix], trainable=False)(encoder_inputs)
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3)
encoder_lstm2 = LSTM(latent_dim, return_state=True, dropout=0.3)
encoder_outputs1, _, _ = encoder_lstm1(enc_emb)
_, state_h, state_c = encoder_lstm2(encoder_outputs1)
encoder_states = [state_h, state_c]

# Decoder (stacked LSTM + dropout)
decoder_inputs = Input(shape=(max_a_len,))
dec_emb_layer = Embedding(input_dim=num_words, output_dim=embedding_dim, mask_zero=True,
                         weights=[a_embedding_matrix], trainable=False)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3)
decoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3)
decoder_outputs1, _, _ = decoder_lstm1(dec_emb, initial_state=encoder_states)
decoder_outputs2, _, _ = decoder_lstm2(decoder_outputs1)
decoder_dense = Dense(num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs2)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Siapkan target output (shifted)
y_train_out = np.expand_dims(np.concatenate([y_train[:,1:], np.zeros((y_train.shape[0],1))], axis=1), -1)
y_val_out = np.expand_dims(np.concatenate([y_val[:,1:], np.zeros((y_val.shape[0],1))], axis=1), -1)

# Training
history = model.fit([X_train, y_train], y_train_out, validation_data=([X_val, y_val], y_val_out), batch_size=32, epochs=60)

# Simpan model dan tokenizer
model.save('chatbot_seq2seq.h5')
with open('q_tokenizer.pkl', 'wb') as f:
    pickle.dump(q_tokenizer, f)
with open('a_tokenizer.pkl', 'wb') as f:
    pickle.dump(a_tokenizer, f)

# Visualisasi hasil training
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Fungsi inference chatbot

def decode_sequence(input_seq, encoder_model, decoder_model, max_a_len, a_tokenizer):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, max_a_len))
    # Mulai dengan token <start>
    start_token = a_tokenizer.word_index.get('<start>', 1)
    target_seq[0, 0] = start_token
    decoded_sentence = []
    for i in range(1, max_a_len):
        output_tokens, h, c = decoder_model.predict([target_seq, states_value])
        sampled_token_index = np.argmax(output_tokens[0, i-1, :])
        sampled_word = a_tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word == '<end>' or sampled_word == '' or sampled_word == '<OOV>':
            break
        decoded_sentence.append(sampled_word)
        target_seq[0, i] = sampled_token_index
        states_value = [h, c]
    return ' '.join(decoded_sentence)

# Siapkan model encoder dan decoder untuk inference
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs1_inf, _, _ = decoder_lstm1(dec_emb2, initial_state=decoder_states_inputs)
decoder_outputs2_inf, state_h2, state_c2 = decoder_lstm2(decoder_outputs1_inf)
decoder_outputs2_inf = decoder_dense(decoder_outputs2_inf)
decoder_states2 = [state_h2, state_c2]
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2_inf] + decoder_states2)

# Contoh inference
pertanyaan_test = [
    'bagaimana cara membuat kk baru?',
    'apa syarat pembuatan akta kelahiran?',
    'berapa lama proses pencatatan kematian?',
    'dokumen apa saja untuk pengajuan ktp?',
]
for q in pertanyaan_test:
    q_clean = clean_text_pro(q)
    q_seq = q_tokenizer.texts_to_sequences([q_clean])
    q_pad = pad_sequences(q_seq, maxlen=max_q_len, padding='post')
    jawaban_pred = decode_sequence(q_pad, encoder_model, decoder_model, max_a_len, a_tokenizer)
    print(f'Pertanyaan: {q}')
    print(f'Jawaban chatbot: {jawaban_pred}')
    print('-'*40)

# Analisis distribusi panjang jawaban (dalam kata)
df['jawaban_len'] = df['jawaban_clean'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,4))
sns.histplot(df['jawaban_len'], bins=20, kde=True, color='orange')
plt.title('Distribusi Panjang Jawaban (dalam kata)')
plt.xlabel('Jumlah Kata per Jawaban')
plt.ylabel('Jumlah Data')
plt.show()

print('Statistik panjang jawaban (dalam kata):')
print(df['jawaban_len'].describe())
print('5 jawaban terpendek:')
print(df.loc[df['jawaban_len'].idxmin(), 'jawaban_clean'])
print('5 jawaban terpanjang:')
print(df.loc[df['jawaban_len'].idxmax(), 'jawaban_clean'])