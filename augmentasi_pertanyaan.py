import pandas as pd
import random

df = pd.read_csv('dataset.csv')

sinonim = {
    'bagaimana': ['gimana', 'bagaimanakah', 'cara'],
    'cara': ['langkah', 'proses', 'tata cara'],
    'membuat': ['bikin', 'mengurus', 'mengajukan'],
    'ktp': ['kartu tanda penduduk', 'ktp elektronik', 'ektp'],
    'kk': ['kartu keluarga'],
    'baru': ['yang baru', 'terbaru'],
    'hilang': ['kehilangan', 'yang hilang'],
    'mengurus': ['buat', 'urus', 'ajukan'],
    'syarat': ['persyaratan', 'dokumen yang dibutuhkan'],
}

sapaan = [
    '',
    'halo, ',
    'hai, ',
    'permisi, ',
    'maaf, ',
    'selamat pagi, ',
    'selamat siang, ',
    'selamat sore, ',
]

def augment_pertanyaan(pertanyaan):
    variasi = set()
    # Tambahkan sapaan
    for s in sapaan:
        variasi.add(s + pertanyaan)
    # Ganti kata dengan sinonim
    kata = pertanyaan.split()
    for i, k in enumerate(kata):
        if k in sinonim:
            for alt in sinonim[k]:
                kata_baru = kata.copy()
                kata_baru[i] = alt
                variasi.add(' '.join(kata_baru))
    return list(variasi)

data_aug = []
for idx, row in df.iterrows():
    pertanyaan_asli = row['pertanyaan']
    variasi = augment_pertanyaan(pertanyaan_asli)
    for q in variasi:
        data_aug.append({
            'pertanyaan': q,
            'jawaban': row['jawaban'],
            'referensi': row['referensi'] if 'referensi' in row else ''
        })

# Simpan ke file baru
aug_df = pd.DataFrame(data_aug)
aug_df.to_csv('dataset_augmented.csv', index=False)

print('Augmentasi selesai. Hasil disimpan di dataset_augmented.csv')
