# Data Folder Notes

Folder `data/` tidak diunggah penuh ke GitHub karena berisi LMDB besar (beberapa file >100 MB).

Yang disertakan di repository:
- `data/valid/gt.txt` sebagai contoh format anotasi.

Contoh struktur lokal:

```text
data/
  train_lmdb/
    data.mdb
    lock.mdb
  valid_lmdb/
    data.mdb
    lock.mdb
  valid2_lmdb/
    data.mdb
    lock.mdb
```

Jika ingin melatih ulang model, siapkan dataset lokal Anda pada folder ini.
