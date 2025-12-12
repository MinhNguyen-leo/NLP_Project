# NLP_Project
Neural Machine Translation (EN → FR)
🚀 Seq2Seq • Bahdanau Attention • Beam Search • BPE Tokenization
<p align="center"> <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" /> <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" /> <img src="https://img.shields.io/badge/License-MIT-green" /> <img src="https://img.shields.io/badge/Model-Baseline%20%7C%20Attention-orange" /> </p> <p align="center"> <img src="./images/LSTM.png.png" width="750"> </p>


# 📑 Table of Contents
- [📘 1. Giới thiệu](#-1-giới-thiệu)
- [📂 2. Cấu trúc dự án](#-2-cấu-trúc-dự-án)
- [📊 3. Dữ liệu và xử lý dữ liệu](#-3-dữ-liệu-và-xử-lý-dữ-liệu)
- [🧠 4. Kiến trúc mô hình](#-4-kiến-trúc-mô-hình)
- [🏋️ 5. Huấn luyện mô hình](#-5-huấn-luyện-mô-hình)
- [📈 6. Kết quả](#-6-kết-quả)
- [💬 7. Ví dụ dịch](#-7-ví-dụ-dịch)
- [📦 8. Tải mô hình ](#-8-tải-mô-hình-pretrained)
- [📝 9. Kết luận](#-10-kết-luận)

---

# 📘 1. Giới thiệu

Dự án xây dựng hệ thống **dịch tự động Neural Machine Translation (NMT)** từ **Tiếng Anh → Tiếng Pháp** gồm:

- **Mô hình gốc (Baseline):** Seq2Seq LSTM  
- **Mô hình mở rộng:** Seq2Seq + **Bahdanau Attention**  
- **Beam Search** mức 5 để tăng chất lượng sinh câu  
- **BPE Tokenization** để hạn chế OOV và tối ưu biểu diễn từ

Mục tiêu:  
✔ So sánh chất lượng dịch 2 mô hình  
✔ Trình bày pipeline xử lý dữ liệu – huấn luyện – đánh giá  
✔ Báo cáo BLEU score và phân tích lỗi  

---

# 📂 2. Cấu trúc dự án
```bash

📦 NLP_Project
│── data/
│ ├── raw/                         # chứa dữ liệu raw
│ ├── proceesed/                   # chứa dữ liệu đã qua xử 
│── best_model.pth                 # best model baseline
│── best_attn.pth                  # best model attention
│── processed_data.py              # Script tiền xử lý dữ liệu 
│── processed_data_bpe.py          # Script tiền xử lý BPE
│── main.ipynb
└── README.md

```

---

# 📊 3. Dữ liệu và xử lý dữ liệu

### ✔ Dataset
Sử dụng tập dữ liệu **Tatoeba / ManyThings EN–FR**, gồm:
- 29k câu train  
- 1k câu validation  
- 1k câu test  

### ✔ Baseline preprocessing (spaCy)
- Tách từ theo word-level  
- Token `<unk>` xuất hiện nhiều  
- Dễ gây mất thông tin ở từ hiếm

### ✔ Extended preprocessing (BPE – SentencePiece)
- Vocab 4000 subwords  
- Giảm từ chưa thấy (OOV)  
- Cải thiện phân rã từ → mô hình dễ học hơn  

---

# 🧠 4. Kiến trúc mô hình

## **4.1. Baseline – Seq2Seq LSTM**
<p align="center">
  <img src="images/baseline_arch.png" width="650">
</p>

- Encoder: 2-layer LSTM  
- Decoder: 2-layer LSTM  
- Không có attention  
- Thông tin câu dài bị “quên” → dịch kém ở câu dài  

---

## **4.2. Mô hình mở rộng – Bahdanau Attention**
<p align="center">
  <img src="images/attention_arch.png" width="650">
</p>

Cải thiện:
- Giữ được ngữ cảnh tốt hơn  
- Giảm lỗi lặp từ  
- Tập trung vào token quan trọng trong từng bước sinh  

---

## **4.3. Beam Search (size = 5)**
Giữ nhiều giả thuyết câu cùng lúc

Tránh greedy decoding (thường bị quá tham địa phương)

length_penalty = 0.7 để giảm bias câu ngắn

---

# 🏋️ 5. Huấn luyện mô hình

## **Baseline**
| Thành phần | Cấu hình |
|-----------|----------|
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropy (ignore pad_id) |
| Epoch | 20 |
| Teacher Forcing | 0.5 |
| Batch size | 64 |
| Scheduler | ReduceLROnPlateau |
| Early stopping | patience = 3 |

---

## **Attention Model**
| Thành phần | Cấu hình |
|-----------|----------|
| Hidden size | **512** |
| Embedding | **320** |
| Dropout | 0.3 |
| Teacher forcing | 0.7 → 0.1 |
| Epoch | 47 (early-stopped) |
| Optimizer | Adam (lr=3e-4) |
| Scheduler | ReduceLROnPlateau |
| Beam size | 5 |

---

# 📈 6. Kết quả

## **BLEU Score**
| Mô hình | BLEU |
|--------|-------|
| **Seq2Seq Baseline** | **0.3832** |
| **Seq2Seq + Attention** | **0.4432** |

👉 Attention **tăng 23%** BLEU so với baseline.  
👉 Giảm rõ rệt lỗi lặp từ, mất thông tin, dịch sai ngữ nghĩa.

---

# 💬 7. Ví dụ dịch

### **Baseline**
| EN | REF | PRED |
|----|-----|-------|
| A man in an orange hat… | un homme… | un homme avec un orange orange… |
| A Boston Terrier… | un terrier… | un gardien de hockey… |

---

### **Attention Model**
| EN | REF | PRED |
|----|-----|-------|
| a man in an orange hat… | un homme… | un homme avec un casquette orange… |
| a boston terrier… | un terrier… | un joueur de bk court… |

---

# 📦 8. Tải mô hình 

➡ **GitHub Releases:**  
https://github.com/MinhNguyen-leo/NLP_Project/releases/tag/nlp

### Tải 2 attention model:

```python

import requests

def download_from_github(url, output_path):
    print(f"Downloading from {url} ...")
    r = requests.get(url)
    if r.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(r.content)
        print(f"Saved: {output_path}")
    else:
        print("Failed to download:", r.status_code)

# Load last model và best model từ Github Releases
baseline_url = "https://github.com/MinhNguyen-leo/NLP_Project/releases/download/nlp/last_attn.pth"
attn_url     = "https://github.com/MinhNguyen-leo/NLP_Project/releases/download/nlp/best_attn.pth"

download_from_github(baseline_url, "last_attn.pth")
download_from_github(attn_url,     "best_attn.pth")
```

# 📝 9. Kết luận

* Baseline Seq2Seq hạn chế: dễ lặp từ, mất ngữ cảnh, dịch sai danh từ riêng.

* Attention cải thiện mạnh: giữ được thông tin toàn câu, tập trung vào token quan trọng.

* Beam Search giúp câu dịch mượt & tự nhiên hơn.

* BLEU tăng từ 0.38 → 0.47.


