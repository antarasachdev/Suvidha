# Suvidha

# 📊 Multi-Document Summarization Benchmarking Report

## 📁 Datasets Used

### 1. **CNN/DailyMail**
- News articles with human-written highlights.
- Used for single and multi-document summarization.
- Source: `cnn_dailymail`, version 3.0.0

### 2. **XSum**
- Extreme summarization of BBC articles into one-sentence summaries.
- Source: `knkarthick/xsum`

### 3. **Newsroom**
- News articles with a variety of summary styles (extractive, abstractive, mixed).
- Source: Manual download of JSONL files

---

## 🧠 Models Benchmarked

| Model Type | Description |
|------------|-------------|
| PEGASUS | Pre-trained on massive summarization corpora with gap-sentence generation. |
| BART | Denoising autoencoder for pretraining sequence-to-sequence models. |
| T5 (Base/Large) | Unified text-to-text transformer. |
| LED | Longformer Encoder-Decoder for long documents. |
| Absformer | Sentence clustering + summarization approach. |
| TG-MultiSum | Graph-based attention transformer using topic-guided structure. |
| Hierarchical Transformer | Encodes multi-document hierarchy in segments. |
| DCA | Multi-agent communication system across document segments. |
| Long-T5 | Transformer pre-trained for very long text. |
| PRIMERA | Enhanced pretraining with relational awareness. |
| External Knowledge Models | Use chunked input with semantic enrichment (retrieved Wikipedia facts, etc.). |

---

## 📊 Manual ROUGE Score Comparison Table (ROUGE-Lsum)

| Model                          | CNN/DM | XSum  | Newsroom |
|-------------------------------|--------|-------|----------|
| PEGASUS                       | 0.125  | 0.40  | 0.2164   |
| BART                          | ~0.202 | 0.000 | ~0.120   |
| T5-Base                       | ~0.153 | ~0.175| ~0.089   |
| T5-Large                      | 0.2791 | 0.1754| 0.1111   |
| LED / TG-MultiSum             | 0.1811 | 0.0988| 0.1106   |
| TG-MultiSum (Alt)             | 0.2018 | 0.1152| 0.1301   |
| Hierarchical Transformer      | 0.0272 | 0.0180| 0.0640   |
| DCA (Deep Communicating Agents)| 0.1704 | 0.1279| 0.1243   |
| Ext Knowledge + Chunking      | 0.2146 | 0.1001| 0.1299   |
| Absformer                     | 0.2204 | 0.0786| 0.1654   |
| Long-T5                       | 0.2497 | 0.1058| 0.1344   |
| PRIMERA                       | 0.2134 | 0.1002| 0.2175   |

---

## 📈 Visualization Code (Plot ROUGE-Lsum)

```python
import matplotlib.pyplot as plt

models = [
    "PEGASUS", "BART", "T5-Base", "T5-Large", "TG-MultiSum", "TG-MultiSum (Alt)",
    "Hierarchical Transformer", "DCA", "External Knowledge", "Absformer",
    "Long-T5", "PRIMERA"
]
cnn =     [0.125, 0.202, 0.153, 0.2791, 0.1811, 0.2018, 0.0272, 0.1704, 0.2146, 0.2204, 0.2497, 0.2134]
xsum =    [0.40,  0.000, 0.175, 0.1754, 0.0988, 0.1152, 0.0180, 0.1279, 0.1001, 0.0786, 0.1058, 0.1002]
newsroom =[0.2164,0.120, 0.089, 0.1111, 0.1106, 0.1301, 0.0640, 0.1243, 0.1299, 0.1654, 0.1344, 0.2175]

x = range(len(models))
plt.figure(figsize=(14, 7))
plt.plot(x, cnn, marker='o', label='CNN/DM')
plt.plot(x, xsum, marker='s', label='XSum')
plt.plot(x, newsroom, marker='^', label='Newsroom')
plt.xticks(x, models, rotation=45, ha='right')
plt.title("ROUGE-Lsum Scores Across Models and Datasets")
plt.ylabel("ROUGE-Lsum")
plt.xlabel("Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 🏁 Conclusion

- **Best Model (Overall)**: Long-T5 (CNN/DM), PRIMERA (Newsroom), PEGASUS (XSum).
- **Fastest Inference**: T5-base and LED with chunked input.
- **Most Balanced**: PRIMERA and Absformer perform well across all datasets.
- **Worst Performer**: Hierarchical Transformer without training.

---

## 🛠️ Environment Notes

- Sample size: 5–20 for each model per dataset
- Execution: Google Colab ( GPU)
- NLTK avoided in some modelsused (to avoid lookup errors)
