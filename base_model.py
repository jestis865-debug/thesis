import pandas as pd
import numpy as np
import re
import jieba
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, mean_absolute_error, cohen_kappa_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('./best_models', exist_ok=True)
os.makedirs('./confusion_matrices', exist_ok=True)

TARGET_NAMES = ['1星(极差)', '2星(较差)', '3星(中等)', '4星(较好)', '5星(极好)']
all_results = []

# 评估与保存工具函数
def evaluate_and_save(y_true, y_pred, model_name, train_duration):
    acc = np.mean(y_pred == y_true)
    acc_pm1 = np.mean(np.abs(y_pred - y_true) <= 1)
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 画归一化混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix: {model_name}')
    plt.ylabel('True Rating')
    plt.xlabel('Predicted Rating')
    plt.savefig(f'./{model_name}_cm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Accuracy: {acc*100:.2f}% | Acc @ ±1: {acc_pm1*100:.2f}% | QWK: {qwk:.4f}")
    
    return {
        "模型代号": model_name,
        "训练方式": "传统基线/从头训练",
        "训练时长(分钟)": round(train_duration, 2),
        "严格准确率(%)": round(acc * 100, 2),
        "容差准确率(Acc±1 %)": round(acc_pm1 * 100, 2),
        "平均绝对误差(MAE)": round(mae, 4),
        "二次卡帕系数(QWK)": round(qwk, 4),
        "宏平均F1(Macro-F1)": round(macro_f1, 4)
    }

print("========== [阶段一] 数据加载与预处理 ==========")
df = pd.read_csv('output.csv')
df = df.dropna(subset=['Review Text', 'Rating'])
df = df[df['Review Text'].str.strip() != '']
df['label'] = df['Rating'].apply(lambda x: int(float(x)) - 1)
df = df[df['label'].isin([0, 1, 2, 3, 4])].copy()

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&#[0-9]+;', '', text)
    text = re.sub(r'([~!?。，！、])\1+', r'\1', text)
    return text.strip()

df['clean_text'] = df['Review Text'].apply(clean_text)
df = df[df['clean_text'] != '']

try:
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
except FileNotFoundError:
    stopwords = set()

def tokenize_text(text):
    words = jieba.lcut(text)
    words = [w for w in words if w not in stopwords and w.strip() != '']
    return " ".join(words)

df['tokenized_text'] = df['clean_text'].apply(tokenize_text)

X = df['tokenized_text'].values
y = df['label'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp)


# ==============================================================
# 模型一：TF-IDF + Logistic Regression
# ==============================================================
print("\n========== [开始训练] 模型一 (TF-IDF + LR) ==========")
start_time_lr = time.time()

# 优化：扩大一点 max_features，使用 n_jobs=-1 加速训练
tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)
duration_lr = (time.time() - start_time_lr) / 60.0

res_lr = evaluate_and_save(y_test, y_pred_lr, "Baseline_TFIDF_LR", duration_lr)
all_results.append(res_lr)


# ==============================================================
# 模型二：Word2Vec + Bi-LSTM
# ==============================================================
print("\n========== [开始准备] 模型二 (Word2Vec + Bi-LSTM) ==========")

# 1. 训练电商专属 Word2Vec
start_time_lstm = time.time()
sentences = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=300, window=5, min_count=2, workers=4)

# 2. 构建词表与权重矩阵
vocab = {'<PAD>': 0, '<UNK>': 1}
embedding_matrix = [np.zeros(300), np.random.uniform(-0.1, 0.1, 300)]

for word in w2v_model.wv.index_to_key:
    vocab[word] = len(vocab)
    embedding_matrix.append(w2v_model.wv[word])

embedding_matrix = np.array(embedding_matrix)
vocab_size = len(vocab)

# 3. 数据截断填充与 Dataset 构建
MAX_SEQ_LENGTH = 128

def text_to_sequence(text, vocab, max_len):
    words = text.split()
    seq = [vocab.get(w, vocab['<UNK>']) for w in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
    return seq

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = [text_to_sequence(t, vocab, max_len) for t in texts]
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_loader = DataLoader(ReviewDataset(X_train, y_train, vocab, MAX_SEQ_LENGTH), batch_size=256, shuffle=True)
val_loader = DataLoader(ReviewDataset(X_val, y_val, vocab, MAX_SEQ_LENGTH), batch_size=256, shuffle=False)
test_loader = DataLoader(ReviewDataset(X_test, y_test, vocab, MAX_SEQ_LENGTH), batch_size=256, shuffle=False)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMClassifier(vocab_size, 300, 128, 5, embedding_matrix).to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

print("\n========== [开始训练] Bi-LSTM (引入早停机制) ==========")
EPOCHS = 30
PATIENCE = 3
best_val_qwk = -1.0
patience_counter = 0
best_model_path = "./Baseline_BiLSTM.pth"

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch_text, batch_labels in train_loader:
        batch_text, batch_labels = batch_text.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_text)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
        
    # 验证集评估
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for val_text, val_labels in val_loader:
            val_text, val_labels = val_text.to(device), val_labels.to(device)
            val_outputs = model(val_text)
            _, val_predicted = torch.max(val_outputs.data, 1)
            
            val_preds.extend(val_predicted.cpu().numpy())
            val_targets.extend(val_labels.cpu().numpy())
            
    val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {total_loss/len(train_loader):.3f} | Val QWK: {val_qwk:.4f}')
    
    if val_qwk > best_val_qwk:
        best_val_qwk = val_qwk
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

duration_lstm = (time.time() - start_time_lstm) / 60.0

# 加载最佳模型在测试集上评估
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for test_text, test_labels in test_loader:
        test_text = test_text.to(device)
        test_outputs = model(test_text)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_preds.extend(test_predicted.cpu().numpy())
        test_targets.extend(test_labels.numpy())

res_lstm = evaluate_and_save(np.array(test_targets), np.array(test_preds), "Baseline_W2V_BiLSTM", duration_lstm)
all_results.append(res_lstm)


# ==============================================================
# 阶段四：导出结果报表
# ==============================================================
print("\n========== [完成] 导出基线模型汇总报表 ==========")
results_df = pd.DataFrame(all_results)
print(results_df.to_markdown(index=False))

excel_path = "baseline_comparison_results.xlsx"
csv_path = "baseline_comparison_results.csv"
try:
    results_df.to_excel(excel_path, index=False)
    print(f"已成功汇总至 Excel: {excel_path}")
except ImportError:
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"指标已保存至 CSV: {csv_path}")