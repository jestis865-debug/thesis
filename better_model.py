import pandas as pd
import numpy as np
import re
import jieba
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, mean_absolute_error, cohen_kappa_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('./best_models', exist_ok=True)
os.makedirs('./confusion_matrices', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
TARGET_NAMES = ['1星(极差)', '2星(较差)', '3星(中等)', '4星(较好)', '5星(极好)']
MAX_LEN = 128

# ==========================================
# 阶段一：统一的数据加载与预处理
# ==========================================
print("\n========== [阶段一] 数据统一加载与预处理 ==========")
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

# 保留 jieba 分词逻辑
df['clean_text'] = df['clean_text'].apply(tokenize_text) 

X_b = df['clean_text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=RANDOM_STATE, stratify=y_train)

classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ==========================================
# 阶段二：定义核心训练与评估引擎
# ==========================================
class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.texts[idx]), max_length=self.max_len, padding='max_length',
                                  truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

def count_parameters(model):
    """统计总参数量与可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def run_experiment(model_path, run_name, use_lora=False, batch_size=16, lr=2e-5, epochs=8, patience=2):
    print(f"\n{'='*40}")
    print(f"开始执行任务: {run_name}")
    print(f"模型路径: {model_path} | LoRA: {use_lora} | LR: {lr} | BatchSize: {batch_size}")
    print(f"{'='*40}")
    
    start_time = time.time()
    save_dir = f"./best_models/{run_name}"
    
    # 1. 准备 DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_loader = DataLoader(TransformerDataset(X_train, y_train, tokenizer, MAX_LEN), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TransformerDataset(X_val, y_val, tokenizer, MAX_LEN), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TransformerDataset(X_test, y_test, tokenizer, MAX_LEN), batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
    
    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["query", "value"],
            modules_to_save=["classifier"]
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
    else:
        model = base_model
        
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*epochs*0.1), num_training_steps=len(train_loader)*epochs)

    # 3. 训练循环
    best_val_qwk = -1.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attn_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                loss = criterion(outputs.logits, labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        # 验证评估
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
                _, preds = torch.max(outputs.logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch['labels'].to(DEVICE).cpu().numpy())
                
        val_qwk = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        print(f"  Epoch {epoch+1}/{epochs} | Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            patience_counter = 0
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    end_time = time.time()
    train_duration = (end_time - start_time) / 60.0

    # 4. 测试集评估 + 参数统计
    if use_lora:
        eval_base = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
        best_model = PeftModel.from_pretrained(eval_base, save_dir).to(DEVICE)
    else:
        best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE)
    
    # 参数统计
    total_params, trainable_params = count_parameters(best_model)
        
    best_model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = best_model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            _, preds = torch.max(outputs.logits, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch['labels'].to(DEVICE).cpu().numpy())
            
    test_preds, test_labels = np.array(test_preds), np.array(test_labels)
    
    # 5. 计算最终指标
    acc = np.mean(test_preds == test_labels)
    acc_pm1 = np.mean(np.abs(test_preds - test_labels) <= 1)
    mae = mean_absolute_error(test_labels, test_preds)
    qwk = cohen_kappa_score(test_labels, test_preds, weights='quadratic')
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    
    # 6. 保存按行归一化的混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix: {run_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f'./{run_name}_cm.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{run_name} 测试完成，耗时: {train_duration:.2f} 分钟 | Test QWK: {qwk:.4f}")
    
    # 清理显存
    del model, best_model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    
    return {
        "模型代号": run_name,
        "训练方式": "LoRA微调" if use_lora else "全量微调",
        "总参数量": f"{total_params:,}",
        "可训练参数量": f"{trainable_params:,}",
        "训练时长(分钟)": round(train_duration, 2),
        "严格准确率(%)": round(acc * 100, 2),
        "容差准确率(Acc±1 %)": round(acc_pm1 * 100, 2),
        "平均绝对误差(MAE)": round(mae, 4),
        "二次卡帕系数(QWK)": round(qwk, 4),
        "宏平均F1(Macro-F1)": round(macro_f1, 4)
    }

# ==========================================
# 阶段三：配置实验列表并批量运行
# ==========================================
experiments = [
    {
        "model_path": "hfl/chinese-roberta-wwm-ext",
        "run_name": "Model1_RoBERTa_Full",
        "use_lora": False,
        "batch_size": 16,
        "lr": 2e-5
    },
    {
        "model_path": "hfl/chinese-macbert-base",
        "run_name": "Model2_MacBERT_Full",
        "use_lora": False,
        "batch_size": 16,
        "lr": 2e-5
    },
    {
        "model_path": "hfl/chinese-electra-180g-small-discriminator",
        "run_name": "Model3_ElectraSmall_Full",
        "use_lora": False,
        "batch_size": 64,
        "lr": 5e-5
    },
    {
        "model_path": "hfl/chinese-macbert-base",
        "run_name": "Model2_MacBERT_LoRA", 
        "use_lora": True,
        "batch_size": 32,
        "lr": 3e-4
    }
]

all_results = []

for exp in experiments:
    result_dict = run_experiment(
        model_path=exp["model_path"],
        run_name=exp["run_name"],
        use_lora=exp["use_lora"],
        batch_size=exp["batch_size"],
        lr=exp["lr"],
        epochs=8,
        patience=2
    )
    all_results.append(result_dict)

# ==========================================
# 阶段四：汇总结果并导出为表格
# ==========================================
print("\n========== [阶段四] 导出汇总报告 ==========")
results_df = pd.DataFrame(all_results)
column_order = [
    "模型代号", "训练方式", "总参数量", "可训练参数量",
    "训练时长(分钟)", "严格准确率(%)", "容差准确率(Acc±1 %)",
    "平均绝对误差(MAE)", "二次卡帕系数(QWK)", "宏平均F1(Macro-F1)"
]
results_df = results_df[column_order]
print(results_df.to_markdown(index=False))

excel_path = "results.xlsx"
csv_path = "results.csv"
try:
    results_df.to_excel(excel_path, index=False)
    print(f"已成功汇总至 Excel: {excel_path}")
except ImportError:
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已将汇总结果保存为 CSV 文件: {csv_path}")