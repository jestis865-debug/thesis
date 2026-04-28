import pandas as pd
import numpy as np
import re
import jieba
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# 消除 tokenizer 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128

print(f"✅ 初始化成功！当前使用计算设备: {DEVICE}")

# ==========================================
# 阶段一：基础数据与停用词准备
# ==========================================
print("\n========== [阶段一] 加载数据与停用词 ==========")
df = pd.read_csv('output.csv')
df = df.dropna(subset=['Review Text', 'Rating'])
df = df[df['Review Text'].str.strip() != '']
df['label'] = df['Rating'].apply(lambda x: int(float(x)) - 1)

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&#[0-9]+;', '', text)
    text = re.sub(r'([~!?。，！、])\1+', r'\1', text)
    return text.strip()

df['clean_text'] = df['Review Text'].apply(clean_text)
df = df[df['clean_text'] != '']

# 提取所有实际打分为 1星(0) 和 2星(1) 的数据
bad_reviews_df = df[df['label'].isin([0, 1])]
all_bad_texts = bad_reviews_df['clean_text'].values
all_bad_labels = bad_reviews_df['label'].values

print(f"📊 数据库提取完毕。共发现实际差评数据: {len(all_bad_texts)} 条。")

# 准备停用词
ignore_words = set([
    '显得', '看起来', '简直', '这个', '那个', '什么', '怎么', '还是', '就是', 
    '可以', '没有', '不过', '所以', '感觉', '觉得', '认为', '完全', '一般',
    '东西', '真的', '不是', '有点', '收到', '一个', '如果', '根本', '确实',
    '无法', '起来', '非常', '像是', '令人', '过于', '喜欢', '实在', '真是', '可惜'
])
try:
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    ignore_words.update(stopwords)
except FileNotFoundError:
    pass

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


# ==========================================
# 阶段二：核心业务探针引擎
# ==========================================
def extract_trigger_words(model_dir, base_model_name, is_lora, display_name):
    print(f"\n\n{'='*50}")
    print(f"🔍 开始解析模型: 【{display_name}】")
    print(f"{'='*50}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir if not is_lora else base_model_name)
    macro_bad_loader = DataLoader(TransformerDataset(all_bad_texts, all_bad_labels, tokenizer, MAX_LEN), batch_size=32, shuffle=False)

    if is_lora:
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5, attn_implementation="eager")
        model = PeftModel.from_pretrained(base_model, model_dir).to(DEVICE)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=5, attn_implementation="eager").to(DEVICE)
        
    model.eval()
    bad_reviews_trigger_words = []
    
    print("⏳ 正在开启黑盒，执行词语级注意力映射...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(macro_bad_loader):
            input_ids, attn_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
                
            _, predicted = torch.max(outputs.logits, 1)
            last_layer_attn = outputs.attentions[-1] 
            
            for i in range(input_ids.size(0)):
                if predicted[i].item() in [0, 1]:
                    original_text = str(all_bad_texts[batch_idx * 32 + i])
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
                    cls_scores = last_layer_attn[i].mean(dim=0)[0] 
                    
                    clean_chars, clean_scores = [], []
                    for tok, score in zip(tokens[1:], cls_scores[1:]): 
                        if tok == '[SEP]': break
                        tok = tok.replace('##', '') 
                        if tok == '[UNK]': tok = '' 
                        for char in tok:
                            clean_chars.append(char)
                            clean_scores.append(score.item())
                            
                    words = jieba.lcut(original_text)
                    word_attn_pairs = []
                    char_idx = 0
                    
                    for w in words:
                        w_strip = w.strip()
                        if not w_strip: continue
                        w_len = len(w_strip)
                        if char_idx + w_len <= len(clean_scores):
                            avg_score = sum(clean_scores[char_idx : char_idx+w_len]) / w_len
                            word_attn_pairs.append((w_strip, avg_score))
                            char_idx += w_len
                        else:
                            break 
                            
                    valid_words = []
                    for word, score in word_attn_pairs:
                        if len(word) > 1 and (word not in ignore_words) and (not word.isnumeric()):
                            valid_words.append((word, score))
                            
                    valid_words.sort(key=lambda x: x[1], reverse=True)
                    bad_reviews_trigger_words.extend([pair[0] for pair in valid_words[:3]])

    word_counts = Counter(bad_reviews_trigger_words)
    top_20 = word_counts.most_common(20)
    
    print(f"\n💡 【{display_name}】 捕获的核心差评原因 Top 20：")
    for i, (word, count) in enumerate(top_20):
        print(f"Top {i+1:02} | 靶点词: 【 {word.ljust(6, ' ')} 】 | 拦截频次: {count}")
        
    del model
    torch.cuda.empty_cache()
    
    return top_20 # 返回 Top 20 列表供后续生成表格

# ==========================================
# 阶段三：执行评测并保存为表格
# ==========================================

# 1. 提取模型一结果
top20_roberta = extract_trigger_words(
    model_dir="./best_models/Model1_RoBERTa_Full", 
    base_model_name="hfl/chinese-roberta-wwm-ext", 
    is_lora=False,
    display_name="模型一: RoBERTa"
)

# 2. 提取模型三结果
top20_electra = extract_trigger_words(
    model_dir="./best_models/Model3_ElectraSmall_Full", 
    base_model_name="hfl/chinese-electra-180g-small-discriminator", 
    is_lora=False,
    display_name="模型三: Electra-Small"
)

print("\n========== [阶段四] 导出靶点词对比表格 ==========")
# 3. 整合数据为 DataFrame
table_data = []
# 考虑到可能某些极端情况下不足20个词，取两者的最大长度
max_len = max(len(top20_roberta), len(top20_electra))

for i in range(max_len):
    word_r, count_r = top20_roberta[i] if i < len(top20_roberta) else ("", 0)
    word_e, count_e = top20_electra[i] if i < len(top20_electra) else ("", 0)
    
    table_data.append({
        "排名": i + 1,
        "RoBERTa_靶点词": word_r,
        "RoBERTa_频次": count_r,
        "Electra_靶点词": word_e,
        "Electra_频次": count_e
    })

df_results = pd.DataFrame(table_data)

# 4. 保存为 Excel 或 CSV
excel_path = "trigger_words_comparison.xlsx"
csv_path = "trigger_words_comparison.csv"

try:
    df_results.to_excel(excel_path, index=False)
    print(f"📊 完美！靶点词对比表格已成功汇总至 Excel: 【{excel_path}】")
except ImportError:
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📊 靶点词对比表格已保存为 CSV 文件: 【{csv_path}】 (建议 pip install openpyxl 获得更好的Excel支持)")

print("\n🎉 全量黑盒透视完成！")