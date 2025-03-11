from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载模型和 tokenizer
MODEL_PATH = 'models'  # 替换为你的BERT模型路径
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=4,ignore_mismatched_sizes=True)
model.eval()

def classify_texts(texts):
    #inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**texts)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.tolist()  # 返回每个文本的分类概率