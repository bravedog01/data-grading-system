import os
import csv
import torch
import pandas as pd
from io import BytesIO
from flask import Flask, render_template, request,jsonify
from flask import send_file
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载预训练模型和分词器
model_name = "bestmodel"
tokenizer = BertTokenizer.from_pretrained("bertmodel")
model = BertForSequenceClassification.from_pretrained(model_name,ignore_mismatched_sizes=True)
model.eval()
results_ii=[]
def safe_get_text(row):
    """智能获取文本内容"""
    # 尝试常见列名
    for col in ['text', 'content', 'comment']:
        if col in row:
            return row[col]
    
    # 尝试首列
    first_column = next(iter(row.values()), None)
    if first_column:
        return first_column
    
    raise ValueError("无法识别文本列")

# 使用方式

def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores, predicted_labels = torch.max(probabilities, dim=1)
    return predicted_labels.tolist(), scores.tolist()

@app.route('/download_unmodified')
def download_unmodified():
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], "results.csv")
    
    # 使用 utf-8-sig 解决 Excel 乱码问题
    with open(results_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "score"])
        writer.writeheader()
        
        # 确保 results_ii 不是空的
        if results_ii:
            for result in results_ii:
                writer.writerow(result)

    return send_file(results_path, as_attachment=True)

@app.route('/download_modified', methods=['POST'])
def download_modified():
    # 确保使用绝对路径
    upload_dir = app.config['UPLOAD_FOLDER']
    mod_path = os.path.join(upload_dir, 'modified_results.csv')

    # 接收并处理数据
    data = request.json
    original_data = data['original']
    modifications = {int(m['index']): m['level'] for m in data.get('modifications', [])}

    # 应用修改
    modified_data = [row.copy() for row in original_data]
    for idx, new_level in modifications.items():
        if idx < len(modified_data):  # 确保索引有效
            modified_data[idx]['label'] = new_level

    # 保存文件
    try:
        df = pd.DataFrame(modified_data)
        df.to_csv(mod_path, index=False, encoding='utf-8-sig')
        print(f"[DEBUG] 文件已保存至: {mod_path}")  # 调试输出
    except Exception as e:
        print(f"[ERROR] 文件保存失败: {str(e)}")
        return jsonify({"error": "文件保存失败"}), 500

    # 返回内存流
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='modified_results.csv'
    )



@app.route('/', methods=['GET', 'POST'])
def index():
    global results_ii
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="请选择文件")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="无效文件")
        
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # 读取CSV文件
            
            texts = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                text_column = request.form.get('text_column', 'text')
                texts = [safe_get_text(row) for row in reader]
                
            # 进行预测
            labels, scores = predict(texts)
            
            # 组装结果
            results = [{
                'text': text,
                'label':  f"{label + 1}级",
                'score': f"{score:.4f}"
            } for text, label, score in zip(texts, labels, scores)]
            results_ii=results
            return render_template('index.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)