import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载数据
data_path = 'titles_yahoo_financial_616.csv'
data = pd.read_csv(data_path, encoding='utf-8')

print(data.head())


# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
model.eval()  # 将模型设置为评估模式


def preprocess(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

def predict(texts):
    inputs = preprocess(texts)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs

# 进行预测
from tqdm import tqdm  # 导入tqdm库

results = []
for title in tqdm(data['News Titles'], desc="Analyzing Sentiments"):
    probs = predict(title)
    label_id = torch.argmax(probs, dim=1).item()  # 获取概率最高的类别ID
    label = model.config.id2label[label_id]  # 将类别ID转换为标签
    results.append(label)

# 将结果添加到原始DataFrame
data['Sentiment'] = results

# 保存或显示结果
print(data.head())
data.to_csv('path_to_your_file/sentiment_output.csv', index=False)

# ---------------------------------------------------------------------------
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
finbert_df = pd.read_csv('sentiment_output.csv')
svm_df = pd.read_csv('SVM_output.csv')
lstm_df = pd.read_csv('LSTM_output.csv')

# Display unique values in predicted columns to understand the labels
print("Unique values in FinBERT Sentiment column:")
print(finbert_df['Sentiment'].unique(), "\n")

print("Unique values in SVM flag_svm column:")
print(svm_df['flag_svm'].unique(), "\n")

print("Unique values in LSTM flag column:")
print(lstm_df['flag'].unique(), "\n")

# Map labels to a common format if necessary
label_mapping = {
    'negative': 'Negative',
    'neutral': 'Neutral',
    'positive': 'Positive',
    # Add other mappings if necessary
}

svm_df['flag_svm'] = svm_df['flag_svm'].map(label_mapping).fillna(svm_df['flag_svm'])
lstm_df['flag'] = lstm_df['flag'].map(label_mapping).fillna(lstm_df['flag'])


# Function to evaluate the model performance
def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, zero_division=0)

    return accuracy, precision, recall, f1, cm, report


# Assuming the true labels are the same in all files
true_labels = finbert_df['Sentiment']
finbert_pred = finbert_df['Sentiment']
svm_pred = svm_df['flag_svm']
lstm_pred = lstm_df['flag']

# Evaluate each model
finbert_results = evaluate_model(true_labels, finbert_pred)
svm_results = evaluate_model(true_labels, svm_pred)
lstm_results = evaluate_model(true_labels, lstm_pred)


# Print evaluation results
def print_evaluation_results(model_name, results):
    accuracy, precision, recall, f1, cm, report = results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    print("\n")


print_evaluation_results("FinBERT", finbert_results)
print_evaluation_results("SVM", svm_results)
print_evaluation_results("LSTM", lstm_results)


# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()


# Plot confusion matrices
plot_confusion_matrix(finbert_results[4], "FinBERT")
plot_confusion_matrix(svm_results[4], "SVM")
plot_confusion_matrix(lstm_results[4], "LSTM")
# ---------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the CSV file
file_path = 'sentiment_output.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

# Set up the visual style
sns.set(style="whitegrid")
colors_pie = sns.cubehelix_palette(rot=.5, dark=0.2, light=0.8)

# 1. Distribution of Sentiments - Pie Chart
plt.figure(figsize=(10, 6))
data['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=colors_pie)
plt.title('Distribution of Sentiments', fontname='Arial', fontsize=14)
plt.ylabel('')
plt.savefig('sentiment_pie_chart.png', bbox_inches='tight')
plt.show()

# 2. Word Clouds for Each Sentiment
sentiments = data['Sentiment'].unique()
for sentiment in sentiments:
    subset = data[data['Sentiment'] == sentiment]
    text = ' '.join(subset['News Titles'].values)
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap=sns.cubehelix_palette(as_cmap=True)).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment', fontname='Arial', fontsize=14)
    plt.savefig(f'wordcloud_{sentiment.lower()}.png', bbox_inches='tight')
    plt.show()


