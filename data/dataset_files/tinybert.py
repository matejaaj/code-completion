import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
import tensorflow as tf



file_path = '/content/drive/MyDrive/notebooks/fakenews.csv'
df = pd.read_csv(file_path)
df = df[['text', 'label']]
df = df.dropna()


label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'].tolist())

train_df, test_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=0)
val_df, test_df = train_test_split(test_df, train_size=0.5, shuffle=True, random_state=0)




tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)

tokenized_train = train_dataset.map(tokenize_data, batched=True)
tokenized_val = val_dataset.map(tokenize_data, batched=True)
tokenized_test = test_dataset.map(tokenize_data, batched=True)


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/results/tinybert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model('/content/drive/MyDrive/tinybert')


from sklearn.metrics import classification_report
import numpy as np

predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=1)

labels = tokenized_test['label']

print("Validation Classification Report:")
print(classification_report(labels, preds, target_names=["Class 0", "Class 1"]))