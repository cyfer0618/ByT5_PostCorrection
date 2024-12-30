from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig, EarlyStoppingCallback
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from jiwer import cer, wer
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
import wandb
import time
wandb.init(project="1million")

invalid_lines_file = "/raid/ganesh/pdadiga/ByT5/invalid_lines.txt"


def log_invalid_line(line, reason):
    with open(invalid_lines_file, "a") as f:
        f.write(f"{line} -- Reason: {reason}\n")


@dataclass
class T5Collator:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, batch):
        # Skip invalid lines
        valid_inputs = []
        invalid_lines = []
        
        for item in batch:
            if 'input' in item and 'target' in item:
                valid_inputs.append(item)
            else:
                invalid_lines.append(str(item))

        with open("invalid_lines.txt", "a") as f:
            for line in invalid_lines:
                f.write(line + "\n")
        
        inputs = [item['input'] for item in valid_inputs]
        targets = [item['target'] for item in valid_inputs]

        input_encoding = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        target_encoding = self.tokenizer(targets, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        labels = target_encoding.input_ids.masked_fill(target_encoding.input_ids == self.tokenizer.pad_token_id, -100)

        return {
            'input_ids': input_encoding.input_ids,
            'attention_mask': input_encoding.attention_mask,
            'labels': labels
        }



print("start time", time.time())

df = pd.read_csv('/raid/ganesh/pdadiga/ByT5/Dataset/inputdataset24.csv', header=None)
df.columns = ['Hypothesis', 'Corrected Hypothesis']


train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df = train_df.rename(columns={"Hypothesis": "target", "Corrected Hypothesis": "input"})
val_df = val_df.rename(columns={"Hypothesis": "target", "Corrected Hypothesis": "input"})

train_dataset = Dataset.from_pandas(train_df.dropna(subset=["input", "target"], how="any"))
val_dataset = Dataset.from_pandas(val_df.dropna(subset=["input", "target"], how="any"))

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
data_collator = T5Collator(tokenizer)

config = AutoConfig.from_pretrained("google/byt5-small")
config.dropout_rate = 0.2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small", config=config, device_map='cuda')

training_args = Seq2SeqTrainingArguments(
    output_dir="/raid/ganesh/pdadiga/ByT5/output",
    num_train_epochs=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    logging_dir="/raid/ganesh/pdadiga/ByT5/modellogs/logs",
    remove_unused_columns=False,
    logging_steps=25000,
    save_steps=25000,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=25000,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=False,
    dataloader_num_workers=32,
    bf16=True,
    gradient_checkpointing=False,
    report_to="wandb",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

accelerator = Accelerator()
model, trainer = accelerator.prepare(model, trainer)

print("Training!")
trainer.train()
print("Training Done!")
print("training time", time.time())


model.save_pretrained("/raid/ganesh/pdadiga/ByT5/Model/llmfine/output24third")
tokenizer.save_pretrained("/raid/ganesh/pdadiga/ByT5/Model/llmtoken/output24third")

def test_model(csv_file_path, model, tokenizer):
    print(f"Loading test data from {csv_file_path}...")
    test_df = pd.read_csv(csv_file_path, header=None)
    test_df.columns = ['Ground Truth', 'Hypothesis']
    
   
    test_df = test_df.rename(columns={"Ground Truth": "target", "Hypothesis": "input"})


    test_df = test_df[test_df['input'].notna() & (test_df['input'] != 'N/A')]


    test_dataset = Dataset.from_pandas(test_df)

    predictions = []
    # references = []

    print("Generating predictions...")
    for item in test_dataset:
        input_text = item['input']

       
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        outputs = model.generate(input_ids, max_length=512)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(decoded_output)

    

    test_df = test_df.iloc[:len(predictions)]  # Ensure test_df aligns with predictions
    test_df['Predictions'] = predictions
    test_df.to_csv("/raid/ganesh/pdadiga/ByT5/output/predw2v.csv", index=False)
    print("Predictions saved")
    print("end time",time.time())



test_csv_file_path = '/raid/ganesh/pdadiga/ByT5/Dataset/testw2v.csv'
test_model(test_csv_file_path, model, tokenizer)
