import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model_and_tokenizer(model_path, tokenizer_path):
    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def run_inference(input_csv_path, output_csv_path, model, tokenizer):
    print(f"Loading data from {input_csv_path}...")
    data_df = pd.read_csv(input_csv_path, header=None)
    data_df.columns = ['Hypothesis', 'Corrected Hypothesis']

    dataset = Dataset.from_pandas(data_df.rename(columns={'Hypothesis': 'input', 'Corrected Hypothesis': 'target'}))

    predictions = []

    print("Running inference...")
    for item in dataset:
        input_text = item['input']


        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        outputs = model.generate(input_ids, max_length=512)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(decoded_output)

    data_df['Predictions'] = predictions
    data_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


model_path = "/raid/ganesh/pdadiga/ByT5/Model/llmfine/output24third"
tokenizer_path = "/raid/ganesh/pdadiga/ByT5/Model/llmtoken/output24third"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)


input_csv = "/raid/ganesh/pdadiga/ByT5/Dataset/testIC.csv"  # Replace with input file path
output_csv = "/raid/ganesh/pdadiga/ByT5/output/predINDIC.csv"  # Replace with output file path
run_inference(input_csv, output_csv, model, tokenizer)
