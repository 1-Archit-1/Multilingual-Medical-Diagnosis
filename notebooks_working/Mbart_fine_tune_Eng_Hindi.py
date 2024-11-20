# %%

import os
import sys
import time
from tqdm import tqdm


# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"
os.environ["HF_HOME"] = "~/scratch/hf-cache"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
token=""
print(os.environ['WANDB_DISABLED'])  # Should output "true"
print(os.environ['HF_HOME'])  # Should output "~/scratch/hf-cache"
#print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])

output_file = open('mbart_output_eng_hindi.log', 'w')
sys.stdout = output_file
sys.stderr = output_file



# %%
import re
import numpy as np 
import pandas as pd 
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)

from trl import SFTTrainer
import torch
from pynvml import *
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer
from huggingface_hub import HfApi, login
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# %%
# Specify the download directory for NLTK data
nltk.data.path.append('./nltk_data')
nltk.download('all', download_dir='./nltk_data')

# %%
def read_token_and_login(token_file):
    with open(token_file, 'r') as file:
        token = file.read().strip()
    api = HfApi()
    login(token=token)
    return api


# %%
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def clear_cuda_cache():
    torch.cuda.empty_cache()

# %%
def get_pretrained_mbart_large_50_many_to_many_mmt():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, force_download=True)
    model = MBartForConditionalGeneration.from_pretrained(model_name, force_download=True)
    return tokenizer, model

# %%
def filter_sentences(example):
    # Check sentence length
    if not (3 < len(example['translation']['en'].split()) < 30):
        return False
    if not (3 < len(example['translation']['hi'].split()) < 30):
        return False
    
    # Check for non-ASCII non-Unicode characters in Hindi text
    if re.search(r'[^\u0000-\u007F\u0900-\u097F]', example['translation']['hi']):
        return False
    
    # Hook for further restrictions (can be customized)
    # Example: if 'specific_word' in example['translation']['en']:
    #     return False
    
    return True

# %%
def get_reduced_dataset(dataset_name, train_size=14000, val_size=2000, test_size=4000):
    orig_data_set = load_dataset(dataset_name)
    print(orig_data_set)
    # Filter the dataset based on the criteria
    filtered_dataset = orig_data_set['train'].filter(filter_sentences)
    print(filtered_dataset)
    
    # Split the filtered dataset into train, validation, and test sets
    train_val_test_split = filtered_dataset.train_test_split(test_size=val_size + test_size, seed=42)
    val_test_split = train_val_test_split['test'].train_test_split(test_size=test_size, seed=42)
    
    small_data_set = DatasetDict({
        'train': train_val_test_split['train'].select(range(train_size)),
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })

    # Verify the size of the new dataset
    print(small_data_set)
    print(f"New train set size: {len(small_data_set['train'])}")
    print(f"New validation set size: {len(small_data_set['validation'])}")
    print(f"New test set size: {len(small_data_set['test'])}")
    
    return small_data_set


# %%
def preprocess_function(examples, tokenizer):
    global last_print_time
    current_time = time.time()
    
    if current_time - last_print_time >= 10:
        print("Examples:", examples['translation'][:2])
        last_print_time = current_time
    
    inputs = [ex['en'] for ex in examples['translation'] if ex['en'] is not None]
    targets = [ex['hi'] for ex in examples['translation'] if ex['hi'] is not None]
    
    if current_time - last_print_time >= 10:
        print("Inputs:", inputs[:2])
        print("Targets:", targets[:2])
    
    if len(inputs) == 0 or len(targets) == 0:
        return {}
    
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    
    if current_time - last_print_time >= 10:
        print("Model Inputs:", {k: v[:2] for k, v in model_inputs.items()})
    
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)
    
    if current_time - last_print_time >= 10:
        print("Labels:", {k: v[:2] for k, v in labels.items()})
    
    if "input_ids" not in labels or len(labels["input_ids"]) == 0:
        print("Labels are empty or not properly structured")
        return {}
    
    model_inputs["labels"] = labels["input_ids"]
    
    if current_time - last_print_time >= 10:
        print("Final Model Inputs:", {k: v[:2] for k, v in model_inputs.items()})
    
    return model_inputs

# %%
def prepare_model_for_training(model, tokenizer, tokenized_datasets, output_dir="./results", learning_rate=2e-5, batch_size=64, num_train_epochs=5, gradient_accumulation_steps=4):
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type="linear",
        warmup_steps=500,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )

    # Freeze all layers except the last few layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers
    for param in model.model.decoder.layers[-2:].parameters():
        param.requires_grad = True

    # Unfreeze the classification head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return trainer

# %%
def fine_tune_and_save(trainer, model, tokenizer, output_dir="./mbart_fine_tune_eng_hindi"):
    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# %%
def load_fine_tuned_model(output_dir="./mbart_fine_tune_eng_hindi"):
    model = MBartForConditionalGeneration.from_pretrained(output_dir)
    tokenizer = MBart50TokenizerFast.from_pretrained(output_dir)
    return model, tokenizer

# %%
def translate_text(model, tokenizer, input_text, src_lang="en_XX", tgt_lang="hi_IN"):
    # Tokenize the input text
    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(input_text, return_tensors="pt")

    # Generate translation
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    # Decode the generated tokens
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print("Translated text:", translated_text)
    return translated_text



# %%
def prepare_test_data(small_data_set, tokenizer, num_examples=100):
    # Load the test data
    test_data = small_data_set['test']
    print(test_data['translation'][0])
    print(len(test_data['translation']))
    
    # Select a subset of the test data
    test_data = test_data.select(range(num_examples))
    print(test_data['translation'][0])
    print(len(test_data['translation']))

    # Preprocess the test data
    def preprocess_test_data(examples):
        inputs = [ex['en'] for ex in examples['translation'] if ex['en'] is not None]
        model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
        return model_inputs

    tokenized_test_data = test_data.map(preprocess_test_data, batched=True, remove_columns=["translation"])
    
    return test_data, tokenized_test_data

#count = 0

# %%
def perform_translation_testing(model, tokenizer, test_data, tokenized_test_data, src_lang="en_XX", tgt_lang="hi_IN"):
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    count = 0
    batch_size = 64

    def generate_translation(batch):
        nonlocal count
        # Ensure input_ids and attention_mask are tensors
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        
        count += 1
        print(f"Processing batch {count}")
        
        # Generate translation
        generated_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
            #max_length=128,
            #num_beams=5,
            #early_stopping=True
        )
        # Decode the generated tokens
        batch["translation"] = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return batch

    torch.cuda.empty_cache()

    translated_test_data = tokenized_test_data.map(generate_translation, batched=True, batch_size=batch_size)

    # Extract test_data from small_data_set
    #test_data = small_data_set['test']

    # Print the first 5 translations for inspection
    for i in range(5):
        print(f"Original: {test_data[i]['translation']['en']}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation']['hi']}")
        print()

    return translated_test_data

# %%


def evaluate_translations_bertscore(test_data, translated_test_data):
    references = [test_data[i]['translation']['hi'] for i in range(len(test_data))]
    translations = [translated_test_data[i]['translation'] for i in range(len(test_data))]
    
    P, R, F1 = score(translations, references, lang="hi", verbose=True)
    
    # Print BERTScore for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation']['en']}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation']['hi']}")
        print(f"BERTScore F1: {F1[i].item():.4f}")
        print()
    
    print(f"Average BERTScore F1: {F1.mean().item():.4f}")

def evaluate_translations_rouge(test_data, translated_test_data, tokenizer):
    """
    Evaluates the translations using ROUGE score.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        tokenizer: The tokenizer for the model.

    Returns:
        None
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    references = [test_data[i]['translation']['hi'] for i in range(len(test_data))]
    translations = [translated_test_data[i]['translation'] for i in range(len(test_data))]
    
    rouge_scores = [scorer.score(ref, trans) for ref, trans in zip(references, translations)]
    avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    print(f"Average ROUGE-1 F1 score: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2 F1 score: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L F1 score: {avg_rougeL:.4f}")

    # Print ROUGE score for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation']['en']}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation']['hi']}")
        print(f"ROUGE-1 F1 score: {rouge_scores[i]['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2 F1 score: {rouge_scores[i]['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L F1 score: {rouge_scores[i]['rougeL'].fmeasure:.4f}")
        print()

def evaluate_translations_bleu(test_data, translated_test_data, tokenizer):
    """
    Evaluates the translations using BLEU score.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        tokenizer: The tokenizer for the model.

    Returns:
        None
    """
    smoothie = SmoothingFunction().method4
    references = [[tokenizer.tokenize(test_data[i]['translation']['hi'])] for i in range(len(test_data))]
    translations = [tokenizer.tokenize(translated_test_data[i]['translation']) for i in range(len(test_data))]
    
    bleu_scores = [sentence_bleu(ref, trans, smoothing_function=smoothie) for ref, trans in zip(references, translations)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu:.4f}")

    # Print BLEU score for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation']['en']}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation']['hi']}")
        print(f"BLEU score: {bleu_scores[i]:.4f}")
        print()

def evaluate_translations_meteor(test_data, translated_test_data, tokenizer):
    """
    Evaluates the translations using METEOR score.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        tokenizer: The tokenizer for the model.

    Returns:
        None
    """
    references = [tokenizer.tokenize(test_data[i]['translation']['hi']) for i in range(len(test_data))]
    translations = [tokenizer.tokenize(translated_test_data[i]['translation']) for i in range(len(test_data))]
    
    meteor_scores = [meteor_score([ref], trans) for ref, trans in zip(references, translations)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    print(f"Average METEOR score: {avg_meteor:.4f}")

    # Print METEOR score for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation']['en']}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation']['hi']}")
        print(f"METEOR score: {meteor_scores[i]:.4f}")
        print()

# %% [markdown]
# # ACTUAL CODE FLOW STARTS NOW!!!

# %%
api = read_token_and_login('hf_token')

# %%

original_tokenizer, original_model = get_pretrained_mbart_large_50_many_to_many_mmt()

# %%

dataset_name = "cfilt/iitb-english-hindi"
#small_data_set = get_reduced_dataset(dataset_name)
small_data_set = get_reduced_dataset(dataset_name, train_size = 56000, val_size=8000, test_size=16000)
# Initialize a global variable to keep track of the last print time
last_print_time = time.time()

# %%
tokenized_datasets = small_data_set.map(lambda examples: preprocess_function(examples, original_tokenizer), batched=True, remove_columns=["translation"])

# %%
#Testing original untuned model
test_data, tokenized_test_data = prepare_test_data(small_data_set, original_tokenizer)

translated_test_data_untuned = perform_translation_testing(original_model, original_tokenizer, test_data, tokenized_test_data)

evaluate_translations_bertscore(test_data, translated_test_data_untuned)

# Evaluate translations using BLEU score
evaluate_translations_bleu(test_data, translated_test_data_untuned, original_tokenizer)

# Evaluate translations using ROUGE score
evaluate_translations_rouge(test_data, translated_test_data_untuned, original_tokenizer)

# Evaluate translations using METEOR score
evaluate_translations_meteor(test_data, translated_test_data_untuned, original_tokenizer)


# %%
trainer = prepare_model_for_training(original_model, original_tokenizer, tokenized_datasets)

# %%
fine_tune_and_save(trainer, original_model, original_tokenizer)

# %%
model, tokenizer = load_fine_tuned_model()

# %%
# Example usage
input_text = "Stop in the name of the law"
translated_text = translate_text(model, tokenizer, input_text)

# %%

test_data, tokenized_test_data = prepare_test_data(small_data_set, tokenizer)

# %%
translated_test_data = perform_translation_testing(model, tokenizer, test_data, tokenized_test_data)

# %%
evaluate_translations_bertscore(test_data, translated_test_data)

# %%
# Evaluate translations using BLEU score
evaluate_translations_bleu(test_data, translated_test_data, tokenizer)

# Evaluate translations using ROUGE score
evaluate_translations_rouge(test_data, translated_test_data, tokenizer)

# Evaluate translations using METEOR score
evaluate_translations_meteor(test_data, translated_test_data, tokenizer)

# %%
#translated_test_data_untuned = perform_translation_testing(original_model, original_tokenizer, test_data, tokenized_test_data)

# %%
#evaluate_translations_bertscore(test_data, translated_test_data_untuned)


output_file.close()