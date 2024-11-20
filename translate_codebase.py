# Description: This script contains functions to translate from English to Hindi and viceversa using the mBART model and mt5-small model. 

# Imports

import os
import sys
import time
from tqdm import tqdm, tqdm_notebook
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict
from IPython.display import display
from ipywidgets import widgets
import torch
from torch.nn import functional as F
from pynvml import *
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, AdamW, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from huggingface_hub import HfApi, login
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score



# Environment variables and settings

os.environ['WANDB_DISABLED']="true"
os.environ["HF_HOME"] = "~/scratch/hf-cache"

token=""
print(os.environ['WANDB_DISABLED'])  # Should output "true"
print(os.environ['HF_HOME'])  # Should output "~/scratch/

output_file = open('logger.log', 'w')
sys.stdout = output_file
sys.stderr = output_file

nltk.data.path.append('./nltk_data')
nltk.download('all', download_dir='./nltk_data')

LANG_TOKEN_MAPPING = {
    'hi': '',
    'en': ''
}
max_seq_len = 25

start_times = {}

# Generic Helper functions (May not be needed!)

def read_token_and_login(token_file):
    with open(token_file, 'r') as file:
        token = file.read().strip()
    api = HfApi()
    login(token=token)
    return api

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def clear_cuda_cache():
    torch.cuda.empty_cache()

# Clock functions to measure execution time
def clock_begin(function_name):
    """
    Records the start time of the specified function.

    Args:
        function_name (str): Name of the function to track.
    """
    global start_times
    start_times[function_name] = time.time()

def clock_end(function_name):
    """
    Calculates the total execution time of the specified function
    since clock_begin was called, and logs it to output_file with
    the prefix 'TIME_'.

    Args:
        function_name (str): Name of the function to track.
    """
    global start_times
    if function_name in start_times:
        end_time = time.time()
        total_time = end_time - start_times[function_name]
        log_message = f"TIME_{function_name}: {total_time:.2f} seconds"
        # Since sys.stdout is redirected to output_file, use print()
        print(log_message)
        # Flush the output to ensure it's written to the file
        sys.stdout.flush()
        # Remove the start time entry
        del start_times[function_name]
    else:
        # If clock_begin was not called, log an error message
        error_message = f"ERROR: clock_begin was not called for function '{function_name}'"
        print(error_message)
        sys.stdout.flush()


# METRICS Functions


def evaluate_translations_bertscore(test_data, translated_test_data, src_lang='hi', tgt_lang='en', bert_lang='en'):
    """
    Evaluates the translations using BERTScore.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        src_lang: The source language key in the dataset.
        tgt_lang: The target language key in the dataset.
        bert_lang: The language to use for BERTScore evaluation.

    Returns:
        None
    """
    references = [test_data[i]['translation'][tgt_lang] for i in range(len(test_data))]
    translations = [translated_test_data[i]['translation'] for i in range(len(test_data))]
    
    P, R, F1 = score(translations, references, lang=bert_lang, verbose=True)
    
    # Print BERTScore for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation'][src_lang]}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation'][tgt_lang]}")
        print(f"BERTScore F1: {F1[i].item():.4f}")
        print()
    
    print(f"Average BERTScore F1: {F1.mean().item():.4f}")


def evaluate_translations_rouge(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en'):
    """
    Evaluates the translations using ROUGE score.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        tokenizer: The tokenizer for the model.
        src_lang: The source language key in the dataset.
        tgt_lang: The target language key in the dataset.

    Returns:
        None
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    references = [test_data[i]['translation'][tgt_lang] for i in range(len(test_data))]
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
        print(f"Original: {test_data[i]['translation'][src_lang]}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation'][tgt_lang]}")
        print(f"ROUGE-1 F1 score: {rouge_scores[i]['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2 F1 score: {rouge_scores[i]['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L F1 score: {rouge_scores[i]['rougeL'].fmeasure:.4f}")
        print()


def evaluate_translations_bleu(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en'):
    """
    Evaluates the translations using BLEU score.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        tokenizer: The tokenizer for the model.
        src_lang: The source language key in the dataset.
        tgt_lang: The target language key in the dataset.

    Returns:
        None
    """
    smoothie = SmoothingFunction().method4
    references = [[tokenizer.tokenize(test_data[i]['translation'][tgt_lang])] for i in range(len(test_data))]
    translations = [tokenizer.tokenize(translated_test_data[i]['translation']) for i in range(len(test_data))]
    
    bleu_scores = [sentence_bleu(ref, trans, smoothing_function=smoothie) for ref, trans in zip(references, translations)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu:.4f}")

    # Print BLEU score for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation'][src_lang]}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation'][tgt_lang]}")
        print(f"BLEU score: {bleu_scores[i]:.4f}")
        print()


def evaluate_translations_meteor(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en'):
    """
    Evaluates the translations using METEOR score.

    Args:
        test_data: The original test data.
        translated_test_data: The test data with translations.
        tokenizer: The tokenizer for the model.
        src_lang: The source language key in the dataset.
        tgt_lang: The target language key in the dataset.

    Returns:
        None
    """
    references = [tokenizer.tokenize(test_data[i]['translation'][tgt_lang]) for i in range(len(test_data))]
    translations = [tokenizer.tokenize(translated_test_data[i]['translation']) for i in range(len(test_data))]
    
    meteor_scores = [meteor_score([ref], trans) for ref, trans in zip(references, translations)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    print(f"Average METEOR score: {avg_meteor:.4f}")

    # Print METEOR score for each example
    for i in range(len(test_data)):
        print(f"Original: {test_data[i]['translation'][src_lang]}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation'][tgt_lang]}")
        print(f"METEOR score: {meteor_scores[i]:.4f}")
        print()







# Common Data Preprocessing Functions



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


def get_reduced_dataset(dataset_name, train_size=56000, val_size=8000, test_size=16000):
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


def prepare_test_data(small_data_set, tokenizer, num_examples=100, src_lang='en'):
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
        inputs = [ex[src_lang] for ex in examples['translation'] if ex[src_lang] is not None]
        model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
        return model_inputs

    tokenized_test_data = test_data.map(preprocess_test_data, batched=True, remove_columns=["translation"])
    
    return test_data, tokenized_test_data




def perform_translation_testing(model, tokenizer, test_data, tokenized_test_data, src_lang="en_XX", tgt_lang="hi_IN", model_type="mbart"):
    """
    Performs translation on the test data using the fine-tuned model.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.
        test_data: The original test data.
        tokenized_test_data: The tokenized test data.
        src_lang: The source language code.
        tgt_lang: The target language code.
        model_type: The type of model ("mbart" or "mt5").

    Returns:
        translated_test_data: The test data with translations.
    """
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
        if model_type == "mbart":
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
            )
        else:  # mt5
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode the generated tokens
        batch["translation"] = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return batch

    torch.cuda.empty_cache()

    translated_test_data = tokenized_test_data.map(generate_translation, batched=True, batch_size=batch_size)

    # Print the first 5 translations for inspection
    for i in range(5):
        print(f"Original: {test_data[i]['translation'][src_lang.split('_')[0]]}")
        print(f"Translated: {translated_test_data[i]['translation']}")
        print(f"Reference: {test_data[i]['translation'][tgt_lang.split('_')[0]]}")
        print()

    return translated_test_data


# Mbart related functions

def get_pretrained_mbart_large_50_many_to_many_mmt():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, force_download=True)
    model = MBartForConditionalGeneration.from_pretrained(model_name, force_download=True)
    return tokenizer, model



def preprocess_function_mbart(examples, tokenizer, src_lang='en', tgt_lang='hi'):
    global last_print_time
    current_time = time.time()
    
    if current_time - last_print_time >= 10:
        print("Examples:", examples['translation'][:2])
        last_print_time = current_time
    
    inputs = []
    targets = []
    
    for ex in examples['translation']:
        if ex.get(src_lang) and ex.get(tgt_lang):
            inputs.append(ex[src_lang])
            targets.append(ex[tgt_lang])
    
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



def prepare_model_for_training_mbart(model, tokenizer, tokenized_datasets, output_dir="./results", learning_rate=2e-5, batch_size=64, num_train_epochs=5, gradient_accumulation_steps=4):
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



def fine_tune_and_save_mbart(trainer, model, tokenizer, output_dir="./mbart_fine_tuned"):
    # Train the model
    print("Save directory:", output_dir)
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_fine_tuned_model_mbart(output_dir="./mbart_fine_tuned"):
    print("Loading model from:", output_dir)
    model = MBartForConditionalGeneration.from_pretrained(output_dir)
    tokenizer = MBart50TokenizerFast.from_pretrained(output_dir)
    return model, tokenizer


def translate_text_mbart(model, tokenizer, input_text, src_lang="en_XX", tgt_lang="hi_IN"):
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







# mt5 related functions


def get_pretrained_mt5_small():
    model_name = 'google/mt5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, force_download=True)
    model = model.cuda()
    return model, tokenizer


def config_mt5_small(tokenizer, model, lang_token_mapping):

    # Add special tokens to the tokenizer
    special_tokens_dict = {'additional_special_tokens': list(lang_token_mapping.values())}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Resize the model's token embeddings to accommodate the new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Optionally, sort the vocabulary by token IDs (if required)
    sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    
    return sorted_vocab



def encode_input_str_mt5_small(text, target_lang, tokenizer, seq_len,
                     lang_token_map=LANG_TOKEN_MAPPING):
  target_lang_token = lang_token_map[target_lang]

  # Tokenize and add special tokens
  input_ids = tokenizer.encode(
      text = target_lang_token + text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)

  return input_ids[0]

def encode_target_str_mt5_small(text, tokenizer, seq_len,
                      lang_token_map=LANG_TOKEN_MAPPING):
  token_ids = tokenizer.encode(
      text = text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)
  
  return token_ids[0]


def process_translation_list_mt5_small(translations_list, lang_token_map, tokenizer, seq_len=128):
    input_ids = []
    output_ids = []

    for translation in translations_list:
        formatted_data = format_translation_data_mt5_small(translation, lang_token_map, tokenizer, seq_len)
        if formatted_data is None:
            continue

        input_token_ids, target_token_ids = formatted_data
        input_ids.append(input_token_ids.tolist())  # Convert tensor to list
        output_ids.append(target_token_ids.tolist())  # Convert tensor to list

    return input_ids, output_ids


def format_translation_data_mt5_small(translation, lang_token_map, tokenizer, seq_len=128, src_lang='en', tgt_lang='hi'):
    """
    Formats a translation example into input and target token IDs.

    Args:
        translation: The translation example to format.
        lang_token_map: A dictionary mapping language codes to special tokens.
        tokenizer: The tokenizer for the model.
        seq_len: The maximum sequence length for the model.
        src_lang: The source language code.
        tgt_lang: The target language code.

    Returns:
        input_token_ids: The input token IDs.
        target_token_ids: The target token IDs.
    """
    # Get the translations for the batch
    input_text = translation.get(src_lang)
    target_text = translation.get(tgt_lang)

    if input_text is None or target_text is None:
        return None

    input_token_ids = encode_input_str_mt5_small(
        input_text, tgt_lang, tokenizer, seq_len, lang_token_map) # Note we use tgt_lang and not src_lang

    target_token_ids = encode_target_str_mt5_small(
        target_text, tokenizer, seq_len, lang_token_map)

    return input_token_ids, target_token_ids



def transform_batch_mt5_small(batch, lang_token_map, tokenizer, src_lang='en', tgt_lang='hi'):
    """
    Transforms a batch of data into input and target token IDs.

    Args:
        batch: The batch of data to transform.
        lang_token_map: A dictionary mapping language codes to special tokens.
        tokenizer: The tokenizer for the model.
        src_lang: The source language code.
        tgt_lang: The target language code.

    Returns:
        batch_input_ids: The input token IDs for the batch.
        batch_target_ids: The target token IDs for the batch.
    """
    inputs = []
    targets = []
    for translation_set in batch['translation']:
        formatted_data = format_translation_data_mt5_small(
            translation_set, lang_token_map, tokenizer, max_seq_len, src_lang, tgt_lang)
        
        if formatted_data is None:
            continue
        
        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))
    
    batch_input_ids = torch.cat(inputs).cuda()
    batch_target_ids = torch.cat(targets).cuda()

    return batch_input_ids, batch_target_ids


def get_data_generator_mt5_small(dataset, lang_token_map, tokenizer, batch_size, src_lang='en', tgt_lang='hi'):
    """
    Generates batches of data for training or evaluation.

    Args:
        dataset: The dataset to generate data from.
        lang_token_map: A dictionary mapping language codes to special tokens.
        tokenizer: The tokenizer for the model.
        batch_size: The batch size for data generation.
        src_lang: The source language code.
        tgt_lang: The target language code.

    Yields:
        Batches of input and target token IDs.
    """
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_input_ids, batch_target_ids = transform_batch_mt5_small(batch, lang_token_map, tokenizer, src_lang, tgt_lang)
        yield batch_input_ids, batch_target_ids




def fine_tune_and_save_model_mt5_small(model, tokenizer, train_dataset, val_dataset, lang_token_mapping, model_path, src_lang='en', tgt_lang='hi'):
    """
    Fine-tunes the model and saves it at checkpoints and at the end of training.

    Args:
        model: The model to be fine-tuned.
        tokenizer: The tokenizer for the model.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        lang_token_mapping: A dictionary mapping language codes to special tokens.
        model_path: The path to save the model.
        src_lang: The source language code.
        tgt_lang: The target language code.

    Returns:
        None
    """
    # Constants
    n_epochs = 8
    batch_size = 16
    print_freq = 50
    checkpoint_freq = 1000
    lr = 5e-4
    n_batches = int(np.ceil(len(train_dataset) / batch_size))
    total_steps = n_epochs * n_batches
    n_warmup_steps = int(total_steps * 0.01)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)
    losses = []

    for epoch_idx in range(n_epochs):
        # Randomize data order
        data_generator = get_data_generator_mt5_small(train_dataset, lang_token_mapping, tokenizer, batch_size, src_lang, tgt_lang)
        
        for batch_idx, (input_batch, label_batch) in tqdm_notebook(enumerate(data_generator), total=n_batches):
            optimizer.zero_grad()

            # Forward pass
            model_out = model(
                input_ids=input_batch,
                labels=label_batch
            )

            # Calculate loss and update weights
            loss = model_out.loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Print training update info
            if (batch_idx + 1) % print_freq == 0:
                avg_loss = np.mean(losses[-print_freq:])
                print('Epoch: {} | Step: {} | Avg. loss: {:.3f} | lr: {}'.format(
                    epoch_idx + 1, batch_idx + 1, avg_loss, scheduler.get_last_lr()[0]))
            
            if (batch_idx + 1) % checkpoint_freq == 0:
                test_loss = eval_model_mt5_small(model, val_dataset, tokenizer, lang_token_mapping, batch_size, src_lang, tgt_lang)
                print('Saving model with test loss of {:.3f}'.format(test_loss))
                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)

    # Save the final model
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    # Graph the loss
    window_size = 50
    smoothed_losses = []
    for i in range(len(losses) - window_size):
        smoothed_losses.append(np.mean(losses[i:i + window_size]))

    plt.plot(smoothed_losses[100:])
    plt.xlabel('Steps')
    plt.ylabel('Smoothed Loss')
    plt.title('Training Loss Over Time')
    plt.show()




def eval_model_mt5_small(model, gdataset, tokenizer, lang_token_mapping, batch_size, src_lang='en', tgt_lang='hi', max_iters=8):
    """
    Evaluates the model on a given dataset and returns the average loss.

    Args:
        model: The model to be evaluated.
        gdataset: The dataset for evaluation.
        tokenizer: The tokenizer for the model.
        lang_token_mapping: A dictionary mapping language codes to special tokens.
        batch_size: The batch size for evaluation.
        src_lang: The source language code.
        tgt_lang: The target language code.
        max_iters: The maximum number of iterations for evaluation.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    test_generator = get_data_generator_mt5_small(gdataset, lang_token_mapping, tokenizer, batch_size, src_lang, tgt_lang)
    eval_losses = []

    for i, (input_batch, label_batch) in enumerate(test_generator):
        if i >= max_iters:
            break

        model_out = model(
            input_ids=input_batch,
            labels=label_batch
        )
        eval_losses.append(model_out.loss.item())

    return np.mean(eval_losses)





def translate_text_mt5_small(input_text, model, tokenizer, target_lang, lang_token_map, seq_len):
    """
    Translates a given text from the source language to the target language.

    Args:
        input_text: The input text to be translated.
        model: The translation model.
        tokenizer: The tokenizer for the model.
        target_lang: The target language code.
        lang_token_map: A dictionary mapping language codes to special tokens.
        seq_len: The maximum sequence length for the model.

    Returns:
        str: The translated text.
    """
    # Encode the input text
    input_ids = encode_input_str_mt5_small(
        text=input_text,
        target_lang=target_lang,
        tokenizer=tokenizer,
        seq_len=seq_len,
        lang_token_map=lang_token_map
    )
    input_ids = input_ids.unsqueeze(0).cuda()

    # Generate the translation
    output_ids = model.generate(input_ids, num_beams=5, max_length=seq_len, early_stopping=True)

    # Decode the generated tokens
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return translated_text




def save_fine_tuned_model_mt5_small(model, tokenizer, model_path):
    """
    Saves the fine-tuned model and tokenizer to the specified path.

    Args:
        model: The fine-tuned model to be saved.
        tokenizer: The tokenizer to be saved.
        model_path: The path to save the model and tokenizer.

    Returns:
        None
    """
    # Ensure the directory exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Save the model and tokenizer
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)


def load_fine_tuned_model_mt5_small(model_path):
    """
    Loads the fine-tuned model and tokenizer from the specified path.

    Args:
        model_path: The path to load the model and tokenizer from.

    Returns:
        model: The loaded fine-tuned model.
        tokenizer: The loaded tokenizer.
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.cuda()

    return model, tokenizer



