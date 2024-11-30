SHOULD_TRAIN = False
#SHOULD_TRAIN = True

import os
import time
import sys
from collections import Counter
import numpy as np
import json
import re

# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"
os.environ["HF_HOME"] = "~/scratch/hf-cache"
token=""
print(os.environ['WANDB_DISABLED'])  # Should output "true"
print(os.environ['HF_HOME'])  # Should output "~/scratch/hf-cache"

output_file = open('logger_tests.log', 'w')
sys.stdout = output_file
sys.stderr = output_file

LANG_TOKEN_MAPPING = {
    'hi': '',
    'en': ''
}
max_seq_len = 25
last_print_time = time.time()
model_path = "./models/"

# Import necessary modules and functions from translate_codebase.py

from translate_codebase import (
    # Environment variables and settings
    # Generic Helper functions
    clock_begin,
    clock_end,
    read_token_and_login,
    print_gpu_utilization,
    clear_cuda_cache,
    # Metrics Functions
    evaluate_translations_bertscore,
    evaluate_translations_rouge,
    evaluate_translations_bleu,
    evaluate_translations_meteor,
    # Common Data Preprocessing Functions
    filter_sentences,
    get_reduced_dataset,
    prepare_test_data,
    perform_translation_testing,
    # MBART related functions
    get_pretrained_mbart_large_50_many_to_many_mmt,
    preprocess_function_mbart,
    prepare_model_for_training_mbart,
    fine_tune_and_save_mbart,
    load_fine_tuned_model_mbart,
    translate_text_mbart,
    # MT5 related functions
    get_pretrained_mt5_small,
    config_mt5_small,
    encode_input_str_mt5_small,
    encode_target_str_mt5_small,
    process_translation_list_mt5_small,
    format_translation_data_mt5_small,
    transform_batch_mt5_small,
    fine_tune_and_save_model_mt5_small,
    eval_model_mt5_small,
    save_fine_tuned_model_mt5_small,
    load_fine_tuned_model_mt5_small,
    translate_text_mt5_small
)

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM

import torch

from team_files.akshay import getDiagnosisLlama
from team_files.aditya import load_unsloth_model_and_tokenizer_phi, generate_diagnosis_phi
from team_files.archit import load_model_mis, generate_text_mis, inference_mis
from team_files.ensemble import ensemble_responses

#from transformers import AutoModelForCausalLM, AutoTokenizer


def get_mbart_eng_hin_huggingface(force_download=True):

    model_path = "sathvikaithalkp456/mbart_fine_tuned_eng_hin"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path, force_download = force_download)
    model = MBartForConditionalGeneration.from_pretrained(model_path, force_download = force_download)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "hi_IN"
    return model, tokenizer


def get_mbart_hin_eng_huggingface(force_download=True):
    revision = "master"
    model_path = "sathvikaithalkp456/mbart_fine_tuned_hin_eng"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path, force_download = force_download, revision=revision)
    model = MBartForConditionalGeneration.from_pretrained(model_path, force_download = force_download, revision=revision)
    return model, tokenizer


def get_mt5_small_eng_hin_huggingface(force_download=True):

    revision = "master"
    model_path = 'sathvikaithalkp456/mbart_fine_tuned_eng_hin'
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download = force_download, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, force_download = force_download, revision=revision)
    model = model.cuda()
    return model, tokenizer


def get_mt5_small_hin_eng_huggingface(force_download=True):

    revision = "master"
    model_path = 'sathvikaithalkp456/mbart_fine_tuned_hin_eng'
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download = force_download, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, force_download = force_download, revision=revision)
    model = model.cuda()
    return model, tokenizer


def translate_text_generic(model, tokenizer, input_text, src_lang, tgt_lang, model_type="mbart"):
    """
    Translates a given text from the source language to the target language.

    Args:
        model: The translation model.
        tokenizer: The tokenizer for the model.
        input_text: The input text to be translated.
        src_lang: The source language code.
        tgt_lang: The target language code.
        model_type: The type of model ("mbart" or "mt5").

    Returns:
        str: The translated text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if model_type == "mbart":
        # Tokenize the input text with padding and truncation
        tokenizer.src_lang = src_lang
        encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate translation
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )

        # Decode the generated tokens
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    else:  # mt5
        # Encode the input text with padding and truncation
        input_ids = encode_input_str_mt5_small(
            text=input_text,
            target_lang=tgt_lang,
            tokenizer=tokenizer,
            seq_len=model.config.max_length,
            lang_token_map=LANG_TOKEN_MAPPING
        )
        input_ids = input_ids.unsqueeze(0).to(device)

        # Generate the translation
        output_ids = model.generate(input_ids, num_beams=5, max_length=model.config.max_length, early_stopping=True)

        # Decode the generated tokens
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    #print("Translated text:", translated_text)
    return translated_text


def translate_sentences_generic(input_lists, output_file, src_lang, tgt_lang, model_type="mbart", direction="eng_hin"):
    """
    Translates each sentence in the input lists to the target language and saves the translated sentences to a file.

    Args:
        input_lists (list): A list of lists, where each inner list contains sentences to be translated.
        output_file (str): The path to the output file where translated sentences will be saved.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        model_type (str): The type of model to use for translation ("mbart" or "mt5").
        direction (str): The direction of translation ("eng_hin" or "hin_eng").

    Returns:
        None
    """
    if model_type == "mbart":
        if direction == "eng_hin":
            model, tokenizer = get_mbart_eng_hin_huggingface()
        else:  # hin_eng
            model, tokenizer = get_mbart_hin_eng_huggingface()
    else:  # mt5
        if direction == "eng_hin":
            model, tokenizer = get_mt5_small_eng_hin_huggingface()
        else:  # hin_eng
            model, tokenizer = get_mt5_small_hin_eng_huggingface()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    translated_data = []

    for input_list in input_lists:
        translated_list = []
        for sentence in input_list:
            translated_sentence = translate_text_generic(model, tokenizer, sentence, src_lang, tgt_lang, model_type=model_type)
            translated_list.append(translated_sentence)
        translated_data.append(translated_list)

    # Save the translated sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

def parse_input(data):
    """
    Parses the input data, replaces 'Y' with 'Yes' and 'N' with 'No', and returns it as a list of formatted inputs.

    Args:
        data (str): The input data in JSON format.

    Returns:
        list: A list of formatted inputs.
    """
    parsed_data = json.loads(data)
    formatted_inputs = []

    for item in parsed_data:
        input_text = item['input']
        # Replace ' Y ' with ' Yes ' and ' N ' with ' No '
        input_text = input_text.replace(' Y ', ' Yes ').replace(' N ', ' No ')
        # Split the input text into individual lines based on delimiters
        lines = re.split(r'[;.]', input_text)
        # Strip leading and trailing whitespace from each line and filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        formatted_inputs.append(lines)

    return formatted_inputs

def generate_input():

    with open('sample_data.json', 'r') as file:
        data = json.load(file)

    data = json.dumps(data)

    parsed_inputs = parse_input(data)
    print(parsed_inputs)

    output_file = "translated_sentences_hindi.json"
    translate_sentences_generic(parsed_inputs, output_file, src_lang="en_XX", tgt_lang="hi_IN", model_type="mbart", direction="eng_hin")

    print("Input generated!!")

def get_inputs():
    with open('translated_sentences_hindi.json', 'r') as file:
        data = json.load(file)
    print(len(data))
    #print(data)
    return data


# Function to translate a sample input back to English
def translate_sample_to_english(sample_input):
    # Wrap the sample input in a list of lists to use with translate_sentences_generic
    input_lists = [sample_input]
    output_file = "temp_file.json"
    
    # Translate the sample input back to English
    translate_sentences_generic(input_lists, output_file, src_lang="hi_IN", tgt_lang="en_XX", model_type="mbart", direction="hin_eng")
    
    # Load the translated text from the output file
    with open(output_file, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)
    
    # Extract the translated text
    print("translated data: ", translated_data)
    #translated_text = translated_data
    #print("Translated text:", translated_text)
    return translated_data

# Function to translate a sample input back to English
def translate_sample_to_hindi(diagnosis,diagnosis_unsloth, diagnosis_mistral):
    # Wrap the sample input in a list of lists to use with translate_sentences_generic
    input_lists = [[diagnosis],[diagnosis_unsloth], [diagnosis_mistral]]
    output_file = "temp_file_2.json"
    
    # Translate the sample input back to English
    translate_sentences_generic(input_lists, output_file, src_lang="en_XX", tgt_lang="hi_IN", model_type="mbart", direction="eng_hin")
    
    # Load the translated text from the output file
    with open(output_file, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)
    
    # Extract the translated text
    print("translated data: ", translated_data)
    #translated_text = translated_data
    #print("Translated text:", translated_text)
    return translated_data


def process_translated_text_and_get_diagnosis_llama(translated_text):
    """
    Processes the translated text, joins the sentences, and gets the diagnosis from the Medical LLM (Llama).

    Args:
        translated_text (list): A list of translated sentences.

    Returns:
        str: The diagnosis from the Medical LLM (Llama).
    """
    # Flatten the list of lists into a single list of strings
    flattened_text = [sentence for sublist in translated_text for sentence in sublist]
    
    # Join the sentences in the translated text
    combined_text = ' '.join(flattened_text)
    print(f"Combined text: {combined_text}")

    # Pass the combined text to the Medical LLM and get the diagnosis
    diagnosis = getDiagnosisLlama(combined_text)
    return diagnosis




def process_translated_text_and_get_diagnosis_unsloth(translated_text):
    """
    Processes the translated text, joins the sentences, and gets the diagnosis from the Medical LLM (Unsloth).

    Args:
        translated_text (list): A list of translated sentences.

    Returns:
        str: The diagnosis from the Medical LLM (Unsloth).
    """
    # Flatten the list of lists into a single list of strings
    flattened_text = [sentence for sublist in translated_text for sentence in sublist]
    
    # Join the sentences in the translated text
    combined_text = ' '.join(flattened_text)
    print(f"Combined text: {combined_text}")

    # Load the model and tokenizer
    model_path = "Buddy1421/medical_diagnosis_phi_3-5"
    model, tokenizer = load_unsloth_model_and_tokenizer_phi(model_path, use_safetensors=True)

    # Pass the combined text to the Medical LLM and get the diagnosis
    diagnosis, diagnosis_json= generate_diagnosis_phi(model, tokenizer, combined_text)
    return diagnosis,diagnosis_json


def process_translated_text_and_get_diagnosis_mistral(translated_text):
    """
    Processes the translated text, joins the sentences, and gets the diagnosis from the Medical LLM (Mistral).

    Args:
        translated_text (list): A list of translated sentences.

    Returns:
        str: The diagnosis from the Medical LLM (Mistral).
    """
    # Flatten the list of lists into a single list of strings
    flattened_text = [sentence for sublist in translated_text for sentence in sublist]
    
    # Join the sentences in the translated text
    combined_text = ' '.join(flattened_text)
    print(f"Combined text: {combined_text}")

    # Load the model and tokenizer
    model, tokenizer = load_model_mis()

    # Pass the combined text to the Medical LLM and get the diagnosis
    full_response , diagnosis_json = inference_mis(combined_text, max_length=512)
    return full_response, diagnosis_json

def format_for_ensemble(mistral_json, llama_json, phi_json):
    outputs = {
        'mistral' :mistral_diagnosis_json,
        'llama': 
        'phi': {
            'most_likely': 'Anemia',
            'differential': ['HIV (initial infection)', 'Anemia', 'Pancreatic neoplasm', 'Chostochondritis']
        },
    }
    return outputs

def demo():
    print("starting demo")
    #print("generating inputs")
    #generate_input()
    print("getting inputs")
    inputs = get_inputs()
    seed = 6
    print("Choosing 1 random input : ",seed, " from 25 inputs")
    test_input = inputs[seed]
    print(test_input)
    print("Translating to English")
    translated_text = translate_sample_to_english(test_input)
    print("Calling MEDICAL LLM (Llama)")
    diagnosis = process_translated_text_and_get_diagnosis_llama(translated_text)
    print("Diagnosis:", diagnosis)
    print("Calling MEDICAL LLM (PHI)")
    diagnosis_unsloth, phi_json = process_translated_text_and_get_diagnosis_unsloth(translated_text)
    print("Diagnosis (Unsloth):", diagnosis_unsloth)
    print("Calling MEDICAL LLM (Mistral)")
    diagnosis_mistral, mistral_diagnosis_json  = process_translated_text_and_get_diagnosis_mistral(translated_text)
    print("Diagnosis (Mistral):", diagnosis_mistral)
    print("Translating to Hindi")
    translate_sample_to_hindi(diagnosis,diagnosis_unsloth, diagnosis_mistral)

    # Set up random responses for now, but expect these from the model functions
    llama_json = {'most_likely': 'XYZ','differential': ['HIV (initial infection)', 'Colitis', 'jaundice']}
    phi_json = {'most_likely': 'XYZ','differential': ['HIV (initial infection)', 'Colitis', 'jaundice']}

    model_outputs = format_for_ensemble(mistral_diagnosis_json, llama_json, phi_json)
    print("Final Response: ")
    print(ensemble_responses(model_outputs))


    


if __name__ == "__main__":
    demo()
