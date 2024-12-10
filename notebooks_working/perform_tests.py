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



def dataset_desc():

    clock_begin("dataset_desc")
    dataset_name = "cfilt/iitb-english-hindi"
    small_data_set = get_reduced_dataset(dataset_name, train_size = 112000, val_size=16000, test_size=32000)
    clock_end("dataset_desc")
    print(small_data_set)
    print(small_data_set['train'])
    print(small_data_set['train'][0])

    def get_word_counts(dataset, lang):
        word_counter = Counter()
        for example in dataset:
            words = example['translation'][lang].split()
            word_counter.update(words)
        return word_counter

    def get_sentence_lengths(dataset, lang):
        lengths = []
        for example in dataset:
            lengths.append(len(example['translation'][lang].split()))
        return lengths

    stats = {}
    for split in ['train', 'validation', 'test']:
        dataset = small_data_set[split]
        stats[split] = {
            'num_sentences': len(dataset),
            'word_counts_en': get_word_counts(dataset, 'en'),
            'word_counts_hi': get_word_counts(dataset, 'hi'),
            'sentence_lengths_en': get_sentence_lengths(dataset, 'en'),
            'sentence_lengths_hi': get_sentence_lengths(dataset, 'hi')
        }

    for split, split_stats in stats.items():
        num_sentences = split_stats['num_sentences']
        word_counts_en = split_stats['word_counts_en']
        word_counts_hi = split_stats['word_counts_hi']
        sentence_lengths_en = split_stats['sentence_lengths_en']
        sentence_lengths_hi = split_stats['sentence_lengths_hi']

        print(f"Statistics for {split} set:")
        print(f"Number of sentences: {num_sentences}")
        print(f"Number of unique words in English: {len(word_counts_en)}")
        print(f"Number of unique words in Hindi: {len(word_counts_hi)}")
        print(f"Average sentence length in English: {np.mean(sentence_lengths_en):.2f} words")
        print(f"Average sentence length in Hindi: {np.mean(sentence_lengths_hi):.2f} words")
        print(f"Median sentence length in English: {np.median(sentence_lengths_en):.2f} words")
        print(f"Median sentence length in Hindi: {np.median(sentence_lengths_hi):.2f} words")
        print(f"Max sentence length in English: {max(sentence_lengths_en)} words")
        print(f"Max sentence length in Hindi: {max(sentence_lengths_hi)} words")
        print(f"Min sentence length in English: {min(sentence_lengths_en)} words")
        print(f"Min sentence length in Hindi: {min(sentence_lengths_hi)} words")
        print()


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
        formatted_inputs.extend(lines)

    return formatted_inputs

def test_example():
    # Example usage
    #data = '[{"instruction": "Provide Diagnosis", "input": "Patient age is 47, sex is M.  Antecedents: Have you had one or several flare ups of chronic obstructive pulmonary disease (COPD) in the past year? Y ; Do you smoke cigarettes? Y ; Do you have a chronic obstructive pulmonary disease (COPD)? Y ; Have you ever been diagnosed with gastroesophageal reflux? Y ; Do you work in agriculture? Y ; Do you work in the mining sector? Y ; Have you traveled out of the country in the last 4 weeks? N . Symptoms: Have you noticed a wheezing sound when you exhale? Y ; Are you experiencing shortness of breath or difficulty breathing in a significant way? Y ; Have you noticed a wheezing sound when you exhale? Y . ", "output": " Differential diagnosis is: Acute pulmonary edema, Anaphylaxis, Guillain-Barre syndrome, Atrial fibrillation, Myocarditis, Acute COPD exacerbation / infection, Pulmonary embolism, Acute dystonic reactions, Myasthenia gravis, Anemia, Scombroid food poisoning, PSVT, SLE, Possible NSTEMI / STEMI, Chagas and the Disease can be Acute COPD exacerbation / infection "}]'

    with open('sample_data.json', 'r') as file:
        data = json.load(file)

    data = json.dumps(data)

    parsed_inputs = parse_input(data)
    print(parsed_inputs)

    #model, tokenizer = load_fine_tuned_model_mt5_small(model_path + './mt5_small_fine_tuned_eng_hindi')

    model_1, tokenizer_1 = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_eng_hin")

    model_2, tokenizer_2 = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_hin_eng")

    temp = []

    for input_text in parsed_inputs:
        print(f"Input: {input_text}")
        #output_text = translate_text_mt5_small(input_text=input_text, model=model, tokenizer=tokenizer, target_lang='hi',lang_token_map=LANG_TOKEN_MAPPING, seq_len=model.config.max_length)
        #print(f"Output mt5: {output_text}")
        translated_text = translate_text_mbart(model_1, tokenizer_1, input_text, src_lang='en_XX', tgt_lang='hi_IN')
        temp.append(translated_text)
        #print(f"Output mbart: {translated_text}")

    print(temp)
    print("")
    print("END TO END FLOW")
    print("")

    outputs = []
    for t in temp:
        print(f"Input: {t}")
        translated_text = translate_text_mbart(model_2, tokenizer_2, t, src_lang='hi_IN', tgt_lang='en_XX')
        #print(f"Output mbart: {translated_text}")
        outputs.append(translated_text)

    print("\n\nFINAL OUTPUT\n\n")
    print(outputs)
    with open("./hindi_input", 'w') as file:
        file.write(temp)
    with open("./english_output", 'w') as file:
        file.write(outputs)
        

def test_example_2(data):
    model_1, tokenizer_1 = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_eng_hin")

    model_2, tokenizer_2 = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_hin_eng")

    print("")
    print("Eng to Hindi")
    print("")

    temp = []
    print(data)

    for input_text in data:
        print(f"Input: {input_text}")
        translated_text = translate_text_mbart(model_1, tokenizer_1, input_text, src_lang='en_XX', tgt_lang='hi_IN')
        temp.append(translated_text)

    print(temp)
    print("")
    print("Hindi to Eng")
    print("")

    outputs = []
    for t in temp:
        print(f"Input: {t}")
        translated_text = translate_text_mbart(model_2, tokenizer_2, t, src_lang='hi_IN', tgt_lang='en_XX')
        #print(f"Output mbart: {translated_text}")
        outputs.append(translated_text)
        
    print(outputs)
    print("")
    print("Eng to Hindi")
    print("")
    

    outputs_2 = []
    for input_text in outputs:
        print(f"Input: {input_text}")
        translated_text = translate_text_mbart(model_1, tokenizer_1, input_text, src_lang='en_XX', tgt_lang='hi_IN')
        outputs_2.append(translated_text)
    print(outputs_2)





def parse_complete_test_data(data):
    """
    Parses the complete test data, splits it based on questions and answers, and returns it as a list of sentences.

    Args:
        data (str): The complete test data as a string.

    Returns:
        list: A list of sentences from the 'Symptoms' and 'Differential diagnosis' sections.
    """
    # Split the data into sections based on the "Symptoms:" and "Differential diagnosis is:" delimiters
    sections = re.split(r'Symptoms:|Differential diagnosis is:', data)
    
    # Extract the input section
    input_section = sections[1].strip()
    
    # Replace ' Y ' with ' Yes ' and ' N ' with ' No '
    input_section = input_section.replace(' Y ', ' Yes ').replace(' N ', ' No ')
    
    # Split the input section into individual lines based on delimiters
    lines = re.split(r'[;.]', input_section)
    
    # Strip leading and trailing whitespace from each line and filter out empty lines
    symptoms = [line.strip() for line in lines if line.strip()]
    
    # Extract the differential diagnosis section
    differential_diagnosis = sections[2].strip()
    
    # Combine symptoms and differential diagnosis into a single list
    formatted_inputs = symptoms + [differential_diagnosis]
    
    return formatted_inputs

    

    #model, tokenizer = load_fine_tuned_model_mt5_small(model_path + './mt5_small_fine_tuned_hindi_eng')


def translate_eng_hin_custom(input_text):
    model, tokenizer = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_eng_hin")
    print(f"Input: {input_text}")
    translated_text = translate_text_mbart(model, tokenizer, input_text, src_lang='en_XX', tgt_lang='hi_IN')
    print(f"Output: {translated_text}")

def translate_hin_eng_custom(input_text):
    model, tokenizer = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_hin_eng")
    print(f"Input: {input_text}")
    translated_text = translate_text_mbart(model, tokenizer, input_text, src_lang='hi_IN', tgt_lang='en_XX')
    print(f"Output: {translated_text}")


def main_2():
    """
    Main function to orchestrate fine-tuning and testing.
    
    print("Running the dataset_desc function")
    #dataset_desc()
    print("Finished the dataset_desc function")
    print("Running the test_example function")
    #test_example()
    print("Finished the test_example function")
    print("Running the parse_complete_test_data function")
    with open("complete_test_data", 'r') as file:
        data = file.read()
    formatted_inputs = parse_complete_test_data(data)
    print("HELLO")
    #print(formatted_inputs)
    print("Finished the parse_complete_test_data function")
    test_example_2(formatted_inputs)
    """

    input_text = "Proper Nouns are difficult to translate : Sathvik Karatattu Padmanabha"
    translate_eng_hin_custom(input_text)
    print("\nGoing other way around\n")
    input_text = "व्यक्तिवाचक संज्ञाओं का अनुवाद करना कठिन है: साथ्विक करतत्तु पद्मनाभ"
    translate_hin_eng_custom(input_text) 



if __name__ == "__main__":
    main_2()