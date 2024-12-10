SHOULD_TRAIN = False
#SHOULD_TRAIN = True

import os
import time
import sys



# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"
os.environ["HF_HOME"] = "~/scratch/hf-cache"
token=""
print(os.environ['WANDB_DISABLED'])  # Should output "true"
print(os.environ['HF_HOME'])  # Should output "~/scratch/hf-cache"

output_file = open('logger_exec.log', 'w')
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


def mbart_eng_hindi():

    global model_path
    global last_print_time

    api = read_token_and_login('hf_token')

    original_tokenizer, original_model = get_pretrained_mbart_large_50_many_to_many_mmt()

    clock_begin("data_preparation_1")
    dataset_name = "cfilt/iitb-english-hindi"
    small_data_set = get_reduced_dataset(dataset_name, train_size = 112000, val_size=16000, test_size=32000)
    clock_end("data_preparation_1")


    clock_begin("translation_testing_untuned_1")
    test_data, tokenized_test_data = prepare_test_data(small_data_set, original_tokenizer, num_examples=100, src_lang='en')
    translated_test_data_untuned = perform_translation_testing(original_model, original_tokenizer, test_data, tokenized_test_data, src_lang="en_XX", tgt_lang="hi_IN", model_type="mbart")
    evaluate_translations_bertscore(test_data, translated_test_data_untuned, src_lang='en', tgt_lang='hi', bert_lang='hi')
    evaluate_translations_bleu(test_data, translated_test_data_untuned, original_tokenizer, src_lang='en', tgt_lang='hi')
    evaluate_translations_rouge(test_data, translated_test_data_untuned, original_tokenizer, src_lang='en', tgt_lang='hi')
    evaluate_translations_meteor(test_data, translated_test_data_untuned, original_tokenizer, src_lang='en', tgt_lang='hi')
    clock_end("translation_testing_untuned_1")

    
    last_print_time = time.time()

    tokenized_datasets = small_data_set.map(lambda examples: preprocess_function_mbart(examples, original_tokenizer, src_lang='en', tgt_lang='hi'), batched=True, remove_columns=["translation"])

    
    clock_begin("fine_tuning_1")
    if SHOULD_TRAIN == True:
        trainer = prepare_model_for_training_mbart(original_model, original_tokenizer, tokenized_datasets)

        fine_tune_and_save_mbart(trainer, original_model, original_tokenizer, output_dir=model_path+"./mbart_fine_tuned_eng_hin")
    clock_end("fine_tuning_1")
    

    model, tokenizer = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_eng_hin")

    input_text = "Stop in the name of the law"
    translated_text = translate_text_mbart(model, tokenizer, input_text, src_lang='en_XX', tgt_lang='hi_IN')


    clock_begin("translation_testing_finetuned_1")
    test_data, tokenized_test_data = prepare_test_data(small_data_set, tokenizer, num_examples=100, src_lang='en')

    translated_test_data = perform_translation_testing(model, tokenizer, test_data, tokenized_test_data, src_lang="en_XX", tgt_lang="hi_IN", model_type="mbart")

    evaluate_translations_bertscore(test_data, translated_test_data, src_lang='en', tgt_lang='hi', bert_lang='hi')
    evaluate_translations_bleu(test_data, translated_test_data, tokenizer, src_lang='en', tgt_lang='hi')
    evaluate_translations_rouge(test_data, translated_test_data, tokenizer, src_lang='en', tgt_lang
    ='hi')
    evaluate_translations_meteor(test_data, translated_test_data, tokenizer, src_lang='en', tgt_lang='hi')
    clock_end("translation_testing_finetuned_1")



def mbart_hindi_eng():

    global model_path
    global last_print_time

    api = read_token_and_login('hf_token')

    original_tokenizer, original_model = get_pretrained_mbart_large_50_many_to_many_mmt()

    clock_begin("data_preparation_2")
    dataset_name = "cfilt/iitb-english-hindi"
    small_data_set = get_reduced_dataset(dataset_name, train_size = 112000, val_size=16000, test_size=32000)
    clock_end("data_preparation_2")

    clock_begin("translation_testing_untuned_2")
    test_data, tokenized_test_data = prepare_test_data(small_data_set, original_tokenizer, num_examples=100, src_lang='hi')
    translated_test_data_untuned = perform_translation_testing(original_model, original_tokenizer, test_data, tokenized_test_data, src_lang="hi_IN", tgt_lang="en_XX", model_type="mbart")
    evaluate_translations_bertscore(test_data, translated_test_data_untuned, src_lang='hi', tgt_lang='en', bert_lang='en')
    evaluate_translations_bleu(test_data, translated_test_data_untuned, original_tokenizer, src_lang='hi', tgt_lang='en')
    evaluate_translations_rouge(test_data, translated_test_data_untuned, original_tokenizer, src_lang='hi', tgt_lang='en')
    evaluate_translations_meteor(test_data, translated_test_data_untuned, original_tokenizer, src_lang='hi', tgt_lang='en')
    clock_end("translation_testing_untuned_2")

    last_print_time = time.time()
    tokenized_datasets = small_data_set.map(lambda examples: preprocess_function_mbart(examples, original_tokenizer, src_lang='hi', tgt_lang='en'), batched=True, remove_columns=["translation"])

    clock_begin("fine_tuning_2")
    if SHOULD_TRAIN == True:
        trainer = prepare_model_for_training_mbart(original_model, original_tokenizer, tokenized_datasets)
        fine_tune_and_save_mbart(trainer, original_model, original_tokenizer, output_dir=model_path+"./mbart_fine_tuned_hin_eng")
    clock_end("fine_tuning_2")

    model, tokenizer = load_fine_tuned_model_mbart(model_path+"./mbart_fine_tuned_hin_eng")

    input_text = "कानून के नाम पर रुकें"
    translated_text = translate_text_mbart(model, tokenizer, input_text, src_lang='hi_IN', tgt_lang='en_XX')

    clock_begin("translation_testing_finetuned_2")
    test_data, tokenized_test_data = prepare_test_data(small_data_set, tokenizer, num_examples=100, src_lang='hi')
    translated_test_data = perform_translation_testing(model, tokenizer, test_data, tokenized_test_data, src_lang="hi_IN", tgt_lang="en_XX", model_type="mbart")
    evaluate_translations_bertscore(test_data, translated_test_data, src_lang='hi', tgt_lang='en', bert_lang='en')
    evaluate_translations_bleu(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en')
    evaluate_translations_rouge(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang
    ='en')
    evaluate_translations_meteor(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en')
    clock_end("translation_testing_finetuned_2")


def mt5_eng_hindi():
    global model_path
    global LANG_TOKEN_MAPPING

    clock_begin("data_preparation_3")
    dataset_name = "cfilt/iitb-english-hindi"
    small_data_set = get_reduced_dataset(dataset_name, train_size = 112000, val_size=16000, test_size=32000)
    clock_end("data_preparation_3")

    model, tokenizer = get_pretrained_mt5_small()
    sorted_vocab = config_mt5_small(tokenizer, model, LANG_TOKEN_MAPPING)

    
    clock_begin("fine_tuning_3")
    if SHOULD_TRAIN == True:
        fine_tune_and_save_model_mt5_small(model, tokenizer, small_data_set['train'], small_data_set['validation'], LANG_TOKEN_MAPPING, model_path + './mt5_small_fine_tuned_eng_hindi', src_lang='en', tgt_lang='hi')

        #Not neeeded
        save_fine_tuned_model_mt5_small(model, tokenizer, model_path + './mt5_small_fine_tuned_eng_hindi')
    clock_end("fine_tuning_3")
    

    model, tokenizer = load_fine_tuned_model_mt5_small(model_path + './mt5_small_fine_tuned_eng_hindi')

    test_sentence = "Stop in the name of the law"
    print('Raw input text:', test_sentence)

    translated_text = translate_text_mt5_small(
        input_text=test_sentence,
        model=model,
        tokenizer=tokenizer,
        target_lang='hi',
        lang_token_map=LANG_TOKEN_MAPPING,
        seq_len=model.config.max_length
    )
    print('Translated text:', translated_text)

    clock_begin("translation_testing_finetuned_3")
    test_data, tokenized_test_data = prepare_test_data(small_data_set, tokenizer, num_examples=100, src_lang='en')

    translated_test_data = perform_translation_testing(model, tokenizer, test_data, tokenized_test_data, src_lang="en", tgt_lang="hi", model_type="mt5_small")

    evaluate_translations_bertscore(test_data, translated_test_data, src_lang='en', tgt_lang='hi', bert_lang='hi')
    evaluate_translations_bleu(test_data, translated_test_data, tokenizer, src_lang='en', tgt_lang='hi')
    evaluate_translations_rouge(test_data, translated_test_data, tokenizer, src_lang='en', tgt_lang='hi')
    evaluate_translations_meteor(test_data, translated_test_data, tokenizer, src_lang='en', tgt_lang='hi')
    clock_end("translation_testing_finetuned_3")


def mt5_hindi_eng():

    global model_path
    global LANG_TOKEN_MAPPING

    clock_begin("data_preparation_4")
    dataset_name = "cfilt/iitb-english-hindi"
    small_data_set = get_reduced_dataset(dataset_name, train_size = 112000, val_size=16000, test_size=32000)
    clock_end("data_preparation_4")

    model, tokenizer = get_pretrained_mt5_small()
    sorted_vocab = config_mt5_small(tokenizer, model, LANG_TOKEN_MAPPING)

    clock_begin("fine_tuning_4")
    if SHOULD_TRAIN == True:
        fine_tune_and_save_model_mt5_small(model, tokenizer, small_data_set['train'], small_data_set['validation'], LANG_TOKEN_MAPPING, model_path + './mt5_small_fine_tuned_hindi_eng', src_lang='hi', tgt_lang='en')

        #Not neeeded
        save_fine_tuned_model_mt5_small(model, tokenizer, model_path + './mt5_small_fine_tuned_hindi_eng')
    clock_end("fine_tuning_4")

    model, tokenizer = load_fine_tuned_model_mt5_small(model_path + './mt5_small_fine_tuned_hindi_eng')

    test_sentence = "कानून के नाम पर रुकें"
    print('Raw input text:', test_sentence)
    
    translated_text = translate_text_mt5_small(
        input_text=test_sentence,
        model=model,
        tokenizer=tokenizer,
        target_lang='en',
        lang_token_map=LANG_TOKEN_MAPPING,
        seq_len=model.config.max_length
    )
    print('Translated text:', translated_text)

    clock_begin("translation_testing_finetuned_4")
    test_data, tokenized_test_data = prepare_test_data(small_data_set, tokenizer, num_examples=100, src_lang='hi')

    translated_test_data = perform_translation_testing(model, tokenizer, test_data, tokenized_test_data, src_lang="hi", tgt_lang="en", model_type="mt5_small")

    evaluate_translations_bertscore(test_data, translated_test_data, src_lang='hi', tgt_lang='en', bert_lang='en')
    evaluate_translations_bleu(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en')
    evaluate_translations_rouge(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en')
    evaluate_translations_meteor(test_data, translated_test_data, tokenizer, src_lang='hi', tgt_lang='en')
    clock_end("translation_testing_finetuned_4")



def main():
    """
    Main function to orchestrate fine-tuning and testing.
    """
    print("Running the mbart_eng_hindi() function")
    #mbart_eng_hindi()
    print("Running the mbart_hindi_eng() function")
    #mbart_hindi_eng()
    print("Running the mt5_eng_hindi() function")
    mt5_eng_hindi()
    print("Running the mt5_hindi_eng() function")
    #mt5_hindi_eng()



if __name__ == "__main__":
    main()