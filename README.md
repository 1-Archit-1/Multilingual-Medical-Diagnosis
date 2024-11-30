# Medical Diagnosis and Hin-Eng Bi-directional Langauge translation Project

This repository contains code and resources for training and fine-tuning machine translation models using the Hugging Face Transformers library. The primary focus is on translating between English and Hindi using various models such as MBart and mT5. The medical diagnosis part uses mistral, phi and llama models.

## Repository Structure
. ├── pycache/ ├── .gitignore ├── .ipynb_checkpoints/ ├── anaconda ├── complete_end_to_end_flow_2.log ├── complete_test_data ├── conda/ │ ├── .condarc │ ├── LLM_test/ │ ├── LLM_TEST_2/ ├── environment.yml ├── final_demo.py ├── getGPU ├── getGPU2 ├── getGPU3 ├── hf_token ├── logger_tests.log ├── logger.log ├── logs/ ├── models/ ├── models_backups/ ├── nltk_data/ ├── notebooks_working/ │ ├── Test3_Eng_Hindi.ipynb │ ├── Test4_Hindi_Eng.ipynb │ ├── Test5_mt5-small_Eng_Hindi.ipynb │ ├── Test6_mt5-small_Hindi_Eng.ipynb │ ├── mt5_TEST.ipynb ├── old/ ├── perform_tests.py ├── results/ ├── sample_data.json ├── sample_validate_data_500.json ├── team_files/ ├── temp_file_2.json ├── temp_file.json ├── translate_codebase.py ├── translate_exec.py ├── translated_sentences_hindi.json

## Notebooks
- `some old draft notebooks`

## Team files
- `Files containing code for medical inference`

## Scripts

- `final_demo.py: Script for running the final demo.`
- `perform_tests.py: Script for performing various tests on the translation models `
- `translate_codebase.py: Contains the translation code functions.`
- `translate_exec.py: Script for training the translation part. Uses functions from translate_codebase.py.`

## Data
- `sample_data.json: Sample data for testing.`
- `sample_validate_data_500.json: 500 samples dataset.`
- `translated_sentences_hindi.json: Sample data translated`

## Logs
- `logger.log: Sample full output log file.`
- `temp*.json : Intermediate outputs`


## Environment Setup
To set up the environment, use the environment.yml file:
conda env create -f environment.yml conda activate <environment_name>


## References
- `Multilingual NMT with mT5: https://github.com/ejmejm/multilingual-nmt-mt5/blob/main/nmt_full_version.ipynb`
- `LING-380 Final Project: https://github.com/zhangmegan/LING-380-Final-Project/tree/9d9bb07fcfcf05fa45942276de5af1265b4f3701/japanese_turkish_finetuning.py`
- `Unsloth : https://github.com/unslothai/unsloth`

## License
This project is licensed under the MIT License.
