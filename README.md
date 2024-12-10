# Medical Diagnosis and Hin-Eng Bi-directional Language translation Project

This repository contains code and resources for training and fine-tuning machine translation models using the Hugging Face Transformers library. The primary focus is on translating between English and Hindi using various models such as MBart and mT5. The medical diagnosis part uses mistral, phi and llama models.

## Team Members
```
Sathvik Karatattu Padmanabha
Nithin Singhal
Aditya Kumar
Akshay Sadhu
Archit Sengupta
```

## Hugging Face Models
```
- sathvikaithalkp456/mt5_small_fine_tuned_eng_hindi
- sathvikaithalkp456/mt5_small_fine_tuned_hindi_eng
- sathvikaithalkp456/mbart_fine_tuned_eng_hin
- sathvikaithalkp456/mbart_fine_tuned_hin_eng
- Buddy1421/medical_diagnosis_phi_3-5
- asadhu8/llama_3.2_1b_ddx_plus_medical
- rhaegar1O1/mistral-ddx-v2
```
## Repository Structure
```
.
├── .gitignore                          # Git ignore file to exclude specific files and directories
├── README.md                           # Repository documentation
├── environment.yml                     # Conda environment configuration file
├── translate_codebase.py               # Core translation code functions
├── translate_exec.py                   # Execution script for running translation tasks
├── final_demo.py                       # Script for running the final demonstration
├── final_demo_notebook_version.ipynb   # Notebook version of the final demonstration
├── notebooks_working/                  # Directory for active Jupyter notebooks
│   ├── *                               # Working draft notebooks (for various tasks) 
├── team_files/                         # Directory for team-member files
│   ├── aditya.py                       # Phi usage example
│   ├── akshay.py                       # Llama usage example
│   ├── archit.py                       # Mistral usage example
│   └── ensemble.py                     # Ensemble logic 
├── medium_cases/                       # Directory for medium case execution
│   ├── final_demo_notebook_version_medium.ipynb                  # Notebook version for medium case execution
│   ├── *.json                          # Data files for medium case testing
│   └── logger*.log                     # Log files for medium case testing
├── translation_examples_and_scores/    # Directory for translation examples and their scores
│   ├── mbart_eng_hind_logs.log         # Translation examples for mbart English -> Hindi direction with scores
│   ├── mbart_hindi_eng_logs.log        # Translation examples for mbart Hindi -> Eng direction with scores
│   └── mt_small_logs.log               # Translation examples for mt5-small for both directions with scores
├── extra_stuff_do_not_delete/          # Directory for extra important files that should not be deleted
│   ├── sample_validate_data_500.json   # 500 full end-to-end medical examples (english)
│   └── anaconda                        # for PACE usage
├── hard_cases/                         # Directory for hard case files
│   ├── mis.txt                         # Mistral hard case examples
│   ├── phi.txt                         # Phi hard case examples
│   └── sample_hard_25.json             # Data for hard cases
├── sample_data.json                    # Sample dataset in JSON format (english)
├── translated_sentences_hindi.json     # JSON file containing translated sentences in Hindi (input file)

```
## Description

### Core Components

- **translate_codebase.py**  
  Contains the core functions responsible for handling the translation logic, including preprocessing, model loading, and translation execution for both mt5 and mbart models

- **translate_exec.py**  
  Serves as the main execution script to run translation tasks. It orchestrates the workflow by utilizing functions from `translate_codebase.py`.

### Team Files

- **team_files/**  
  Directory for team-members' scripts and resources.
  
  - **aditya.py**  
    Example usage of the Phi model.
    
  - **akshay.py**  
    Example usage of the Llama model.
    
  - **archit.py**  
    Example usage of the Mistral model.
    
  - **ensemble.py**  
    Logic for ensembling outputs from multiple models to improve diagnostic accuracy.

### Demonstrations

- **final_demo.py**  
  Script for running the final demonstration of the translation and medical diagnosis capabilities of the project.

- **final_demo_notebook_version.ipynb**  
  Interactive Jupyter Notebook version of the final demonstration, allowing for a more hands-on walkthrough of the project's functionalities.

### Notebooks

- **notebooks_working/**  
  Directory containing active Jupyter notebooks used for various development and testing tasks.
  
  - **\***  
    Placeholder for working draft notebooks addressing different aspects of the project.


### Extra Resources

- **extra_stuff_do_not_delete/**  
  Contains essential files that should not be deleted.
  
  - **sample_validate_data_500.json**  
    A dataset of 500 complete medical examples in English for validation purposes.
    
  - **anaconda/**  
    Configuration and resources for PACE usage.


### Sample and Translated Data

- **sample_data.json**  
  A sample dataset in English used for testing the translation models.

- **translated_sentences_hindi.json**  
  Contains translated sentences in Hindi, serving as input data for further processing.


## Results

### Translation Examples and Scores

- **translation_examples_and_scores/**  
  Stores examples of translations along with their evaluation scores. (Please download the files to view)
  
  - **mbart_eng_hind_logs.log**  
    Translation examples and scores for MBart model translating English to Hindi (Please download the files to view).
    
  - **mbart_hindi_eng_logs.log**  
    Translation examples and scores for MBart model translating Hindi to English (Please download the files to view).
    
  - **mt_small_logs.log**  
    Translation examples and scores for mT5-small model in both translation directions.

### Medium Cases

- **medium_cases/**  
  Code, logs and data for testing medium complexity cases within the project.
  
  - **final_demo_notebook_version_medium.ipynb**  
    Notebook for executing medium complexity demonstrations.
    
  - **\*.json**  
    Data files used for testing medium complexity scenarios.
    
  - **logger*.log**  
    Log files capturing the execution details of medium case tests.

### Hard Cases

- **hard_cases/**  
  Code, logs and data for testing medium complexity cases within the project. Contains challenging cases to test the robustness of the translation and diagnostic models.
  
  - **mis.txt**  
    Hard case examples for the Mistral model.
    
  - **phi.txt**  
    Hard case examples for the Phi model.
    
  - **sample_hard_25.json**  
    Data for 25 hard case scenarios.

  - **xx.ipynb**  
    Notebook for executing medium complexity demonstrations.



## Usage

### Environment Setup

To set up the development environment, use the provided `environment.yml` file with Conda:

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

Replace <environment_name> with your desired environment name.

### Running the Final Demonstration

Execute the final demo script to showcase the project's capabilities

```bash
python final_demo.py
```

Alternatively, use the interactive notebook:

Open final_demo_notebook_version.ipynb in Jupyter Notebook or VS Code.
Run each cell in the notebook. (Self explanatory)

Note: The logs are stored in logger.log or logger_tests.log files by default if nothing appears on the terminal.

### Running individual models

For translation for both mt5 and mbart models, you can edit the main function in the translate_exec.py file to choose the models, and run the translate_exec.py file. Set SHOULD_TRAIN = True for fine-tuning from scratch. For mbart, you can open the notebooks_working/perform_tests_notebook_version.ipynb file for a very simple code template to translate.

You can also create notebooks and use code template functions from translate_codebase.py for more rigorous testing. 

Alternatively, several notebooks inside notebooks_working directory have .py and .ipynb files showcasing how to use the individual models. 

For Medical LLM examples, refer the team_files directory for each of the 3 models. More notebooks are available in the notebooks_working directory.


## References
- `https://huggingface.co/transformers/`
- `Multilingual NMT with mT5: https://github.com/ejmejm/multilingual-nmt-mt5/blob/main/nmt_full_version.ipynb`
- `LING-380 Final Project: https://github.com/zhangmegan/LING-380-Final-Project/tree/9d9bb07fcfcf05fa45942276de5af1265b4f3701/japanese_turkish_finetuning.py`
- `Unsloth : https://github.com/unslothai/unsloth`

## License
This project is licensed under the MIT License.
