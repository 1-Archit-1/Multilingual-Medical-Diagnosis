from unsloth import FastLanguageModel
import re 
def load_model_mis():
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = 'rhaegar1O1/mistral-ddx-v2',
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = True,
        )
    return model,tokenizer

def extract_response_mis(text):
    # Split the input by the "Response:" keyword
    response_key = "### Response:"
    if response_key in text:
        response = text.split(response_key, 1)[-1].strip()
        return response[:-10]
    else:
        return "No response found."

def extract_diagnosis(input_string):
    match = re.search(r'\*\*(.*?)\*\*', input_string)
    most_likely = match.group(1) if match else '' # Return the matched disease
    
    differential_match = re.search(r'Differential Diagnosis is:\s*(.*?),?\s*and the most likely is', input_string)
    differential_diseases = []
    if differential_match:
        differential_diseases = [d.strip() for d in differential_match.group(1).split(',')]
    
    return most_likely, differential_diseases

def generate_text_mis(model,tokenizer, text, max_length,alpaca_prompt):
    FastLanguageModel.for_inference(model)
    # Generate predictions
    instruction = """Perform Diagnosis"""
    inputs = tokenizer(
    [
        alpaca_prompt.format(instruction, text, "", )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 200, use_cache = True)
    text= tokenizer.batch_decode(outputs)
    full_response = extract_response_mis(text[0])
    most_likely,differential = extract_diagnosis(full_response)
    diagnosis_json = {
            'most_likely': most_likely,
            'differential': differential
        }
    return full_response, diagnosis_json 

def inference_mis(text:str, max_length:int = 256):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    model,tokenizer = load_model_mis()
    full_response, diagnosis_json = generate_text_mis(model,tokenizer,text,max_length,alpaca_prompt)
    return full_response, diagnosis_json

text = 'my head hurts'
inference_mis(text,256)
