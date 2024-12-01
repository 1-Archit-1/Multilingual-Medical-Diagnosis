'''
def load_unsloth_model_and_tokenizer(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer
'''
def load_unsloth_model_and_tokenizer_phi(model_path, use_safetensors=False):

    if use_safetensors:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
            use_safetensors=True
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
            use_safetensors=False,
            device_map="auto"
        )
    return model, tokenizer

def generate_diagnosis_phi(model, tokenizer, user_input):
    FastLanguageModel.for_inference(model)
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction: {}
    ### Input: {}
    ### Response: {}"""
    
    formatted_input = alpaca_prompt.format(
        "Perform Diagnosis in the format: Differential Diagnosis is:... a,b,c... and Disease can be X",
        user_input,
        ""
    )
    
    inputs = tokenizer(
        [formatted_input],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    if "### Response:" in decoded_output:
        response = decoded_output.split("### Response:")[1].strip()
    else:
        response = decoded_output.strip()
    
    return response,format_diagnosis_output(response)

def format_diagnosis_output(output_text):
    """Convert model output text to structured JSON format"""
    try:
        # Extract disease and differential diagnoses using regex
        disease_match = re.search(r"Disease can be (.*?)(?:\s|$)", output_text)
        diff_match = re.search(r"Differential Diagnosis is: (.*?) and", output_text)
        
        most_likely = disease_match.group(1).strip() if disease_match else None
        differential = []
        
        if diff_match:
            differential = [d.strip() for d in diff_match.group(1).split(',')]
        
        # Create formatted output
        formatted_output = {
            'most_likely': most_likely,
            'differential': differential
        }
        
        return formatted_output
    except Exception as e:
        return {
            'most_likely': None,
            'differential': []
        }

def main():
    model_path = "Buddy1421/medical-diagnosis-phi"
    model, tokenizer = load_unsloth_model_and_tokenizer_phi(model_path)
    
    user_input = input("Enter the patient symptoms and history: ")
    diagnosis, diagnosis_json = generate_diagnosis_phi(model, tokenizer, user_input)
    print("Diagnosis:", diagnosis)

if __name__ == "__main__":
    main()
