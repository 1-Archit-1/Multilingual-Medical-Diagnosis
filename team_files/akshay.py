import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def load_model(path="asadhu8/llama_3.2_1b_ddx_plus_medical"):
    # Load the fine-tuned model and tokenizer
    model_path = path
    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded successfully!")

    # Move model to the appropriate device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def unload_model(model, tokenizer):
    """Unload the model and clear memory."""
    print("Unloading the model to save memory...")
    del model  # Delete the model object
    del tokenizer  # Delete the tokenizer object
    torch.cuda.empty_cache()  # Clear GPU memory
    print("Model unloaded successfully!")

# Function to test prompts
def generate_response(model, tokenizer, prompt, max_length=1024):
    """Generate a response for a given prompt using the fine-tuned model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,  # Use beam search for more coherent responses
            early_stopping=True,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def format_output_for_voting(output_text):
    """Convert model output text to structured JSON format"""
    try:
        # Extract disease and differential diagnoses using regex
        disease_match = output_text.split("the Disease can be")[-1].strip()
        differential_diagnosis = output_text.split("Differential diagnosis is:")[1].split("and the Disease can be")[0].strip()
        differential = [diagnosis.strip() for diagnosis in differential_diagnosis.split(",")]

        most_likely = disease_match.strip() if disease_match else None


        # Create formatted output
        formatted_output = [{
            'most_likely': most_likely,
            'differential': differential
        }]

        return formatted_output
    except Exception as e:
        return [{
            'most_likely': None,
            'differential': []
        }]


def getDiagnosisLlama(prompt, path="asadhu8/llama_3.2_1b_ddx_plus_medical_v2"):
    model, tokenizer = load_model(path)
    response = generate_response(model, tokenizer, prompt)
    output_match = re.search(r"Output:\s*(.*)", response, re.DOTALL)
    print(output_match.group(1))
    unload_model(model, tokenizer)  # Unload the model to free memory
    return format_output_for_voting(output_match.group(1))


if __name__ == '__main__':
    prompt = "Instruction: Provide Diagnosis Patient age is 65, sex is M.  Antecedents: Do you have a known issue with one of your heart valves? Y ; Do you have severe Chronic Obstructive Pulmonary Disease (COPD)? Y ; Do you have diabetes? Y ; Do you have high blood pressure or do you take medications to treat high blood pressure? Y ; Do you have a known heart defect? Y ; Have you traveled out of the country in the last 4 weeks? N . Symptoms: Do you feel slightly dizzy or lightheaded? Y ; Are you experiencing shortness of breath or difficulty breathing in a significant way? Y ; Do you feel slightly dizzy or lightheaded? Y ; Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations? Y ; Do you feel your heart is beating very irregularly or in a disorganized pattern? Y ; Do you have symptoms that are increased with physical exertion but alleviated with rest? Y ."
    print(getDiagnosisLlama(prompt))