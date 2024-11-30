from unsloth import FastLanguageModel
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
    return extract_response_mis(text[0])

def inference_mis(text:str, max_length:int = 256):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    model,tokenizer = load_model_mis()
    return generate_text_mis(model,tokenizer,text,max_length,alpaca_prompt)


text = 'my head hurts'
inference_mis(text,256)
