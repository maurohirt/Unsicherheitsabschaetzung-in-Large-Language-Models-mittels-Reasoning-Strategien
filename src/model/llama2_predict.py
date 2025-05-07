import torch
import time
import os
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

HF_NAMES = {
    'llama3-1_8B': 'meta-llama/Llama-3.1-8B',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf'
}

def model_init(args):
    model_path = args.model_path # Replace to your model path
    device = torch.device("cuda:0")
    
    # Determine the datatype based on hardware capabilities
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[model_init] Using dtype: {dtype}")
    
    # Check if model_path is one of the predefined keys
    if model_path in HF_NAMES:
        # Try to find a symlink in the HF cache
        symlink_path = f"/root/.cache/huggingface/symlinks/{model_path}"
        if os.path.exists(symlink_path):
            print(f"Using symlink at {symlink_path}")
            actual_path = symlink_path
        else:
            # Fallback to the HF Hub name
            actual_path = HF_NAMES[model_path]
    else:
        # Use the provided path directly
        actual_path = model_path
    
    print(f"Loading model from: {actual_path}")
    
    if "llama" in model_path.lower():
        # Use local_files_only=True for offline mode
        model = LlamaForCausalLM.from_pretrained(
            actual_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(actual_path, local_files_only=True)
    elif "mistral" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            actual_path, 
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(actual_path, local_files_only=True)
    else:
        raise ValueError(f"Invalid Model Path: {model_path}")
    
    return model, tokenizer, model.device

def predict(args, prompt, model, tokenizer):
    # Use the model's device directly
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens = args.max_length_cot,
        temperature=args.temperature, 
        pad_token_id=tokenizer.eos_token_id)
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res


def tokenize(prompt, tokenizer, model_name, device, tokenizer_args=None):
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to(device)
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to(device)
    return model_input


def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None):

    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = None

    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  **(additional_kwargs or {}))

    return model_output


def generate_model_answer(args, prompt, model, tokenizer, device, do_sample=False, output_scores=False,
                           temperature=1.0,
                           top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):

    model_path = args.model_path 
    model_name = model_path  # Use the direct model path instead of HF_NAMES lookup

    model_input = tokenize(prompt, tokenizer, model_name, device)

    with torch.no_grad():
        model_output = generate(model_input, model, model_name, do_sample, output_scores, max_new_tokens=max_new_tokens,
                                top_p=top_p, temperature=temperature, stop_token_id=stop_token_id, tokenizer=tokenizer)

    answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])
    output_ids = model_output['sequences'][0][len(model_input[0]):].cpu()
    if output_scores:
        scores = torch.concatenate(model_output['scores']).cpu()  # shape = (new_tokens, len(vocab))
        input_output_ids = model_output['sequences'][0].cpu()
        return answer, input_output_ids, scores, output_ids
    else:
        return answer, output_ids