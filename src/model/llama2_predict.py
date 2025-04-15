import torch
import time
import os
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

import os

# Use relative path with os.path to work in both local and Slurm environments
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
HF_NAMES = {
    'llama3-1_8B': os.path.join(ROOT_DIR, 'models/Llama-3.1-8B'),
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf'
}

def model_init(args):
    model_path = args.model_path
    device = torch.device("cuda:0")
    
    # Get the actual model path from HF_NAMES or use direct path
    actual_model_path = HF_NAMES.get(model_path, model_path)
    
    print(f"Loading model from: {actual_model_path}")
    
    # Check if the path is a local directory
    is_local_path = actual_model_path.startswith('/') or os.path.exists(actual_model_path)
    
    # Parameters for local vs remote loading
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    
    tokenizer_kwargs = {}
    
    # Add local_files_only flag only for local paths
    if is_local_path:
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True
    
    try:
        # For local paths, use standard HF loading but with local_files_only=True
        if "llama" in model_path.lower():
            print(f"Loading LLaMA model with kwargs: {model_kwargs}")
            model = LlamaForCausalLM.from_pretrained(
                actual_model_path,
                **model_kwargs
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path,
                **tokenizer_kwargs
            )
        elif "mistral" in model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(
                actual_model_path, 
                **model_kwargs
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path,
                **tokenizer_kwargs
            )
        else:
            raise ValueError(f"Invalid Model Path: {model_path}")
            
        print(f"✅ Model successfully loaded from: {actual_model_path}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error loading model from {actual_model_path}: {str(e)}")
        raise

def predict(args, prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens = args.max_length_cot,
        temperature=args.temperature, 
        pad_token_id=tokenizer.eos_token_id)
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res


def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to('cuda')
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
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
    model_name = HF_NAMES[model_path]

    model_input = tokenize(prompt, tokenizer, model_name).to(device)

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
