import torch
import time
import os
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login as hf_login
# from peft import PeftModel

HF_NAMES = {
    # Official Llama-3 repositories (with custom modeling code supporting GQA)
    # Llama-3.1 release (preferred)
    'llama3-1_8B': 'meta-llama/Llama-3.1-8B',
    #'llama3-1_8B_instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    # Legacy repo name kept for backward compatibility
   # 'llama3-1_8B_meta': 'meta-llama/Meta-Llama-3-8B',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
}

def model_init(args):
    model_path = args.model_path # Replace to your model path
    device = torch.device("cuda:0")
    # login to Hugging Face Hub if token provided via env or args (for offline cache access)
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", getattr(args, "hf_token", ""))
    if token:
        try:
            hf_login(token=token, add_to_git_credential=False)
        except Exception:
            pass
    if "llama3" in args.model_path:
        # Use AutoModel for Llama-3 because HF provides custom QKV shapes
        model = AutoModelForCausalLM.from_pretrained(
            HF_NAMES[model_path],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(HF_NAMES[model_path], trust_remote_code=True)
    elif "llama2" in args.model_path:
        model = LlamaForCausalLM.from_pretrained(
            HF_NAMES[model_path],
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Enable multi-GPU if available
        )
        tokenizer = AutoTokenizer.from_pretrained(HF_NAMES[model_path])
    elif "mistral" in args.model_path:
        model = AutoModelForCausalLM.from_pretrained(HF_NAMES[model_path], torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(HF_NAMES[model_path])
    else:
        raise("Invalid Model Path")
    return model, tokenizer, device

def predict(args, prompt, model, tokenizer):
    # Get device from model's parameters rather than hard-coding 'cuda'
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens = args.max_length_cot,
        temperature=args.temperature, 
        pad_token_id=tokenizer.eos_token_id)
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res


def tokenize(prompt, tokenizer, model_name, model=None, tokenizer_args=None):
    # Determine device from model if provided, default to CUDA
    device = next(model.parameters()).device if model is not None else torch.device("cuda")
    
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to(device)
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors="pt", **(tokenizer_args or {}))
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