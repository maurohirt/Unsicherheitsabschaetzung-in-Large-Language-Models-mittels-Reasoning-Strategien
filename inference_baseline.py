import json
import logging
import re
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F

from config import args
from src.model.llama2_predict import predict, model_init
from src.format.get_cot_prompt import get_cot_prompt

from utils import load_data, print_exp, setup_log, is_effectively_empty, find_subsequence_position, \
                    parse_response_to_dict, find_token_indices, is_word_in_sentence, match_final_answer_token_ids 

def llama_inference_baseline():
    # Add baseline to output path
    baseline_dir = os.path.join(args.output_path, "baseline")
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir) 
    
    output_file = os.path.join(baseline_dir, "output_v1.json")
    
    log = setup_log(args)

    if args.dataset in ["hotpotQA", "2WikimhQA"]:
        question, answer, ids, types = load_data(args)
    else:
        question, answer, ids = load_data(args)
    model, tokenizer, device = model_init(args)
    model.eval()
    
    for idx, q in enumerate(tqdm(question, total=len(question))):
        with open(output_file, "a", encoding="utf-8") as f:

            if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                a = answer[idx]
                id = ids[idx]
            else:
                a = answer[idx]       
                id = ids[idx]
                t = types[idx]

            log.debug(f"##### This is the --{idx + 1}th-- Question #####")

            cot_prompt = get_cot_prompt(args, q)

            inputs = tokenizer(cot_prompt, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            try_time = 0
            while try_time < args.try_times:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens = args.max_length_cot, 
                    temperature=args.temperature, 
                    pad_token_id=tokenizer.eos_token_id, 
                    return_dict_in_generate=True, 
                    output_scores=True,
                )

                generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):-1]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                llm_answer, steps_dict, response = parse_response_to_dict(response)
                if generated_ids.size(0) >= args.max_length_cot:
                    log.debug(f'New Reasoning Tokens Are Too Much, Current try is {try_time + 1}')
                    try_time += 1
                elif generated_ids.size(0) == 0:
                    log.debug(f'New Reasoning Tokens Are Null, Current try is {try_time + 1}')
                    try_time += 1
                elif llm_answer is None or llm_answer in ['', ' ']:
                    log.debug(f'New Reasoning Tokens Are None, Current try is {try_time + 1}')
                    try_time += 1
                else:
                    response_tokens = tokenizer.tokenize(response)
                    response_token_ids = tokenizer.convert_tokens_to_ids(response_tokens)
                    original_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
                    probabilities = [{i: p for i, p in enumerate(prob[0]) if p > 0} for prob in [torch.softmax(score, dim=1).tolist() for score in outputs.scores]]

                    final_answer_probabilities = {}
                    final_answer_token_ids = {}
                    answer_start_indice, answer_token_ids = match_final_answer_token_ids(args, original_tokens, response_tokens, generated_ids)
                    if answer_start_indice == None:
                        log.debug(f'Cannot locate the Final Answer, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    answer_probs = []
                    flag = False
                    for j, token_id in enumerate(answer_token_ids):
                        idxx = j + answer_start_indice
                        if token_id not in probabilities[idxx].keys():                       
                            flag = True
                            break
                        answer_probs.append(probabilities[idxx][token_id])
                    if flag:
                        log.debug(f'Cannot locate the Final Answer Token Probability, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    final_answer_probabilities[llm_answer] = answer_probs
                    final_answer_token_ids[llm_answer] = answer_token_ids.tolist()

                    # BASELINE: Skip keyword extraction since we treat every token as a keyword
                    keywords_probabilities = {}
                    keywords_contributions = {}
                    keywords_token_ids = {}
                    step_wise_keywords = {}
                    
                    for step_idx, (step_name, step_text) in enumerate(steps_dict.items()):
                        # Tokenize the step
                        step_tokens = tokenizer.tokenize(step_text)
                        processed_step_tokens = [token[1:] if token.startswith('Ġ') or token.startswith('▁')  else token for token in step_tokens]
                        step_token_ids = tokenizer.convert_tokens_to_ids(step_tokens)
                        
                        # Find position in generated sequence
                        start_position = find_subsequence_position(step_token_ids[1:-2], generated_ids) - 1
                        if start_position < 0:
                            log.debug(f"Could not locate step {step_name} in generated sequence")
                            continue
                            
                        step_token_ids = generated_ids[start_position: start_position + len(step_tokens)]
                        
                        # For baseline, each token is a keyword with importance 1
                        keywords_probabilities_dict = {}
                        keywords_contributions_dict = {}
                        keywords_token_ids_dict = {}
                        step_keywords = []
                        
                        # Group tokens into words for more meaningful "keywords"
                        # This is a simple approximation - in real text, words might span multiple tokens
                        current_word = ""
                        current_word_probs = []
                        current_word_ids = []
                        
                        for i, token in enumerate(step_tokens):
                            # Simple word boundary detection
                            is_new_word = token.startswith('Ġ') or token.startswith('▁') or i == 0
                            
                            # If new word and we have a previous word, add it to our dictionaries
                            if is_new_word and current_word:
                                # Add the completed word
                                keywords_probabilities_dict[current_word] = current_word_probs
                                keywords_contributions_dict[current_word] = 1  # Equal importance
                                keywords_token_ids_dict[current_word] = current_word_ids
                                step_keywords.append(current_word)
                                
                                # Reset for new word
                                current_word = ""
                                current_word_probs = []
                                current_word_ids = []
                            
                            # Clean token for display
                            clean_token = token[1:] if (token.startswith('Ġ') or token.startswith('▁')) else token
                            current_word += clean_token
                            
                            # Get probability for this token
                            token_id = step_token_ids[i]
                            position = start_position + i
                            if position < len(probabilities) and token_id.item() in probabilities[position]:
                                current_word_probs.append(probabilities[position][token_id.item()])
                            else:
                                current_word_probs.append(0.0)
                            
                            current_word_ids.append(token_id.item())
                        
                        # Add the last word if exists
                        if current_word:
                            keywords_probabilities_dict[current_word] = current_word_probs
                            keywords_contributions_dict[current_word] = 1  # Equal importance
                            keywords_token_ids_dict[current_word] = current_word_ids
                            step_keywords.append(current_word)
                        
                        # Add to overall dictionaries
                        keywords_probabilities[step_name] = keywords_probabilities_dict
                        keywords_contributions[step_name] = keywords_contributions_dict
                        keywords_token_ids[step_name] = keywords_token_ids_dict
                        
                        # Format step-wise keywords string for this step
                        step_wise_keywords[step_name] = step_keywords

                    # Check if we successfully processed all steps
                    if is_effectively_empty(keywords_probabilities):
                        log.debug(f'Token Probability from All Steps are All None, Current try is {try_time + 1}')
                        try_time += 1
                        continue

                    # Format step-wise keywords string
                    step_wise_keywords_formatted = ""
                    for step_name, keywords in step_wise_keywords.items():
                        # Format as: "Step X: word1(/1/); word2(/1/)"
                        step_wise_keywords_formatted += f"{step_name}: " + "; ".join([f"{word}(/1/)" for word in keywords]) + "\n"

                    # Create the same output format as the original
                    if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                        formatted_data = {
                            "id": id,
                            "question": q,
                            "correct answer": a,
                            "llm response": response,
                            "llm answer": llm_answer,
                            "llm answer token probability": final_answer_probabilities, 
                            "step-wise keywords": step_wise_keywords_formatted,
                            "keyword token probability": keywords_probabilities,
                            "keyword contribution": keywords_contributions,
                            "keyword token ids": keywords_token_ids,
                            "final answer token ids": final_answer_token_ids,
                        }
                    else:
                        formatted_data = {
                            "id": id,
                            "question": q,
                            "correct answer": a,
                            "type": t,
                            "llm response": response,
                            "llm answer": llm_answer,
                            "llm answer token probability": final_answer_probabilities, 
                            "step-wise keywords": step_wise_keywords_formatted,
                            "keyword token probability": keywords_probabilities,
                            "keyword contribution": keywords_contributions,
                            "keyword token ids": keywords_token_ids,
                            "final answer token ids": final_answer_token_ids,
                        }
                    f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
                    break
        if try_time >= args.try_times:
            log.debug(f'Cannot extract all necessary data after {args.try_times} tries')
            with open(f"{baseline_dir}/error.log", "a", encoding="utf-8") as err_f:
                err_f.write(f"Failed inference for question {idx+1}: {q}\n")


if __name__ == '__main__':
    print_exp(args) 

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        llama_inference_baseline()
    else:
        raise ValueError(f"Model engine {args.model_engine} is not supported by this script.")
