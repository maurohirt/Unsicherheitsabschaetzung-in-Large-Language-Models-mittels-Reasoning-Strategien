import json
import torch
import time
import logging
from tqdm import tqdm

from config import args
from utils import print_exp
from torchmetrics import AUROC

from openai import OpenAI, APIError

SYSTEM_PROMPT_ORACLE_EQUIVALENCY = (
    "You are an automated grading assistant helping a teacher grade student answers."
)

PROMPT_ANSWER_KEY_EQUIVALENCY = (
    "The problem is: <question>\n\n The correct answer for this problem is: <ground-truth>\n "
    + "A student submitted the answer: <prediction>\n "
    + "The student's answer must be correct and specific but not overcomplete "
    + "(for example, if they provide two different answers, they did not get the question right). "
    + "However, small differences in formatting should not be penalized (for example, 'New York City' is equivalent to 'NYC'). "
    + "Did the student provide an equivalent answer to the ground truth? Please answer yes or no without any explanation: "
)

def openai_query(system_prompt, prompt, openai_model_name="gpt-4o-mini"):
    client = OpenAI()

    sampled_response = None
    while sampled_response is None:
        try:
            response = client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            sampled_response = response.choices[0].message.content
        except APIError:
            logging.exception("OpenAI API Error. Waiting 9 seconds before retrying...", exc_info=True)
            time.sleep(9)
    return sampled_response


def label_samples():
    # Check if output_v1_w_labels.json already exists
    import os
    output_file = f"{args.output_path}/output_v1_w_labels.json"
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Labels file {output_file} already exists. Skipping label generation.")
        return
    
    print(f"Generating labels and saving to {output_file}...")
    with open(f"{args.output_path}/output_v1.json", 'r', encoding='utf-8') as f:
        json_data = []
        for line in f.readlines():
            dic = json.loads(line)
            json_data.append(dic)

    # Open in write mode ('w') instead of append ('a') to avoid duplicates
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(json_data, total=len(json_data))):
            # give labels for open-ended llm answer
            id = line['id']
            question = line['question']
            correct_answer = line['correct answer']
            llm_answer = line['llm answer']

            if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                label = (str(correct_answer) in llm_answer.lower()) or (str(int(correct_answer)) in llm_answer.lower())
            else:
                t = line.get('type', '')
                prompt = (
                    PROMPT_ANSWER_KEY_EQUIVALENCY.replace("<ground-truth>", str(correct_answer))
                    .replace("<prediction>", llm_answer)
                    .replace("<question>", question)
                )

                sampled_response = openai_query(
                    system_prompt=SYSTEM_PROMPT_ORACLE_EQUIVALENCY, prompt=prompt
                )
                
                label = "yes" in sampled_response.strip().lower()

            formatted_data = {
                "id": id,
                "question": question,
                # "type": t,
                "correct answer": correct_answer,
                "llm answer": llm_answer,
                "label": label,
                "llm response": line['llm response'],
                "llm answer token probability": line['llm answer token probability'], 
                "step-wise keywords": line['step-wise keywords'],
                "keyword token probability": line['keyword token probability'],
                "keyword contribution": line['keyword contribution'],
            }
            f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")


def compute_auroc():
    json_data_labels = []
    with open(f"{args.output_path}/output_v1_w_labels.json", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            json_data_labels.append(dic)

    json_data_output = []
    with open(f"{args.output_path}/confidences/output_v1_{args.uq_engine}.json", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            json_data_output.append(dic)

    label_dict = {}

    for idx, line in enumerate(json_data_labels):
        label_dict[line['question']] = 1 if line['label'] == True else 0

    all_confidences = []
    all_auroc_target = []
    for idx, line in enumerate(json_data_output):
        question = line['question']
        all_confidences.append(line['confidence'])
        all_auroc_target.append(label_dict[question])

    # calculate AUROC
    auroc = AUROC(task="binary")
    auroc_value = auroc(torch.tensor(all_confidences), torch.tensor(all_auroc_target))

    print(f"AUROC: {auroc_value}")


if __name__ == '__main__':
    print_exp(args) 

    label_samples()
    compute_auroc()
