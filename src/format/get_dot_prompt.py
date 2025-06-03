import json
import logging

from config import args

# Draft of Thought (DoT) prompt templates - Chain-of-Draft approach
# DoT uses minimal draft steps with 5 words at most per step

instruction = '''
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Label each step as "Step i:", where "i" is the step number.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

Question: <QUESTION>
Response: 
'''

instruction_hotpotqa = '''
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Label each step as "Step i:", where "i" is the step number.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

The following is an example:
Question: Which band has more members, We Are the Ocean or The Dream Academy?
Response: 
Step 1: We Are the Ocean: 5
Step 2: The Dream Academy: 3  
Step 3: 5 > 3
Final Answer: We Are the Ocean

Question: <QUESTION>
Response: 
'''

instruction_math = '''
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Label each step as "Step i:", where "i" is the step number.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

The following is an example:
Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Response: 
Step 1: Blue fiber: 2 bolts
Step 2: White: half of 2
Step 3: White: 1 bolt
Step 4: Total: 2 + 1
Final Answer: 3

Question: <QUESTION>
Response: 
'''

instruction_2wiki = '''
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Label each step as "Step i:", where "i" is the step number.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

Question: <QUESTION>
Response: 
'''

def get_dot_prompt(args, question):
    if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
        prompt = instruction_math.replace('<QUESTION>', question)
    elif args.dataset == "hotpotQA":
        prompt = instruction_hotpotqa.replace('<QUESTION>', question)
    elif args.dataset == "2WikimhQA":
        prompt = instruction_2wiki.replace('<QUESTION>', question)
    else:
        prompt = instruction.replace('<QUESTION>', question)
    return prompt
