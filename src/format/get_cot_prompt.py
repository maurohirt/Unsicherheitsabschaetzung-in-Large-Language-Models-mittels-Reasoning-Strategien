import json
import logging

from config import args

instruction = '''
Please reason the following question step by step. Label each reasoning step as "Step i:", where "i" is the step number.
You need to ensure that each step builds on the previous one and contributes meaningfully toward reaching the final answer.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

Question: <QUESTION>
Response: Let's think step by step.
'''

instruction_hotpotqa = '''
Please reason the following question step by step. Label each reasoning step as "Step i:", where "i" is the step number.
You need to ensure that each step builds on the previous one and contributes meaningfully toward reaching the final answer.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

The following is an example:
Question: Which band has more members, We Are the Ocean or The Dream Academy?
Response: Let's think step by step.
Step 1: We Are the Ocean has 5 members.
Step 2: The Dream Academy has 3 members.
Step 3: 5 is greater than 3.
Step 4: Therefore, We Are the Ocean has more members.
Final Answer: We Are the Ocean

Question: <QUESTION>
Response: Let's think step by step.
'''

instruction_math = '''
Please reason the following question step by step. Label each reasoning step as "Step i:", where "i" is the step number.
You need to ensure that each step builds on the previous one and contributes meaningfully toward reaching the final answer.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

The following is an example:
Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Response: Let's think step by step.
Step 1: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  
Step 2: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  
Step 3: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.
Final Answer: 3

Question: <QUESTION>
Response: Let's think step by step.
'''

instruction_2wiki = '''
Please reason the following question step by step. Label each reasoning step as "Step i:", where "i" is the step number.
You need to ensure that each step builds on the previous one and contributes meaningfully toward reaching the final answer.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

The following is an example:
Question: 
Who is the paternal grandmother of Joseph Ferdinand Of Bavaria?
Response: Let's think step by step.
Step 1: Joseph Ferdinand's father was Maximilian II Emanuel, Elector of Bavaria.
Step 2: Maximilian II Emanuel was the son of Ferdinand Maria, Elector of Bavaria, and his wife, Henriette Adelaide of Savoy.
Step 3: As the mother of Maximilian II Emanuel, Henriette Adelaide of Savoy is the paternal grandmother of Joseph Ferdinand of Bavaria.
Final Answer: Henriette Adelaide of Savoy

Question:
<QUESTION>
Response: Let's think step by step.
'''


def get_cot_prompt(args, question):
    if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
        prompt = instruction_math.replace('<QUESTION>', question)
    elif args.dataset == "hotpotQA":
        prompt = instruction_hotpotqa.replace('<QUESTION>', question)
    elif args.dataset == "2WikimhQA":
        prompt = instruction_2wiki.replace('<QUESTION>', question)
    else:
        prompt = instruction.replace('<QUESTION>', question)
    
    return prompt