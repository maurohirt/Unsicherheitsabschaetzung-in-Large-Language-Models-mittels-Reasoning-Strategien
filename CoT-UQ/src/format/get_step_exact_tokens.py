instruction = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:

Question:
<QUESTION>
Multi-Step Response:
<RESPONSE>
Keywords for Each Reasoning Step:
'''


instruction_hotpotqa_w_contribution_score = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: Which band has more members, "We Are the Ocean" or "The Dream Academy"?
A: Step 1: The question is asking which band has more members.
Step 2: "We Are the Ocean" has 5 members.
Step 3: "The Dream Academy" has 3 members.
Step 4: 5 is greater than 3.
Step 5: Therefore, "We Are the Ocean" has more members.
Final Answer: We Are the Ocean
Keywords for Each Reasoning Step: 
Step 1: NO ANSWER
Step 2: We Are the Ocean(/5/); 5(/10/)
Step 3: The Dream Academy(/5/); 3(/10/)
Step 4: greater(/7/)
Step 5: We Are the Ocean(/5/)

The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:
'''

instruction_math_w_contribution_score = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Step 1: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  
Step 2: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  
Step 3: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.
Final Answer: 3
Keywords for Each Reasoning Step: Step 1: 2 bolts (/3/)
Step 2: 1 bolt (/10/)
Step 3: 3 bolts (/7/)

The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:
'''

instruction_2wiki_w_contribution_score = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: Who is the paternal grandmother of Joseph Ferdinand Of Bavaria?
A: Step 1: Joseph Ferdinand's father was Maximilian II Emanuel, Elector of Bavaria.
Step 2: Maximilian II Emanuel was the son of Ferdinand Maria, Elector of Bavaria, and his wife, Henriette Adelaide of Savoy.
Step 3: As the mother of Maximilian II Emanuel, Henriette Adelaide of Savoy is the paternal grandmother of Joseph Ferdinand of Bavaria.
Final Answer: Henriette Adelaide of Savoy
Keywords for Each Reasoning Step: Step 1: father (/8/); Maximilian II Emanuel, Elector of Bavaria (/8/)
Step 2: son (/8/); Ferdinand Maria, Elector of Bavaria (/5/); Henriette Adelaide of Savoy (/9/)
Step 3: mother (/10/)

The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:
'''


def get_step_exact_tokens(args, question, cot_response):
    if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
        prompt = instruction_math_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    elif args.dataset == "hotpotQA":
        prompt = instruction_hotpotqa_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    elif args.dataset == "2WikimhQA":
        prompt = instruction_2wiki_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    else:
        prompt = instruction.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    return prompt