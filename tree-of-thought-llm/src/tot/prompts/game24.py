# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

# 5-shot
cot_prompt = '''Your Task is to generate the Answer field that shows the full trajectory of the solution. No explanation or comments. Strictly adhere to the format shown in the example.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: {input}
'''

# 1-shot
old_propose_prompt = '''Input: 2 2 4 6
Possible next steps:
4 * 6 = 24  (left: 2 2 24)
2 * 6 = 12  (left: 2 4 12)
2 * 4 = 8   (left: 2 8 6)
4 + 6 = 10  (left: 2 2 10)
2 + 6 = 8   (left: 2 4 8)
2 + 4 = 6   (left: 2 6 6)
6 / 2 = 3   (left: 2 4 3)
2 / 2 = 1   (left: 1 4 6)
6 - 2 = 4   (left: 2 4 4)
4 - 2 = 2   (left: 2 2 6)
2 - 2 = 0   (left: 0 4 6)
Input: {input}
Possible next steps:
'''

#this prompt was used for the variant that uses token probs and generates only one solution per step
single_solution_propose_prompt = """
## TASK: You are an expert Game of 24 Solver.

Given remaining numbers, output exactly one valid operation in the format:
a [operation] b = result (left: remaining numbers)
Use only one of the operations (+, -, *, /) between exactly two numbers.
Replace the used numbers with the result and list the new remaining numbers.
Always output a valid step even if not possible to reach 24(only 2 numbers exist that cant be combined to 24).
if no more valid steps exist output "None"
No extra text or comments; do not explain.

Example:
Input: 2 8 8 14
Possible next step:
14 - 8 = 6 (left: 2 6 8)

Input: {input}
Possible next step:
"""

# this prompt was used for the variant that uses token probs and generates multiple solutions per step 
propose_prompt = """
## TASK: Game of 24 Solver

**Rules:**
* Use each number exactly once.
* Combine exactly two numbers per step using only one of the operations +, -, *, or /.
* No extra text or comments, just the "Possible next step:" output as shown in the example.
* Only include the next step that is most likely to lead to a solution for Game of 24.

Input: 2 8 8 14
Possible next step:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next step:
"""
# this promt was used for the baseline that uses the self-probing variant and for the ranom baseline. 
multiple_solutions_propose_prompt = """
**Rules:**
* Use each number exactly once.
* Combine two numbers per step using +, -, *, or /.
* No extra text or comments, just the "Possible next steps:" output as shown in the example.

Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
'''

value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {input}
Answer: {answer}
Judge:'''