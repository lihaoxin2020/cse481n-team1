import json
import gzip
from tqdm import tqdm
import string
import re
# import openai
from time import sleep
from load_model import Engine

counter_path = './datasets/substitution-sets/MRQANaturalQuestionsDev-corpus-substitution.jsonl'
orig_path = './datasets/normalized/MRQANaturalQuestionsDev.jsonl.gz'

openai.api_key = 'MY_API_KEY'

# class Engine:
#     def __init__(self, engine='text-davinci-003'):
#         self.engine = engine

#     def request(self, prompt):
#         num_retry = 0
#         while True:
#             try:
#                 response = openai.Completion.create(
#                     engine=self.engine,
#                     prompt=prompt,
#                     max_tokens=128,
#                     temperature=0,
#                 )
#             except Exception:
#                 if num_retry >= 3:  # Retried too many times, return Error
#                     print('Retried too many times, skip this instance.')
#                     return None
#                 sleep(2)
#                 num_retry += 1
#                 continue
#             break
#         answer = response.choices[0].text
#         return answer

# Process substituted dataset
counter_examples = []
uids = []
with open(counter_path, "r") as inf:
    header = json.loads(inf.readline())
    for i, line in enumerate(inf):
        datum = json.loads(line)
        uid = datum['uid']
        uid = uid.split('_')[-1]
        assert datum['is_substitute'] is True
        context = datum['context'].replace('<P>', '').replace('</P>', '').strip()
        question = datum['query'] + '?'
        assert len(datum['gold_answers']) == 1
        answer = datum['gold_answers'][0]['text']
        example = {
            'question': question,
            'context': context,
            'answer': answer,
            'uid': uid
        }
        counter_examples.append(example)
        uids.append(uid)

# Process original dataset, sort uids in same order of the substituted dataset
original_examples = {}
with gzip.open(orig_path, "r") as inf:
    uids_set = set(uids)
    header = json.loads(inf.readline())
    for i, line in enumerate(inf):
        datum = json.loads(line)
        uid = datum['uid']
        if uid in uids_set:
            context = datum['context'].replace('<P>', '').replace('</P>', '').strip()
            question = datum['query'] + '?'
            answers = [answer['text'] for answer in datum['gold_answers']]
            example = {
                'question': question,
                'context': context,
                'answer': answers,
                'uid': uid
            }
            original_examples[uid] = example
assert len(uids) == len(original_examples)
original_examples = [original_examples[uid] for uid in uids]

def simple_qa_prompt(query):
    prompt = 'Q:{}\nA:'.format(query)
    return prompt

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def is_memorized(pred, gold_answers):
    pred = normalize_answer(pred)
    for gold in gold_answers:
        gold = normalize_answer(gold)
        if gold in pred:
            return True
    return False

filtered_orig_examples = []
filtered_counter_examples = []
added_instances = set()
engine = Engine() # specify which model you want to use here by 
gold_answers, pred_answers, orig_answers = [], [], []
for oe, ce in tqdm(zip(original_examples, counter_examples), total=len(original_examples)):
    query, context, answer = ce['question'], ce['context'], ce['answer']
    orig_answer = oe['answer']
    prompt = simple_qa_prompt(query)
    pred = engine.complete(prompt)
    if pred is None:
        continue
    if is_memorized(pred, orig_answer):
        if (question, context, answer) not in added_instances:  # Avoid adding duplicate instances
            added_instances.add((question, context, answer))
        else:
            continue
        filtered_orig_examples.append(oe)
        filtered_counter_examples.append(ce)
print('Get {} filtered examples.'.format(len(filtered_orig_examples)))

out_counter_path = f'./conflict_dev_filtered_{engine.engine}.json'
out_orig_path = f'./orig_dev_filtered_{engine.engine}.json'

with open(out_orig_path, 'w') as fh:
    json.dump(filtered_orig_examples, fh)
with open(out_counter_path, 'w') as fh:
    json.dump(filtered_counter_examples, fh)
