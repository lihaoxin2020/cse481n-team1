import json
from knowledge_conflict import eval
import tqdm

log1 = "" # log file 1

log2 = "" # log file 2

engine1 = "" # model 1
engine2 = "" # model 2

outfile1 = f'results_{engine1}.txt'
outfile2 = f'results_{engine2}.txt'

def generate_intersection():
    od1_uids = set()
    with open(log1, "r") as lg1:
        line = lg1.readline()
        datum = json.loads(line)
        for data in datum:
            uid = data['uid']
            od1_uids.add(uid)

    od2_uids = set()
    with open(log2, "r") as lg2:
        line = lg2.readline()
        datum = json.loads(line)
        for data in datum:
            uid = data['uid']
            od2_uids.add(uid)

    # print(len(od1_uids), len(od2_uids))
    od_intersect = od1_uids.intersection(od2_uids)
    # print(len(od_intersect))
    return od_intersect

def get_answers(file):
    common_uids = generate_intersection()

    orig_answers = []
    pred_answers = []
    gold_answers = []
    with open(file, 'r') as od:
        datum = json.loads(od.readline())
        # cd_datum = json.loads(cd.readline())
        # print(len(datum))
        for oe in datum:
            if oe['uid'] in common_uids:
                if 'orig_answer' in oe and 'prediction' in oe and 'gold_answer' in oe:
                    # print(oe['orig_answer'])
                    orig_answers.append(oe['orig_answer'])
                    pred_answers.append(oe['prediction'])
                    gold_answers.append(oe['gold_answer'])
    
    return (orig_answers, pred_answers, gold_answers)

if __name__ == '__main__':
    origs, preds, golds = get_answers(log1)
    # print(len(origs))
    # print(len(preds))
    # print(len(golds))
    eval(preds, origs, golds, outfile1)
    origs, preds, golds = get_answers(log2)
    eval(preds, origs, golds, outfile2)

