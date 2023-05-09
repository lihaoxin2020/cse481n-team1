import json

orig_dataset1 = "" 
orig_dataset2 = ""

counter_dataset1 = ""
counter_dataset2 = ""

engine1 = ""
engine2 = ""

out_counter_path = f'./conflict_dev_filtered_{engine1}_{engine2}.json'
out_orig_path = f'./orig_dev_filtered_{engine1}_{engine2}.json'

od1_uids = set()
with open(orig_dataset1, "r") as od1:
    for i, line in enumerate(od1):
        datum = json.loads(line)
        uid = datum['uid']
        od1_uids.add(uid)

od2_uids = set()
with open(orig_dataset2, "r") as od2:
    for i, line in enumerate(od2):
        datum = json.loads(line)
        uid = datum['uid']
        od2_uids.add(uid)


od_intersect = od1.intersection(od2)

orig_examples = []
with open(orig_dataset1, 'r') as od1, with open(out_orig_path, 'w') as out_od:
     for i, line in enumerate(od1):
        datum = json.loads(line)
        uid = datum['uid']
        if uid in od_intersect:
            orig_examples.add(datum)
    
    json.dump(orig_examples, out_od)



counter1_uids = set()
with open(counter_dataset1, "r") as cd1:
    for i, line in enumerate(cd1):
        datum = json.loads(line)
        uid = datum['uid']
        counter1_uids.add(uid)

counter2_uids = set()
with open(counter_dataset2, "r") as cd2:
    for i, line in enumerate(cd2):
        datum = json.loads(line)
        uid = datum['uid']
        counter2_uids.add(uid)


cd_intersect = cd1.intersection(cd2)

counter_examples = []
with open(counter_dataset1, 'r') as cd1, with open(out_counter_path, 'w') as out_cd:
     for i, line in enumerate(cd1):
        datum = json.loads(line)
        uid = datum['uid']
        if uid in cd_intersect:
            counter_examples.add(datum)
    
    json.dump(counter_examples, out_cd)
            
            






