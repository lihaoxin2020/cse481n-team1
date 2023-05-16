import json

# enter dataset file paths here, make sure to match 1 and 2 for all 3 sets of fields.

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
    line = od1.readline()
    datum = json.loads(line)
    for data in datum:
        uid = data['uid']
        od1_uids.add(uid)

od2_uids = set()
with open(orig_dataset2, "r") as od2:
    line = od2.readline()
    datum = json.loads(line)
    for data in datum:
        uid = data['uid']
        od2_uids.add(uid)

# print(type(od1))
od_intersect = od1_uids.intersection(od2_uids)
# od_intersect = od1_uids & od2

orig_examples = []
with open(orig_dataset1, 'r') as od1, open(out_orig_path, 'w') as out_od:
    line = od1.readline()
    datum = json.loads(line)
    for data in datum:
        uid = data['uid']
        if uid in od_intersect:
            orig_examples.append(data)
    
    print(f'Found {len(orig_examples)} original examples in intersection')
    json.dump(orig_examples, out_od)    



counter1_uids = set()
with open(counter_dataset1, "r") as cd1:
    line = cd1.readline()
    datum = json.loads(line)
    for data in datum:
        uid = data['uid']
        counter1_uids.add(uid)

counter2_uids = set()
with open(counter_dataset2, "r") as cd2:
    line =  cd2.readline()
    datum = json.loads(line)
    for data in datum:
        uid = data['uid']
        counter2_uids.add(uid)


cd_intersect = counter1_uids.intersection(counter2_uids)

counter_examples = []
with open(counter_dataset1, 'r') as cd1, open(out_counter_path, 'w') as out_cd:
    line = cd1.readline()
    datum = json.loads(line)
    for data in datum:
        uid = data['uid']
        if uid in cd_intersect:
            counter_examples.append(data)

    print(f'Found {len(counter_examples)} counter examples in intersection')
    json.dump(counter_examples, out_cd)
            
            






