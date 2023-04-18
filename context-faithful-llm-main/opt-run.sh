

for k in 125m 350m 1.3b
do
    for s in base attr instr opin instr+opin
    do
        for d in none original counter
        do
        python knowledge_conflict.py \
            --engine opt-$k \
            --schema $s \
            --demo_mode $d \
            --outfile opt_result.txt
        done
    done
done
    