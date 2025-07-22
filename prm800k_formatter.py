"""
Format PRM800K to use for lower-level optimization for DreamPRM.
"""

import json
from random import shuffle
from tqdm import tqdm

num_samples = 100000
in_f  = "data/prm800k/phase2_test.jsonl"
# out_f = f"data/lower_data/test_prm800k_{num_samples}samples.json"

records = []
sample_id = 0
sampled = []

with open(in_f) as fin:
    pbar = tqdm(fin)
    for sample_id, line in enumerate(pbar):
        ex = json.loads(line)
        instr = ex["question"]["problem"]
        # pull the official answer string
        ground_truth = ex["question"].get("ground_truth_answer", "")

        # iterate steps with a step‐index sid
        reason = ex["label"]["finish_reason"]
        if reason != "solution" and reason != "found_error":
            continue
        prev_add_str = ""
        for sid, step_obj in enumerate(ex["label"]["steps"], start=1):
            comps = step_obj.get("completions", [])
            if not comps:
                continue
            for comp in comps:
                text = comp["text"].strip()
                add_str = (
                    prev_add_str + "Step " + str(sid) + ": " + text + "\n\n"
                )
                rating = comp["rating"]
                if rating is None:
                    continue
                accuracy = rating * 0.5 + 0.5
                records.append({
                    "id":           sample_id,      # unique sample ID
                    "sid":          sid,            # step number within problem
                    "input":        instr,          # full question prompt
                    "add":          add_str,        # this single CoT step
                    "ground_truth": ground_truth,   # correct final answer
                    "image_path":   "",             # no image for PRM800K
                    "dataset":      str(sample_id), # domain name
                    "score":        rating,         # here: human rating {-1, 0, 1}
                    "times":        1,              # default 1 annotation
                    "accuracy":     accuracy        # {0, 0.5, 1}
                })
            prev_add_str = add_str

while len(sampled) < num_samples:
    shuffle(records)
    checked_id_pairs = set() # don't sample the same id and sid pair twice
    for record in records:
        if (record["id"], record["sid"]) in checked_id_pairs:
            continue
        checked_id_pairs.add((record["id"], record["sid"]))
        sampled.append(record)
        if len(sampled) >= num_samples:
            print(f"Sampled {len(sampled)} samples")
            break
    else:
        print(
            f"Checked all samples and sampled {len(sampled)} samples "
            f"despite num_samples: {num_samples}"
        )
        break

out_f = f"data/lower_data/test_prm800k_{len(sampled)}samples.json"
# write as a single JSON array
with open(out_f, "w") as fout:
    json.dump(sampled, fout, indent=2)
print(f"Written to {out_f}")
