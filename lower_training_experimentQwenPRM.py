# pyright: reportPrivateImportUsage=false
from argparse import ArgumentParser
from datetime import datetime
import json
import random
import re

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parser = ArgumentParser()
parser.add_argument("--lower_training_data_path", type=str, default="./data/test_prm800k_18278samples.json")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--interval", type=int, default=100)
parser.add_argument("--subset_size", type=str, default="None")
parser.add_argument("--plot_type", type=str, default="epoch")
parser.add_argument("--pretrained", type=str, default="True")
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

lower_training_data_path = args.lower_training_data_path
batch_size = args.batch_size
num_epochs = args.num_epochs
interval = args.interval
subset_size = int(args.subset_size) if args.subset_size != "None" else None
plot_type = args.plot_type
pretrained = True if args.pretrained.lower() != "false" else False
lr = args.lr

model_name = "Qwen/Qwen2.5-Math-PRM-7B"
if pretrained == "True":
    model_name = f"{model_name}-Pretrained"
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1) # bs, seq_len, num_labels:2
    # probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    token_masks = (token_masks.bool().unsqueeze(-1)
                   .expand(-1, -1, probabilities.size(-1)))
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[token_masks[i]].view(-1, 2)[:, 1] # valid_tokens, num_labels
        # positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        all_scores_res.append(positive_probs[-1])
    return torch.stack(all_scores_res)

class Qwen_PRM_7B(torch.nn.Module):
    def __init__(self, model_name=model_name):
        super().__init__()
        if pretrained:
            self.base_model = AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
            )
        else:
            model_config = AutoConfig.from_pretrained(model_name)
            self.base_model = AutoModel.from_config(
                model_config,
            )
            self.base_model.to(torch.bfloat16)
        self.base_model.to(device)
    
    def forward(self, input_ids, attention_mask, token_masks):
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
        )
        return make_step_rewards(outputs[0], token_masks)    

with open(lower_training_data_path, "r") as f:
    data = json.load(f)
if subset_size is not None:
    data = random.sample(data, subset_size)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen_PRM_7B()

class LowerTrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

formatted_data = []
for sample in data:
    query = sample["input"]
    add_str = sample["add"]
    matches = list(re.finditer(r"Step\s+\d+:", add_str))
    steps = []
    if not matches:
        continue
    for i in range(len(matches)):
        start_idx = matches[i].end()
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            ans_match = list(re.finditer(r"# Answer:", add_str))
            if ans_match:   
                end_idx = ans_match[-1].end()
                answer = add_str[end_idx:].strip()
                add_str = add_str[:end_idx] + f" \\boxed{{{answer}}}"
            end_idx = len(add_str)
        step = add_str[start_idx:end_idx].strip()
        steps.append(step)
    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": query,
        "response": steps,
    }
    messages = [
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    # print(conversation_str) # debug
    # exit() # debug
    input_ids = tokenizer.encode(
        conversation_str, 
        return_tensors="pt", 
    )
    formatted_data.append({
        "input_ids": input_ids[0],
        "accuracy": torch.tensor(sample["accuracy"])
    })

def custom_collate_fn(batch_list):
    accuracies = [sample["accuracy"] for sample in batch_list]
    inputs = tokenizer.pad(batch_list, padding=True, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "accuracy": torch.stack(accuracies),
    }

dataset = LowerTrainingDataset(formatted_data)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    collate_fn=custom_collate_fn
)
step_sep_id = tokenizer.encode("<extra_0>")[0]
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

interval_loss = None
loss_log = [] # to plot loss curve
iter_list = [] # to plot loss curve
for epoch in range(num_epochs):
    tmp_sum_loss = 0.
    interval_count = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for i, samples in enumerate(pbar, start=1):
        optimizer.zero_grad()
        input_ids = samples["input_ids"].to(device)
        attention_mask = samples["attention_mask"].to(device)
        accuracies = samples["accuracy"].to(torch.bfloat16).to(device)
        token_masks = (input_ids == step_sep_id)
        step_rewards = model(input_ids, attention_mask, token_masks)
        # print(step_rewards) # debug
        # break # debug
        # print(step_rewards, accuracies) # debug
        loss = criterion(step_rewards, accuracies)
        loss.backward()
        optimizer.step()
        loss_float = loss.item()
        tmp_sum_loss += loss_float
        if plot_type != "epoch":
            interval_count += 1
            if interval_count >= interval:
                interval_loss = tmp_sum_loss / interval
                interval_count = 0
                tmp_sum_loss = 0.
                loss_log.append(interval_loss)
                iter_list.append(len(loss_log) * interval)
        if i == len(dataloader) and plot_type == "epoch":
            interval_loss = tmp_sum_loss / len(dataloader)
            tmp_sum_loss = 0.
            loss_log.append(interval_loss)
            iter_list.append(epoch+1)
        pbar.set_postfix({"loss": loss_float, "interval_loss": interval_loss})

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.plot(iter_list, loss_log)
if plot_type == "epoch":
    plt.xlabel("Epoch")
else:
    plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(f"plot_{date_str}.png")