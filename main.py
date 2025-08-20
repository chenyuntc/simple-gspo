import os
import re
import math
import random
import gc
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# extract code from markdown blocks
def extract_code_from_markdown(text: str) -> str:
    import re
    # Look for ```python code blocks
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: look for any ``` code blocks
    match = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# extract the generated code from <answer> tags
def extract_xml_answer(text: str) -> str:
    try:
        answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
        # Extract code from the answer content
        return extract_code_from_markdown(answer_content)
    except IndexError:
        return ""

def compute_format_score(batch_responses, verbose=True):
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    matches = [bool(re.match(pattern, g_a, flags=re.DOTALL)) for g_a in batch_responses]
    # re.match(pattern, g_a, flags=re.DOTALL)
    # match2 = 'def candidate(' in batch_responses[0]
    matches2 =['def candidate(' in g_a for g_a in batch_responses]
    format_scores = [0.5 if match else 0.0 for match in matches]
    format_scores2 = [1.0 if match2 else 0.0 for match2 in matches2]
    format_scores = [fs1 + fs2 for fs1, fs2 in zip(format_scores, format_scores2)]
    # if verbose:
    #     print(format_scores, 'format scores')
    return format_scores

def execute_code_with_tests(code: str, test_code: str) -> bool:
    """Execute generated code with test cases and return pass/fail."""
    import subprocess
    import tempfile
    # import ipdb;ipdb.set_trace()
    
    try:
        # Create temporary file with code + tests
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_file = f.name
            f.write(code + "\n\n" + test_code + "\n \ncheck(candidate)")
            f.flush()
        
        # Execute the code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=5.0  # 5 second timeout
        )
        
        # Clean up
        # os.unlink(temp_file)
        
        # Return True if no errors (exit code 0)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Code execution error: {e}")
        return False

def compute_reward(batch_codes, test_codes, verbose=True):
    """Reward function that executes code and checks if tests pass."""
    reward_scores = []
    for code, test_code in zip(batch_codes, test_codes):
        if not code:  # Empty code
            reward_scores.append(-1.5)
        elif execute_code_with_tests(code, test_code):
            reward_scores.append(2.0)  # Test passed
        else:
            reward_scores.append(-0.5)  # Test failed
    # if verbose:
    #     print(reward_scores, 'reward scores')
    return reward_scores
    
# learning rate schedule
def get_lr(it, max_steps, warmup_steps = None, max_lr=1e-5, min_lr=1e-6):
    warmup_steps = int(0.1*max_steps)
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def evaluate(policy_model, tokenizer, test_loader, local_rank):
    policy_model.eval()
    total_rewards = []
    all_format_scores = []
    all_reward_scores = []

    with torch.no_grad():
        for prompts, _, tests in tqdm(test_loader, desc=f"Evaluating on rank {local_rank}"):
            prompt_enc = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                padding_side='left',
                truncation=True
            )
            input_ids = prompt_enc["input_ids"].to(local_rank)
            attention_mask = prompt_enc["attention_mask"].to(local_rank)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_ids = policy_model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use greedy decoding for evaluation
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                )

            prompt_len = input_ids.shape[1]
            batch_responses_ids = generated_ids[:, prompt_len:]
            batch_responses = tokenizer.batch_decode(batch_responses_ids, skip_special_tokens=True)
            batch_codes = [extract_xml_answer(res) for res in batch_responses]

            format_scores = compute_format_score(batch_responses, verbose=False)
            reward_scores = compute_reward(batch_codes, tests, verbose=False)
            
            rewards = [fs + rs for fs, rs in zip(format_scores, reward_scores)]

            total_rewards.extend(rewards)
            all_format_scores.extend(format_scores)
            all_reward_scores.extend(reward_scores)

    # Gather results from all processes
    world_size = dist.get_world_size()
    if world_size > 1:
        # Each process sends its list of results to all other processes
        gathered_rewards = [None] * world_size
        gathered_format_scores = [None] * world_size
        gathered_reward_scores = [None] * world_size
        
        dist.all_gather_object(gathered_rewards, total_rewards)
        dist.all_gather_object(gathered_format_scores, all_format_scores)
        dist.all_gather_object(gathered_reward_scores, all_reward_scores)

        # On master process, combine the results
        if dist.get_rank() == 0:
            total_rewards = [item for sublist in gathered_rewards for item in sublist]
            all_format_scores = [item for sublist in gathered_format_scores for item in sublist]
            all_reward_scores = [item for sublist in gathered_reward_scores for item in sublist]
        else:
            # Other processes don't need to do the calculation
            total_rewards = []
            all_format_scores = []
            all_reward_scores = []


    policy_model.train()

    # Calculation is now only done on master process if world_size > 1
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    avg_format_score = sum(all_format_scores) / len(all_format_scores) if all_format_scores else 0
    avg_reward_score = sum(all_reward_scores) / len(all_reward_scores) if all_reward_scores else 0

    return avg_reward, avg_format_score, avg_reward_score

# --------------------------------------
# Constants: Chat Prompt Templates
# --------------------------------------
SYSTEM_PROMPT = ("""
    A conversation between User and Assistant. The user asks a coding problem, and the Assistant solves it.
    The assistant first thinks about the reasoning process and then provides the user
    with the complete Python solution. The reasoning process and code are enclosed within <reasoning> </reasoning> and
    <answer> </answer> tags, respectively. The output python function should be named `candidate`.
    Example:
    <reasoning>I need to implement a function that checks if a number is even...</reasoning>
    <answer>
    ```python
    def candidate(n):
        return n % 2 == 0
    ```
    </answer>
    """
)
TASK_SPECIFIC_INSTRUCTIONS = "Provide complete, working Python code in markdown code blocks."
# --------------------------------------
# Distributed Setup
# --------------------------------------
# Initialize the process group for multi-GPU training
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
master_process = rank == 0

# Set random seeds uniquely per process for reproducibility
seed = 42 + rank
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------------------------------------
# Hyperparameters
# --------------------------------------
max_grad_norm = 0.1          # For gradient clipping
ppo_clip_range = 0.2         # ε for PPO clipping
kl_coef = 0.005                # Multiplier for KL divergence penalty
batch_size = 4               # Prompts per batch per GPU
K = 4                        # Number of samples (responses) per prompt
ppo_steps = 4               # PPO update epochs per batch
max_new_tokens = 256*2         # Max tokens to generate per sample
num_epochs = 5               # Number of training epochs
weight_decay = 0.1  
initial_learning_rate = 1e-5

# --------------------------------------
# Load Model & Tokenizer
# --------------------------------------
model_name  = 'Qwen/Qwen2.5-3B-Instruct'
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure pad token is defined (use eos as pad)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------
# Model Setup
# --------------------------------------

# Load policy model (trainable)
policy_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
policy_model.train()
# Load reference model (fixed)
ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
ref_model.eval()
# Freeze reference model parameters
for p in ref_model.parameters():
    p.requires_grad = False

# Wrap the policy model with DDP for gradient synchronization
policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank)

# Setup optimizer
param_dict = {pn: p for pn, p in policy_model.named_parameters() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr = initial_learning_rate)
optimizer.zero_grad(set_to_none = True)

# log
if master_process:
    log_dir = "./gspo_models"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    print(log_file)

# --------------------------------------
# Load Dataset
# --------------------------------------
train_data = load_dataset("openai_humaneval")["test"]
print(len(train_data))

class WrapperDataset(Dataset):
    """Dataset wrapper for HumanEval data: yields (prompt, canonical_solution, test) pairs."""
    def __init__(self, hf_dataset, train=True):
        self.data = hf_dataset
        self.train = train
    def __len__(self):
        return len(self.data)-16 if self.train else 16
    def __getitem__(self, idx):
        if not self.train:
            idx = len(self.data) - 16 + idx
        item = self.data[idx]
        return item["prompt"], item["canonical_solution"], item["test"]

test_data = WrapperDataset(train_data, train=False)
train_data = WrapperDataset(train_data)

def collate_fn(batch):
    prompts, solutions, tests = [], [], []
    for problem_prompt, canonical_solution, test_code in batch:
        chat_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "def candidate(n):\n    # Check if number is even\n    pass"},
            {'role': 'assistant', 'content': "<reasoning>I need to check if a number is divisible by 2. If n % 2 equals 0, then n is even.</reasoning>\n<answer>\n```python\ndef candidate(n):\n    return n % 2 == 0\n```\n</answer>"},
            {'role': 'user', 'content': problem_prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            chat_prompt,
            tokenize = False,
            add_generation_prompt = True
        )
        prompts.append(prompt)
        solutions.append(canonical_solution)
        tests.append(test_code)

    return prompts, solutions, tests

# Set up DataLoader with DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=world_size, rank=rank, shuffle=True
)
train_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=4,    
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_data, num_replicas=world_size, rank=rank, shuffle=False
)
test_loader = DataLoader(
    test_data, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn, num_workers=4,    
)

max_steps = len(train_loader)
if master_process:
    print(f"It will run {max_steps} steps per epoch.")

 
# --------------------------------------
# Training Loop
# --------------------------------------

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    # Define the new hyperparameter for GSPO's smaller clipping range
    gspo_clip_range = 4e-4 # A typical value from the paper, should be tuned

    for step, (prompts, solutions, tests) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        prompt_enc = tokenizer(
            prompts,
            return_tensors = 'pt',
            padding = True,
            padding_side = 'left',
            truncation = True
        )
        input_ids = prompt_enc["input_ids"].to(local_rank) # (B, prompt_len) and left_pad
        attention_mask = prompt_enc["attention_mask"].to(local_rank)

        # ------------------------------
        # Generate K Samples Per Prompt
        # ------------------------------
        policy_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                explore_generations = policy_model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        num_return_sequences=K,
                        top_p=0.9,
                        temperature=1.0,
                        eos_token_id=tokenizer.eos_token_id
                    ) # (batch_size * K, prompt_len + max_new_tokens)
        policy_model.train()
        prompt_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        # Clear generation cache
        torch.cuda.empty_cache()
        gc.collect()

        # --------------------------------------
        # Compute Masks & Labels
        # --------------------------------------
        batch_attention_mask = (explore_generations != tokenizer.pad_token_id).long() # [batch_size * K, seq_len]
        batch_action_mask = batch_attention_mask.clone()
        batch_action_mask[:, :prompt_len] = 0 # [batch_size * K, seq_len]
        labels = explore_generations.clone() # [batch_size * K, seq_len]
        labels[batch_action_mask == 0] = -100
        
        # --------------------------------------
        # Compute Old Logprobs
        # --------------------------------------
        policy_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_old = policy_model(explore_generations,  batch_attention_mask, use_cache=False)
            logits_old = out_old.logits # [batch_size * K, seq_len, vocab_size]
            del out_old
        policy_model.train()
        logits_old = logits_old[:, :-1, :].contiguous()
        labels_old = labels[:, 1:].contiguous()
        logprobs_old = -F.cross_entropy(
            logits_old.view(-1, logits_old.shape[-1]), labels_old.view(-1), reduction = 'none', ignore_index=-100
        ).view(logits_old.shape[0], -1) # [batch_size * K, seq_len - 1]
        del logits_old, labels_old
        logprobs_old = logprobs_old.view(batch_size, K, -1) # [batch_size, K, seq_len - 1]
        
        # --------------------------------------
        # Compute Reference Logprobs
        # --------------------------------------
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_ref = ref_model(explore_generations,  batch_attention_mask, use_cache=False)
            logits_ref = out_ref.logits # [batch_size * K, seq_len, vocab_size]
            del out_ref
        logits_ref = logits_ref[:, :-1, :].contiguous()
        labels_ref = labels[:, 1:].contiguous()
        logprobs_ref = -F.cross_entropy(
            logits_ref.view(-1, logits_ref.shape[-1]), labels_ref.view(-1), reduction = 'none', ignore_index=-100
        ).view(logits_ref.shape[0], -1) # [batch_size * K, seq_len - 1]
        del logits_ref, labels_ref
        logprobs_ref = logprobs_ref.view(batch_size, K, -1) # [batch_size, K, seq_len - 1]
        
        # --------------------------------------
        # Compute Advantages
        # --------------------------------------
        batch_responses_ids = explore_generations[:, prompt_len:]
        batch_responses = tokenizer.batch_decode(batch_responses_ids, skip_special_tokens = True)
        batch_codes = [extract_xml_answer(batch_responses[i]) for i in range(len(batch_responses))]
        tests_K = [t for t in tests for _ in range(K)]
        assert len(batch_codes) == len(tests_K)
        batch_format_scores = compute_format_score(batch_responses)
        batch_reward_scores = compute_reward(batch_codes, tests_K)
        batch_rewards = torch.tensor([bfs + brs for bfs, brs in zip(batch_format_scores, batch_reward_scores)], dtype=torch.float16)
        batch_rewards = batch_rewards.view(batch_size, K) # (batch_size, K)
        batch_advantages = (batch_rewards - batch_rewards.mean(dim = -1, keepdim = True))
        batch_advantages = batch_advantages.to(local_rank) # (batch_size, K)
        # For GSPO The advantage is a single scalar value per sequence.
        
        # --------------------------------------
        # PPO Update: multiple epochs per batch
        # --------------------------------------
        for ppo_step in range(ppo_steps):
            if ppo_step > 0:
                torch.cuda.empty_cache()
                gc.collect()
            # --------------------------------------
            # Compute New Logprobs
            # --------------------------------------
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_new = policy_model(explore_generations,  batch_attention_mask, use_cache = False)
            logits_new = out_new.logits # [batch_size * K, seq_len, vocab_size]
            del out_new
            logits_new = logits_new[:, :-1, :].contiguous()
            labels_new = labels[:, 1:].contiguous()
            logprobs_new = -F.cross_entropy(
                logits_new.view(-1, logits_new.shape[-1]), labels_new.view(-1), reduction = 'none', ignore_index=-100
            ).view(logits_new.shape[0], -1) # [batch_size * K, seq_len - 1]
            del logits_new, labels_new
            logprobs_new = logprobs_new.view(batch_size, K, -1) # [batch_size, K, seq_len - 1]
            
            # --------------------------------------
            # Compute GSPO loss
            # --------------------------------------
            valid_mask = batch_action_mask[:, :-1].contiguous().float().view(batch_size, K, -1) # [batch_size, K, seq_len - 1]

            # 1. Compute logprob
            seq_logprobs_new = (logprobs_new * valid_mask).sum(dim=-1) # [batch_size, K]
            seq_logprobs_old = (logprobs_old * valid_mask).sum(dim=-1) # [batch_size, K]
            
            # 2. Calculate the new ratio `s_i` with length normalization
            seq_len = valid_mask.sum(dim=-1).clamp_min(1.0) # [batch_size, K]
            ratio = torch.exp((seq_logprobs_new - seq_logprobs_old) / seq_len) # [batch_size, K]

            # 3. Compute gspo reward 
            ratio_clipped = torch.clamp(ratio, 1.0 - gspo_clip_range, 1.0 + gspo_clip_range)
            gspo_reward = torch.min(ratio * batch_advantages, ratio_clipped * batch_advantages) # [batch_size, K]

            # 4. Compute KL
            kl_per_token = torch.exp(logprobs_ref - logprobs_new) - (logprobs_ref - logprobs_new) - 1
            kl_penalty = (kl_per_token * valid_mask).sum(dim=-1) / seq_len # [batch_size, K]

            # 5. Compute the final GSPO loss
            gspo_loss = - (gspo_reward - kl_coef * kl_penalty).mean()
            
            # --------------------------------------
            # Record and log loss
            # --------------------------------------
            loss_ = 0.0
            loss_ += gspo_loss.detach()
            dist.all_reduce(loss_, op=dist.ReduceOp.AVG)
            # if master_process :
            #     print(f'gspo training loss at step {step} with ppo epoch {ppo_step} is: {loss_:.4f}')
            #     with open(log_file, "a") as f:
            #         f.write(f'gspo training loss at step {step} with ppo epoch {ppo_step} is: {loss_:.4f}\n')
            
            # --------------------------------------
            # Optimization Step
            # --------------------------------------
            gspo_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm = max_grad_norm)
            lr = get_lr(step, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
               
        # Checkpoint
        if master_process and (step % 10 == 0 or step == max_steps - 1):
            avg_reward = batch_rewards.mean().item()
            log_message = f"--- Step {step+1} ---\n"
            log_message += f"Average reward: {avg_reward:.4f}\n"
            
            # Log a couple of examples
            log_message += "Example generations:\n"
            for i in range(min(2, len(batch_responses))): # Log at most 2
                log_message += f"--- Example {i+1} ---\n"
                log_message += f"Response:\n{batch_responses[i]}\n"
                log_message += f"Extracted code:\n{batch_codes[i]}\n"
                log_message += f"Format score: {batch_format_scores[i]}, Reward score: {batch_reward_scores[i]}\n"
                log_message += "-" * 20 + "\n"
            
            # print(log_message)
            with open(log_file, "a") as f:
                f.write(log_message)
            # print(log_message)

            # ckpt_dir = f"{log_dir}/qwen-gspo-step{step+1}"
            # policy_model.module.save_pretrained(ckpt_dir)
            # tokenizer.save_pretrained(ckpt_dir)
    
    # Evaluation at the end of each epoch
    avg_reward, avg_format_score, avg_reward_score = evaluate(policy_model, tokenizer, test_loader, local_rank)
    
    if master_process:
        log_message = f"-------------------- Epoch {epoch+1} Evaluation ------------------\n"
        log_message += f"Average Reward: {avg_reward:.4f}\n"
        log_message += f"Average Format Score: {avg_format_score:.4f}\n"
        log_message += f"Average Code Reward: {avg_reward_score:.4f}\n"
        log_message += "-" * 50 + "\n"
        print(log_message)
        with open(log_file, "a") as f:
            f.write(log_message)


dist.barrier()
dist.destroy_process_group()    

# --------------------------------------
# Post-Training Comparison
# --------------------------------------
if master_process:
    print("\n" + "="*60)
    print("POST-TRAINING MODEL COMPARISON")
    print("="*60 + "\n")
    
    # Load one test example
    for test_idx in range(len(test_data)):
        test_example = test_data[test_idx]
        problem_prompt, canonical_solution, test_code = test_example
        
        # Prepare the prompt
        chat_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "def candidate(n):\n    # Check if number is even\n    pass"},
            {'role': 'assistant', 'content': "<reasoning>I need to check if a number is divisible by 2. If n % 2 equals 0, then n is even.</reasoning>\n<answer>\n```python\ndef candidate(n):\n    return n % 2 == 0\n```\n</answer>"},
            {'role': 'user', 'content': problem_prompt}
        ]
        
        prompt = tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("Test Problem:")
        print("-" * 40)
        print(problem_prompt[:900] + "..." if len(problem_prompt) > 900 else problem_prompt)
        print("-" * 40 + "\n")
        
        # Tokenize the prompt
        prompt_enc = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        input_ids = prompt_enc["input_ids"].to(local_rank)
        attention_mask = prompt_enc["attention_mask"].to(local_rank)
        
        # Generate with reference model (old model)
        print("Generating with REFERENCE MODEL (Original)...")
        ref_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_generated = ref_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for deterministic comparison
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                )
        
        ref_response_ids = ref_generated[:, input_ids.shape[1]:]
        ref_response = tokenizer.decode(ref_response_ids[0], skip_special_tokens=True)
        ref_code = extract_xml_answer(ref_response)
        
        # Generate with trained policy model (new model)
        print("Generating with TRAINED MODEL (New)...")
        policy_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                policy_generated = policy_model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for deterministic comparison
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                )
        
        policy_response_ids = policy_generated[:, input_ids.shape[1]:]
        policy_response = tokenizer.decode(policy_response_ids[0], skip_special_tokens=True)
        policy_code = extract_xml_answer(policy_response)
        
        # Compute rewards for both
        ref_format_score = compute_format_score([ref_response], verbose=False)[0]
        ref_reward_score = compute_reward([ref_code], [test_code], verbose=False)[0]
        ref_total_reward = ref_format_score + ref_reward_score
        
        policy_format_score = compute_format_score([policy_response], verbose=False)[0]
        policy_reward_score = compute_reward([policy_code], [test_code], verbose=False)[0]
        policy_total_reward = policy_format_score + policy_reward_score
        
        # Print comparison
        print("\n" + "="*60)
        print("REFERENCE MODEL OUTPUT:")
        print("="*60)
        print("Full Response (including reasoning):")
        print("-" * 40)
        print(ref_response)
        print("-" * 40)
        print(f"Format Score: {ref_format_score}")
        print(f"Code Execution Score: {ref_reward_score}")
        print(f"Total Reward: {ref_total_reward}")
        
        print("\n" + "="*60)
        print("TRAINED MODEL OUTPUT:")
        print("="*60)
        print("Full Response (including reasoning):")
        print("-" * 40)
        print(policy_response)
        print("-" * 40)
        print(f"Format Score: {policy_format_score}")
        print(f"Code Execution Score: {policy_reward_score}")
        print(f"Total Reward: {policy_total_reward}")
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY:")
        print("="*60)
        print(f"Reference Model Total Reward: {ref_total_reward}")
        print(f"Trained Model Total Reward: {policy_total_reward}")
        print(f"Improvement: {policy_total_reward - ref_total_reward}")
        
        # Log to file
        with open(log_file, "a") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("POST-TRAINING MODEL COMPARISON\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Problem:\n{problem_prompt}\n\n")
            f.write("REFERENCE MODEL OUTPUT:\n")
            f.write(f"Response: {ref_response}\n")
            f.write(f"Rewards: Format={ref_format_score}, Code={ref_reward_score}, Total={ref_total_reward}\n\n")
            f.write("TRAINED MODEL OUTPUT:\n")
            f.write(f"Response: {policy_response}\n")
            f.write(f"Rewards: Format={policy_format_score}, Code={policy_reward_score}, Total={policy_total_reward}\n\n")
            f.write(f"Improvement: {policy_total_reward - ref_total_reward}\n")
            f.write("="*60 + "\n")

