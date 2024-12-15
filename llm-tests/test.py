# %%
import torch
import multiprocessing
import os

# Set multiprocessing start method for CUDA compatibility
multiprocessing.set_start_method("spawn", force=True)

# Set CUDA_VISIBLE_DEVICES for specific GPUs if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPUs 0 and 1

def device_count():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device_count()

# Check GPU memory before starting
print("Initial GPU memory usage:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e6} MB")
    print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1e6} MB")

# Clear CUDA cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# %%
from accelerate import Accelerator

# DeepSpeed configuration for ZeRO Stage 2
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    }
}

accelerator = Accelerator(mixed_precision="fp16", deepspeed_plugin=deepspeed_config)

# %%
hf_token = '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'

# %%
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

modelName = "google/gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(modelName, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(modelName, 
                                             quantization_config=bnb_config, 
                                             device_map="auto",
                                             token=hf_token)

# Prepare model with accelerator
model = accelerator.prepare(model)

# %%
print("Device map of model layers:")
for name, param in model.named_parameters():
    print(f"{name} is on {param.device}")

# %%
from datasets import load_dataset
dataset = load_dataset("myzens/alpaca-turkish-combined", split="train")
dataset, dataset[0]

# %%
gemma_prompt = """<start_of_turn>user
{}: {}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""
gemma_prompt

# %%
eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
tokenizer.padding_side = "right"

eos_token, pad_token

# %%
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = gemma_prompt.format(instruction, input, output) + eos_token
        texts.append(text)
    return { "text" : texts, }
pass

# %%
dataset = dataset.map(formatting_prompts_func, batched = True)
dataset

# %%
print(dataset["text"][2])

# %%
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    # Labels are identical to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

print("Tokenizing dataset...")
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenized:", dataset[0])

# %%
from torch.utils.data import DataLoader

def collate_fn(batch):
    input_ids = torch.tensor([example["input_ids"] for example in batch])
    attention_mask = torch.tensor([example["attention_mask"] for example in batch])
    labels = torch.tensor([example["labels"] for example in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)
train_dataloader = accelerator.prepare(train_dataloader)

# %%
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
optimizer = accelerator.prepare(optimizer)

# %%
from transformers import get_scheduler

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=30,
    num_training_steps=len(train_dataloader) * 3,  # 3 epochs
)

# %%
import time
model.train()

start_time = time.time()
for epoch in range(3):
    for step, batch in enumerate(train_dataloader):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds")

# %%
# Monitor GPU usage during training
print("Final GPU memory usage after training:")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e6} MB")
    print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1e6} MB")
