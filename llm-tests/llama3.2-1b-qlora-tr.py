# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
from peft import LoraConfig
from transformers import BitsAndBytesConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

modelName = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForCausalLM.from_pretrained(modelName, quantization_config=bnb_config, device_map="auto")

# %%
from datasets import load_dataset
dataset = load_dataset("myzens/alpaca-turkish-combined", split="train")
dataset, dataset[0]

# %%
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token

tokenizer.pad_token_id = 128002
pad_token = tokenizer.pad_token

# %%
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = bos_token + alpaca_prompt.format(instruction, input, output) + eos_token
        texts.append(text)
    return { "text" : texts, }
pass

# %%
dataset = dataset.map(formatting_prompts_func, batched = True)

# %%
print(dataset["text"][0])

# %%
from transformers import TrainingArguments

train_args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 150,
        num_train_epochs = 1,
        gradient_checkpointing = True,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 250,
        optim = "adamw_hf",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        output_dir = "llama3.2-1b-tr",
)

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    args = train_args,
    peft_config = lora_config,
    train_dataset = dataset,
    dataset_text_field = "text",
    packing = False,
)
trainer.train()

# %%
model.push_to_hub("emre570/llama3.2-1b-tr-qlora")
tokenizer.push_to_hub("emre570/llama3.2-1b-tr-qlora")