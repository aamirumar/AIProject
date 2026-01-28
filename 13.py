import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# -------------------------------------------------
# Step 1: Load Dataset (FIXED)
# -------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("karpathy/tiny_shakespeare")

# Use small subset for fast training
train_data = dataset["train"].select(range(2000))
test_data = dataset["train"].select(range(2000, 2500))

# -------------------------------------------------
# Step 2: Load GPT-2 Tokenizer and Model
# -------------------------------------------------
print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # REQUIRED FIX

model = GPT2LMHeadModel.from_pretrained("gpt2")

# -------------------------------------------------
# Step 3: Tokenization
# -------------------------------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

print("Tokenizing dataset...")
tokenized_train = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_test = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# -------------------------------------------------
# Step 4: Data Collator
# -------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------------------------------
# Step 5: Training Arguments
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,     # FAST & SAFE
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=50,
    report_to="none"
)

# -------------------------------------------------
# Step 6: Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator
)

# -------------------------------------------------
# Step 7: Train the Model
# -------------------------------------------------
print("Starting training...")
trainer.train()

# -------------------------------------------------
# Step 8: Save the Model
# -------------------------------------------------
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("Model saved to './fine_tuned_gpt2'")

# -------------------------------------------------
# Step 9: Generate Text
# -------------------------------------------------
print("\nGenerating text...\n")

model.eval()
input_text = "To be or not to be"
inputs = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(
    inputs,
    max_length=80,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

