import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load pre-trained transformer model and tokenizer
print("Loading the DialoGPT model...")

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()  # Set model to evaluation mode

# Step 2: Initialize chat history
chat_history_ids = None
step = 0

print("Chatbot is ready! Type 'exit' to end the chat.\n")

# Step 3: Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Encode user input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Append to chat history
    if step == 0:
        chat_history_ids = new_input_ids
    else:
        chat_history_ids = torch.cat(
            [chat_history_ids, new_input_ids],
            dim=-1
        )

    # Generate chatbot response
    response_ids = model.generate(
        chat_history_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode response
    response = tokenizer.decode(
        response_ids[:, chat_history_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print("Chatbot:", response)
    step += 1

