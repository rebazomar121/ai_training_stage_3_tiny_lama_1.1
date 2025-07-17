from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def chat():
    # Load the base model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_path = "./personal_model_lora"

    # Load the base model in 16-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)

    print("Your personalized AI is ready! Type 'exit' to end the conversation.")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Format the input using the prompt template
        prompt = f"Question: {user_input}\nAnswer:"

        # Tokenize the input and move it to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate a response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The model will output the prompt as well, so we need to remove it.
        response = response.replace(prompt, "").strip()
        print(f"Personal AI: {response}")


if __name__ == "__main__":
    chat()
