from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline
)

# Load the model and tokenizer
model_ckpt = "TheBloke/Llama-2-13B-chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_ckpt, device_map="auto", trust_remote_code=False,
    revision="main"
)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)

# Load the generator pipeline
generator = pipeline(
    "text-generation", model=model,
    tokenizer=tokenizer, max_new_tokens=512,
    do_sample=False
)

# Format the prompt
prompt = (
    "1. A bat and a ball cost $1.10 in total. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost?\n"
    "2. If it takes 5 machines 5 minutes to make 5 widgets, "
    "how long would it take 100 machines to make 100 widgets?\n"
    "3. In a lake, there is a patch of lily pads. "
    "Every day, the patch doubles in size. "
    "If it takes 48 days for the patch to cover the entire lake, "
    "how long would it take for the patch to cover half of the lake?\n"
)
prompt_template = (
    f"[INST] <<SYS>>\n"
    "You are about to participate in a psychology experiment with three questions. "
    "Please take your time to consider your answer to each question, "
    "and provide a short answer.\n"
    f"<</SYS>>\n{prompt}[/INST]\n"
)

# Generate and print the output
output = generator(prompt_template)
print(output[0]['generated_text'][len(prompt_template):])
