import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

pretrained_model_name_or_path = "bavest/fin-llama-33b-merged"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    load_in_4bit=True,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)


def prompt(q, i):
    if i == "":
        return f"""I want you to act as an finance analyst. I will ask you questions about finance and you will answer them based on the
        provided input.
           '### Instruction:\n{q}\n\n### Response:
           """

    return f"""
    A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's question.
    '### Instruction:\n{q}\n\n### Input:\n{i}\n\n### Response:
    """

while True:
    q = input("> ")
    i = ""

    input_ids = tokenizer.encode(prompt(q, i), return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            max_length=128,
        )

    generated_text = tokenizer.decode(
        [el.item() for el in generated_ids[0]], skip_special_tokens=True
    )

    print(generated_text)
