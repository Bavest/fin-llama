![alt text](assets/finlama.png)

# FIN-LLAMA

> Efficient Finetuning of Quantized LLMs for Finance

[Adapter Weights](https://huggingface.co/bavest/fin-llama-33b-merged)
|  [Dataset](https://huggingface.co/datasets/bavest/fin-llama-dataset)

## Installation

To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source
and make sure you have the latest version of the bitsandbytes library (0.39.0).

```bash
pip3 install -r requirements.txt
```

### Other dependencies

If you want to finetune the model on a new instance. You could run
the `setup.sh` to install the python and cuda package.

```bash
bash scripts/setup.sh
```

## Finetuning

```bash
bash script/finetune.sh
```

## Usage

Quantization parameters are controlled from the `BitsandbytesConfig`

- Loading in 4 bits is activated through `load_in_4bit`
- The datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- Nested quantization is activated through `bnb_4bit_use_double_quant`
- The datatype used for qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported
  quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal
  for normally distributed weights and we recommend using `nf4`.

```python
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

question = "What is the market cap of apple?"
input = "" # context if needed

prompt = f"""
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's question.
'### Instruction:\n{question}\n\n### Input:{input}\n""\n\n### Response: 
"""

input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        max_length=128
    )

generated_text = tokenizer.decode(
    [el.item() for el in generated_ids[0]], skip_special_tokens=True
)
```

## Dataset for FIN-LLAMA

The dataset is released under bigscience-openrail-m.
You can find the dataset used to train FIN-LLAMA models on HF
at [bavest/fin-llama-dataset](https://huggingface.co/datasets/bavest/fin-llama-dataset).

## Known Issues and Limitations

Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the
problem.
See [QLORA](https://github.com/artidoro/qlora) for any other limitations.

1. 4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix
   multiplication
2. Currently, using `bnb_4bit_compute_type='fp16'` can lead to instabilities.
3. Make sure that `tokenizer.bos_token_id = 1` to avoid generation issues.

## Acknowledgements

We also thank Meta for releasing the LLaMA models without which this work would not have been possible.

This repo builds on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
, [QLORA](https://github.com/artidoro/qlora), [Chinese-Guanaco](https://github.com/jianzhnie/Chinese-Guanaco/tree/main)
and [LMSYS FastChat](https://github.com/lm-sys/FastChat) repos.

## License and Intended Use
We release the resources associated with QLoRA finetuning in this repository under GLP3 license. In addition, we release the FIN-LLAMA model family for base LLaMA model sizes of 7B, 13B, 33B, and 65B. These models are intended for purposes in line with the LLaMA license and require access to the LLaMA models.

## Prompts 
### Act as an Accountant
> I want you to act as an accountant and come up with creative ways to manage finances. You'll need to consider budgeting, investment strategies and risk management when creating a financial plan for your client. In some cases, you may also need to provide advice on taxation laws and regulations in order to help them maximize their profits. My first suggestion request is “Create a financial plan for a small business that focuses on cost savings and long-term investments".

## Paged Optimizer
You can access the paged optimizer with the argument --optim paged_adamw_32bit

## Cite

```tex
@misc{Fin-LLAMA,
  author = {William Todt, Ramtin Babaei, Pedram Babaei},
  title = {Fin-LLAMA: Efficient Finetuning of Quantized LLMs for Finance},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Bavest/fin-llama}},
}
```
