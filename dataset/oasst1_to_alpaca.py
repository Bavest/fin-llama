import json


def to_alpaca(instruction, input, output):
    return {
        "instruction": instruction,
        "input": input,
        "output": output,
        "text": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
    }


def read_jsonl():
    with open("../fin-llama-dataset/train.jsonl", "r") as f:
        return [json.loads(line) for line in f.readlines()]

if __name__ == "__main__":

    dataset = read_jsonl()
    for data_entry in dataset:
        j = data_entry["text"]
        input = j.split("### Input:")[1].split("###")[0].strip() if "### Input:" in j else ""
        instruction = j.split("### Human:")[1].split("###")[0].strip()
        output = j.split("### Assistant:")[1].split("###")[0].strip()

        with open(f'train.jsonl', 'a') as f:
            f.write(json.dumps(to_alpaca(instruction, input, output)) + "\n")

