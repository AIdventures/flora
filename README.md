# FLoRA: Fine-tuning LLMs with LoRA

![Welcome Illustration](assets/tranquil_riverbank.jpg "Welcome Illustration")


This repository contains the code for fine-tune large language models (LLMs) with [LoRA](https://aidventure.es/blog/lora/). LoRA is a simple yet effective way to fine-tune LLMs using low-rank decomposition, lowering the computational cost and memory requirements of fine-tuning. The aim of this repository is to provide a simple and easy-to-use implementation of LoRA for fine-tuning LLMs. To test the implementation, we will tackle the MNLI subtask of the [GLUE](https://huggingface.co/datasets/nyu-mll/glue) benchmark.


# Index

1. [Project Overview](#1.-Project-Overview)
2. [The Process](#2.-The-Process)
    1. [Data Preprocessing](#2.1.-Data-Preprocessing)
    2. [Model Load and Preparation](#2.2.-Model-Load-and-Preparation)
    3. [Fine-tuning](#2.3.-Fine-tuning)
    4. [Logging and Evaluation](#2.4.-Logging-and-Evaluation)
3. [Installation](#3.-Installation)
4. [Results](#4.-Results)
5. [Conclusion](#5.-Conclusion)

# 1. Project Overview

```
├── utils          <- Utility functions
│   ├── args.py          <- Arguments parser
│   ├── completions.py   <- Completions generation
│   ├── data.py          <- Data pre/post-processing
│   └── evaluation.py    <- Evaluation functions
|
└── notebooks  <- Jupyter notebooks
    ├── eda.ipynb        <- Exploratory Data Analysis
    ├── baseline.ipynb   <- Evaluate the model before fine-tuning
    ├── training.ipynb   <- Fine-tune the model using LoRA
    └── evaluation.ipynb <- Evaluate trained model
```

# 2. The Process

The process of fine-tuning LLMs with LoRA can be divided into the following steps:

- **Data Preprocessing**: Prompting, preparing the data into a consistent format, and creating the dataloaders.
- **Model Load and Preparation**: Load the pre-trained model and prepare it for fine-tuning. LoRA and Quantization are applied to the model.
- **Fine-tuning**: Fine-tune the model on the MNLI dataset.
- **Logging and Evaluation**: Log the training process and evaluate the model on the validation set.

## 2.1. Data Preprocessing

The data preprocessing step involves preparing the data into a consistent format and creating the dataloaders. The MNLI dataset is used for fine-tuning the model. The data is preprocessed using the `datasets` library from Hugging Face.
The code for this part is mainly in the [data.py](utils/data.py) file. The first step is to load the dataset from the `datasets` library. Then, we prepare the train, validation, and test sets by using a **prompt template**, a string placeholder that contains the placeholders for the premise and hypothesis. 

```python
PROMPT_TEMPLATE = """
You are given a premise and a hypothesis below. If the premise entails the
hypothesis, return 0. If the premise contradicts the hypothesis, return 2.
Otherwise, if the premise does neither, return 1.

### Premise: {premise}

### Hypothesis: {hypothesis}

### Label: {label}
"""
```

Note that for the validation and test sets, the `label` is not provided, as it is the target of the model and is used to evaluate the model's performance. The next step is to tokenize the data using the tokenizer provided by the model. The tokenized data is then converted into input_ids, attention_mask, and labels. The input_ids are the tokenized input, the attention_mask is a binary mask that indicates which tokens are part of the input, and the labels are the target labels for the model.

## 2.2. Model Load and Preparation

The model load and preparation step involve loading the pre-trained model and preparing it for fine-tuning. LoRA and Quantization are applied to the model. The code for this part is mainly in the [training.py](notebooks/training.ipynb) file. The first step is to load the pre-trained model using the `AutoModelForCausalLM` class from the `transformers` library. The model is then prepared for fine-tuning by applying Quantization and LoRA.

```python
model_name = "microsoft/phi-2"

# Load the base quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"  #{"":0},
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
```

## 2.3. Fine-tuning

The fine-tuning step involves fine-tuning the model on the MNLI dataset. The code for this part is mainly in the [training.py](notebooks/training.ipynb) file. The model is fine-tuned using standard training loops with the AdamW optimizer and a cosine learning rate scheduler. The model is trained for a fixed number of epochs, and the training and validation losses are logged.

```python
for iter_num in range(training_iters):
    
    model.train()
    for micro_step in range(gradient_accumulation_steps):
        # Extract a batch of data
        batch = next(iter(train_dataloader))
        # remove from batch keys that are not needed
        train_batch = {k: v for k, v in batch.items() if k in forward_keys}

        outputs = model(**train_batch)
        # El modelo calcula su loss, pero podriamos acceder a los logits del modelo
        # y las labels del batch y calcular nuestra loss propia
        # scale the loss to account for gradient accumulation
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if iter_num % 50 == 0:
        train_loss = loss.item() * gradient_accumulation_steps
        print(f"### ITER {iter_num} ###")
        print(f"Train Loss: {train_loss:.4f}")
        wandb.log({
            "iter": iter_num,
            "train/loss": train_loss,
            "lr": lr_scheduler.get_last_lr()[0],
        })   
```

## 2.4. Logging and Evaluation

The logging is performed using the `wandb` library, which allows for easy logging of metrics and visualizations. The metrics are logged during training and are embedded in the training loop from the previous step. The evaluation is performed on the validation set after training is completed. The code for this part is mainly in the [evaluation.py](notebooks/evaluation.ipynb) file. 

For the evaluation, the prompt is used without the label and the output is decoded with the tokenizer to get the predictions. The predictions are then **post-processed** to get the metrics. The post-process mainly aims to find the first integer in the output of the model, which is the prediction. The predictions are then compared to the labels to get the accuracy and other metrics.


# 3. Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

# 4. Results

The results are obtained by comparing the predictions of the model with the labels of the MNLI validation set, which was not seen during training. Remember that the MNLI dataset has a test partition, but is not labeled.
The metric used to evaluate the model is the accuracy, which is the percentage of correct predictions. The results are shown in the table below:

| Method     | Validation Matched | Validation MisMatched |
|------------|--------------------|-----------------------|
| Baseline   | 0.2101             | 0.2291                |
| Fine-tuned | 0.8561             | 0.8567                |

The results show that the fine-tuned model significantly outperforms the baseline model on the MNLI dataset. The fine-tuned model achieves an average accuracy of 85.6% on the validation set, compared to 2.5% for the baseline model. This demonstrates the effectiveness of fine-tuning LLMs with LoRA for the MNLI task.


# 5. Conclusion

As special mention over all the steps in the [process](#2.-the-process), **logging and evaluation** are crucial to understand the performance of the model and the training process, as well as one of the most problematic parts of the process. For MNLI, we need to clean the output of the model to get the predictions and the metrics. Given a premise and a hypothesis, the model has to return 0 if the premise entails the hypothesis, 2 if the premise contradicts the hypothesis, and 1 otherwise. Without long fine-tuning the models tend to return verbose outputs such as "the label is X", or "the output is X", which could led to valid solutions. To take this into account we need to clean the output of the model, by searching the first integer, to get the predictions and the metrics.

Another interesting point is how we send the full sentence, with the solution, as `input_ids` and `labels`(check the [data.py](utils/data.py) file). This is done due to the nature of the training objective: next token prediction. The model is trained to predict the next token given the previous tokens, so we need to send the full sentence to make the model learn the whole sentence. To predict the next token, the sentences need to be **shifted** by one token, which is [done automatically](https://discuss.huggingface.co/t/how-is-the-data-shifted-by-one-token-during-causallm-fine-tuning/36386/3) by the `transformers` library.