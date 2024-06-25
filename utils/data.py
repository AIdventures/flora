import re
from datasets import load_dataset, DatasetDict


def string_cleaner(original_string: str) -> str:
    """
    Removes \n and more than one space from a string,
    maintaining multiple \n\n

    Args:
        string (str): String to be cleaned

    Returns:
        str: Cleaned string
    """
    # Remove leading and trailing spaces
    cleaned_string = original_string.strip()

    # Replace \n\n\n with \n\n
    cleaned_string = cleaned_string.replace("\n\n\n", "\n\n")

    # Replace multiple spaces with a single space
    cleaned_string = re.sub(r" +", " ", cleaned_string)

    # Replace simple \n (not part of \n\n) with a single space
    cleaned_string = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned_string)

    # Restore any \n\n that were turned into single spaces
    cleaned_string = cleaned_string.replace("\n ", "\n").replace(" \n", "\n")

    return cleaned_string


def get_tokenize_function(tokenizer, max_seq_len: int = 1024):
    def tokenize_function(examples):
        # Tokenize the prompts
        tokenized_outputs = tokenizer(
            examples["prompt"],  # truncation=True, max_length=max_seq_len, padding=True
        )  # , padding='max_length'

        # Set labels to input_ids. This assumes a task like text generation where
        # the model learns to predict the input sequence itself (next word).
        # You don’t need labels (also known as an unsupervised task)
        # because the next word is the label
        # shifting is handled inside the model. ‘input_ids’ and ‘labels’ can be very same tensors, however the model will do a ‘causal-shift’ inside.
        # https://discuss.huggingface.co/t/how-is-the-data-shifted-by-one-token-during-causallm-fine-tuning/36386/3
        tokenized_outputs["labels"] = tokenized_outputs["input_ids"].copy()

        # Finally return the tokenized outputs, they will be the 'input_ids'
        return tokenized_outputs

    return tokenize_function


def get_mnli(
    tokenizer, validation_size: float = 0.05, max_seq_len: int = 1024
) -> DatasetDict:
    """Get the preprocessed MNLI dataset"""

    """Preprocess the MNLI dataset to be used"""

    """We need to transform MNLI's data into a prompt-completion format."""
    PROMPT_TEMPLATE = string_cleaner("""
    You are given a premise and a hypothesis below. If the premise entails the
    hypothesis, return 0. If the premise contradicts the hypothesis, return 2.
    Otherwise, if the premise does neither, return 1.


    ### Premise: {premise}


    ### Hypothesis: {hypothesis}


    ### Label: {label}
    """)

    # Data Load
    mnli_dataset = load_dataset("glue", "mnli")

    # HumanEval Specific Preprocessing for Training Data
    def _preprare_train_prompt_completion(example):
        example["prompt"] = PROMPT_TEMPLATE.format(
            premise=example["premise"],
            hypothesis=example["hypothesis"],
            label=example["label"],
        )

        example["prompt_length"] = len(example["prompt"]) - len(str(example["label"]))

        example["premise"] = example["premise"]
        example["hypothesis"] = example["hypothesis"]
        example["label"] = example["label"]
        example["idx"] = example["idx"]

        return example

    # HumanEval Specific Preprocessing for Validation & Test Data
    def _preprare_test_prompt_completion(example):
        example["prompt"] = PROMPT_TEMPLATE.format(
            premise=example["premise"],
            hypothesis=example["hypothesis"],
            label="",  # We don't know the label for the test data
        )

        example["prompt_length"] = len(example["prompt"])

        example["premise"] = example["premise"]
        example["hypothesis"] = example["hypothesis"]
        example["label"] = example["label"]
        example["idx"] = example["idx"]

        return example

    # Split the dataset into train and validation
    split_dataset = mnli_dataset["train"].train_test_split(
        test_size=validation_size, shuffle=True, seed=42
    )

    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    # dataset = mnli_dataset.map(_preprare_prompt_completion)
    # prepare the train partition
    train_dataset = train_dataset.map(_preprare_train_prompt_completion)

    # prepare the validation and test partitions
    validation_dataset = validation_dataset.map(_preprare_test_prompt_completion)
    validation_matched_dataset = mnli_dataset["validation_matched"].map(
        _preprare_test_prompt_completion
    )
    validation_mismatched_dataset = mnli_dataset["validation_mismatched"].map(
        _preprare_test_prompt_completion
    )

    # join the datasets
    base_dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
            "test_matched": validation_matched_dataset,  # used for test
            "test_mismatched": validation_mismatched_dataset,  # used for test
        }
    )

    # Use a lambda function to wrap the tokenize function with the specified arguments
    tokenize_function_with_args = get_tokenize_function(tokenizer, max_seq_len)

    tokenized_dataset = base_dataset.map(tokenize_function_with_args, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    dataset = DatasetDict(
        {
            "train": tokenized_dataset["train"],
            "validation": tokenized_dataset["validation"],
            "test_matched": tokenized_dataset["test_matched"],
            "test_mismatched": tokenized_dataset["test_mismatched"],
        }
    )

    # Remove columns that can't be converted to tensors
    dataset = dataset.remove_columns(
        [
            "premise",
            "hypothesis",
            "prompt",
        ]
    )
    # Rename the label column to class_label
    dataset = dataset.rename_column("label", "class_label")

    return dataset
