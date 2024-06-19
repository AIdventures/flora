"""
This file contains utility functions to clean completions from the model
"""

import re


def clean_completion(text: str, eos_token: str, problem_statement: str) -> str:
    """Given a code completion, clean it by removing the problem statement and not indented lines.

    Args:
        text (str): The completion to clean.
        eos_token (str): The token to cut the text at.
        problem_statement (str): The problem statement to remove.

    Returns:
        str: The cleaned completion.
    """
    # cut the text at tokenizer.eos_token
    text = text[: text.find(eos_token)]

    # remove the problem statement
    if text.startswith(problem_statement):
        text = text[len(problem_statement) :]

    # the text should have indentation as is part of a function
    # if in a new line it not indented, remove it till the end
    aux = ""
    for line in text.split("\n"):
        # if the line is not empty, and not indented, break
        if line and not line.startswith(" "):
            break
        aux += line + "\n"

    # remove all after first return line
    res = ""
    for line in aux.split("\n"):
        res += line + "\n"
        if line.strip().startswith("return"):
            break

    # if exists, clean from the end the \n
    if res.endswith(" \n"):
        res = res[:-2]

    return res


def parse_int_with_commas(number_str):
    """
    Parse a string representing an integer that includes commas as thousands separators
    and remove any non-numeric characters to the right.

    Args:
    number_str (str): The string to parse.

    Returns:
    int: The parsed integer.

    Raises:
    ValueError: If the input string is not a valid integer.

    Examples:
    >>> parse_int_with_commas("1,000")
    1000
    >>> parse_int_with_commas("1,000,000")
    1000000
    >>> parse_int_with_commas("1,000,000.00")
    1000000
    """
    try:
        # Remove everything from the first non-numeric character
        numeric_part = ""
        for char in number_str:
            if char.isdigit() or char == ",":
                numeric_part += char
            else:
                break

        # Remove commas from the numeric part
        clean_str = numeric_part.replace(",", "")
        # Convert the cleaned string to an integer
        result = int(clean_str)
        return result
    except ValueError as e:
        # Raise a ValueError if the string is not a valid integer
        raise ValueError(
            f"Invalid literal for int() with base 10: '{number_str}'"
        ) from e


def get_first_number(text: str) -> int:
    """Extract the first number from a string.

    Args:
        text (str): The text to extract the number from.

    Returns:
        str: The first number found in the text.
    """
    # Pattern explanation:
    # \d{1,3}(?:,\d{3})* - Matches numbers with commas for thousands
    # (?:\.\d+)? - Optional decimal part
    pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"

    # Find all matches in the string
    matches = re.findall(pattern, text)

    # Return the first match or empty if no match is found
    return parse_int_with_commas(matches[0]) if matches else -1


def inference(
    text,
    model,
    tokenizer,
    max_input_tokens=1000,
    max_output_tokens=256,
    temperature=0.75,
):
    """
    Generate text from a prompt using a model and tokenizer.

    Args:
        text (str) or (List<str>): The prompt to generate text from.
        model (transformers.PreTrainedModel): The model to generate text with.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        max_input_tokens (int): The maximum number of tokens to use as input.
        max_output_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    # Tokenize
    # Tokenize
    input_ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_input_tokens
    ).to(model.device)

    # Generate
    generated_tokens_with_prompt = model.generate(
        **input_ids,
        max_length=max_output_tokens,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(
        generated_tokens_with_prompt, skip_special_tokens=True
    )

    # Strip the prompt
    # generated_text_answer = generated_text_with_prompt[0][len(text):]

    # if the input was a string, return a string
    if isinstance(text, str):
        return generated_text_with_prompt[0]

    return generated_text_with_prompt
