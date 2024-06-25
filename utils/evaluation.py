import torch
from tqdm.auto import tqdm
from pydantic import BaseModel
from utils.completions import get_first_number


class EvaluationItem(BaseModel):
    """A class to store the evaluation item."""

    case_index: int  # The index of the case
    case_text: str  # The text of the case
    generation: str  # The generated text
    y_true: int  # The true label
    y_pred: int  # The predicted label


# Evaluation function
def evaluate(
    model,
    dataloader,
    tokenizer,
    prompt_key: str = "input_ids",
    max_output_tokens: int = 64,
    temperature: float = 0.0,
) -> list[EvaluationItem]:
    """Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader to evaluate the model on.
        tokenizer (transformers Tokenizer): The tokenizer to use.
        prompt_key (str): The key to access the prompt input in the batch.
        max_output_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use for generation.

    Returns:
        list[EvaluationItem]: A list of evaluation items
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            generated_tokens_with_prompt = model.generate(
                batch[prompt_key],
                max_new_tokens=max_output_tokens,
                pad_token_id=tokenizer.pad_token_id,
                temperature=temperature,
            )

            # Decode the generated tokens
            generated_text_with_prompt = tokenizer.batch_decode(
                generated_tokens_with_prompt, skip_special_tokens=True
            )

            for batch_index, generation_with_prompt in enumerate(
                generated_text_with_prompt
            ):
                case_text = tokenizer.decode(batch["input_ids"][batch_index])
                # remove eos_token/pad_token from texts
                case_text = case_text.replace(tokenizer.pad_token, "")
                y_true = batch["class_label"][batch_index].item()
                generation = generation_with_prompt[len(case_text) :]
                y_pred = get_first_number(generation)
                results.append(
                    EvaluationItem(
                        case_index=batch["idx"][batch_index].item(),
                        case_text=case_text,
                        generation=generation,
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                )

    return results
