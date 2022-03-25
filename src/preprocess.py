import torch
import numpy as np
import random
from copy import deepcopy


def corrupt_spans(text, mask_ratio=0.15):
    """T5-style Masked Language Modeling with corrupted span prediction
    Args:
        text

    Returns:
        source_text (masked_text)
        target_text

    Ex) (in vocab ids)
    input
        In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.

    masked_text
        <extra_id_0> this tutorial, we’ll explore how to preprocess your data <extra_id_1> Transformers. The main tool for this is what <extra_id_2> call a tokenizer.
    target_text
        <extra_id_0> In <extra_id_1> using <extra_id_2> we
    """

    tokens = text.split()

    n_tokens = len(tokens)

    n_mask = int(max(mask_ratio * n_tokens, 1))
    mask_indices = torch.randperm(n_tokens)[:n_mask].sort().values

    assert len(mask_indices) > 0, text

    mask_indices = mask_indices.tolist()
    span = [mask_indices[0], mask_indices[0]+1]
    spans = []

    for i, mask_index in enumerate(mask_indices):
        # if current mask is not the last one & the next mask is right after current mask
        if i < len(mask_indices) - 1 and mask_indices[i+1] == mask_index + 1:
            contiguous = True
        else:
            contiguous = False

        if contiguous:
            span[1] += 1

        else:
            # non contiguous -> output current span
            spans.append(span)
            # if current mask is not the last one -> create next span
            if i < len(mask_indices) - 1:
                span = [mask_indices[i+1], mask_indices[i+1]+1]

    masked_tokens = deepcopy(tokens)

    target_tokens = []
    cum_span_length = 0
    for i, span in enumerate(spans):
        start, end = span

        masked_tokens[start-cum_span_length+i: end -
                      cum_span_length+i] = [f'<extra_id_{i}>']

        target_tokens.append(f'<extra_id_{i}>')
        target_tokens.extend(tokens[start:end])

        cum_span_length += (end - start)

    # target_tokens.append(f'<extra_id_{i+1}>')
    # target_tokens.append(f'</s>')

    masked_text = " ".join(masked_tokens)
    source_text = masked_text

    target_text = " ".join(target_tokens)

    return source_text, target_text


def corrupt_prefix(input_text):
    """T5-style Prefix Language Modeling
    Args:
        text

    Returns:
        source_text (prefix)
        target_text

    Ex) (in vocab ids)
    input
        In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.

    source text
        this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool
    target_text
        for this is what we call a tokenizer.
    """

    tokens = input_text.split()

    n_tokens = len(tokens)
    split = random.randint(1, n_tokens-1)
    source_text = " ".join(tokens[:split])
    target_text = " ".join(tokens[split:])

    return source_text, target_text

