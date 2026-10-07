import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase


def get_wikitext2(split: str = None) -> datasets.DatasetDict:
    ds = datasets.load_dataset(
        "wikitext", name="wikitext-2-raw-v1", split=split
    )
    return ds


def get_gsm8k_text_dataset(split: str = "test") -> datasets.Dataset:
    """GSM8K examples as a single text column (question + answer)."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
    texts = [f"{row['question']}\n{row['answer']}" for row in ds]
    return datasets.Dataset.from_dict({"text": texts})


CALIB_SOURCES = ("wikitext", "gsm8k")


def get_calib_dataset(name: str, split: str = "train") -> datasets.Dataset:
    """Calibration corpus as a single text column. Short examples are packed
    to max_seqlen by prepare_dataloader."""
    if name == "wikitext":
        return get_wikitext2(split=split)
    if name == "gsm8k":
        return get_gsm8k_text_dataset(split=split)
    raise ValueError(f"Unknown calibration source {name!r}; expected one of {CALIB_SOURCES}")


def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Build a DataLoader of fixed-length calibration samples.

    Each sample starts at a random example and joins the following examples
    (separated by blank lines) until it reaches max_seqlen tokens, then is cut
    to exactly max_seqlen. A sample that runs out of text first is dropped.

    Args:
        dataset: Dataset whose first column holds the text.
        tokenizer: Tokenizer for the model being calibrated.
        max_seqlen: Tokens per sample.
        batch_size: Samples per batch.
        nsamples: Number of samples to build.
        seed: Seed for picking start points and for the sampling order.

    Returns:
        A DataLoader.
    """

    data_name = dataset.column_names[0]
    # Drop empty rows.
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
    data_list = ds[data_name]
    new_data_list = []

    torch.manual_seed(seed)
    indices = list(range(len(data_list)))

    while len(new_data_list) < nsamples and len(indices) > 0:
        start_idx = torch.randint(0, len(indices), (1,)).item()
        idx = start_idx
        tokens = []
        while len(tokens) < max_seqlen and idx < len(indices):
            item = data_list[indices[idx]]
            sep = "" if not tokens else "\n\n"
            tokens += tokenizer.tokenize(sep + item, add_special_tokens=False)
            idx += 1

        indices = indices[:start_idx] + indices[idx:]  # remove the used indices

        if len(tokens) >= max_seqlen:
            tokens = tokens[:max_seqlen]  # truncate to max_seqlen
            decoded = tokenizer.convert_tokens_to_string(tokens)
            new_data_list.append(decoded)
    ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return loader
