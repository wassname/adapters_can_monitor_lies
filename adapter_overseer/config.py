from simple_parsing import Serializable, field
from dataclasses import InitVar, dataclass, replace
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent


@dataclass
class ExtractConfig(Serializable):
    """Config for extracting hidden states from a language model."""

    datasets: tuple[str, ...] = ("amazon_polarity", "glue:qnli")
    """datasets to use, e.g. `"super_glue:boolq"` or `"imdb"` `"glue:qnli` super_glue:rte super_glue:axg sst2 hans"""

    datasets_ood: tuple[str, ...] = ( 'imdb', "super_glue:boolq")
    """Out Of Distribution datasets to use, e.g. `"super_glue:boolq"` or `"imdb"` `"glue:qnli"""
    
    model: str = "failspy/Llama-3-8B-Instruct-abliterated"

    collection_layers: tuple[str, ...] = ("base_model.model.model.layers.10", "base_model.model.model.layers.20", )
    # """Names of layers to extract from using baukit.nethook.TraceDict"""

    batch_size: int = 4

    prompt_format: str | None = None
    """if the tokenizer does not have a chat template you can set a custom one. see src/prompts/templates/prompt_formats/readme.md."""
    
    num_shots: int = 2
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    max_length: int | None = 776
    """Maximum length of the input sequence passed to the tokenize encoder function"""

    max_examples: tuple[int, int] = 3000
    """Maximum number of examples"""

    seed: int = 42
    """Random seed."""

    max_epochs: int = 1
    """Maximum number of epochs to train for."""
