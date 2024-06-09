# adapter_overseer

Inspired by the short circuit paper (https://arxiv.org/pdf/2406.04313), we see if this approach can be used for honesty rather than harmlessness. In particular, we:
- use an adapter
- to make sure the internal representation for truth is maintained
- and for lies the repr is orthogonalized
- we use a schedule to move toward retaining the truth new the end
- we will use [abliterated llama](https://huggingface.co/failspy/Llama-3-8B-Instruct-abliterated) to make sure we have enough examples of lies

## Project Organization

    ├── Justfile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── 30_processed      <- The final, canonical data sets for modeling.
    │   ├── 20_interim        <- Intermediate data that has been transformed.
    │   └── 10_raw            <- The original, immutable data dump.
    │
    ├── nbs                   <- Jupyter notebooks. Namiwith creator's initials, a number (for ordering), and short `-` delimited description, e.g.
    │                         `jqp-1.0-initial-data-exploration`.
    │
    ├── pyproject.toml    <- defines project dependencies and build configuration
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py


## Install requirements

This project uses poetry for requirement and is set up for torch using cuda.
```
poetry install
```

## How to get data

TODO document how to get the data

## How to run

This project uses [just](https://github.com/casey/just)

~~~
just --list
~~~

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
