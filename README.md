# Turkish Tokenization Study, Part 1

**Do BPE tokenizers cut Turkish words at morpheme boundaries?**

Turkish is agglutinative: words are built by adding suffixes to a stem, one morpheme at a time. *pencerelerden* ("from the windows") is *pencere-ler-den*. A tokenizer that cuts somewhere else leaves pieces that carry no meaning of their own. This project measures how often each tokenizer's cuts land on real morpheme boundaries, using the [Zemberek](https://github.com/ahmetaa/zemberek-nlp) morphological analyser as the reference.

## Tokenizers

Nine BPE tokenizers, chosen to cross two factors: how much Turkish the tokenizer saw in training, and its vocabulary size.

| Turkish exposure | Small (50–100k) | Medium (100–200k) | Large (200k+) |
|---|---|---|---|
| None | `gpt2` | `LumiOpen/Poro-34B` | `bigscience/bloom-560m` |
| Multilingual, incl. Turkish | `ai-forever/mGPT` | `meta-llama/Meta-Llama-3-8B-Instruct` | `google/gemma-2-2b` |
| Turkish only | `burakaytan/roberta-base-turkish-uncased` | `erenyanic/turkish_bpe_128k` | `tahaenesaslanturk/ts-corpus-bpe-256k-cased` |

## Method

1. **Reference boundaries.** Zemberek analyses each word into morphemes. Adding up the lengths of their surface forms gives the character positions where one morpheme ends and the next begins.
2. **Tokenizer boundaries.** Each tokenizer tokenizes the word on its own (no leading space, no special tokens). The end offset of each token gives the positions where the tokenizer cut.
3. **Scores**, counted across the whole word list:
   - **Precision**: of the tokenizer's cuts, the share that fall on a real morpheme boundary
   - **Recall**: of the real morpheme boundaries, the share the tokenizer cut at
   - **F0.5**: combines the two, weighting precision more than recall

Why F0.5 instead of F1: a cut inside a morpheme (lower precision) leaves fragments with no meaning of their own, while a missed boundary (lower recall) keeps two meaningful units intact, just merged. The first is the more damaging error, so it should cost more.

## Setup

You need:

- Python 3.11 (developed on 3.11.9)
- Java JDK 23 or newer (the script passes JVM flags that older versions don't recognise)
- `zemberek-full.jar`: download it from the [Zemberek-NLP](https://github.com/ahmetaa/zemberek-nlp) distributions and put it in the same folder as the script

Install the Python packages:

```bash
pip install -r requirements.txt
```

Llama 3's tokenizer is gated on Hugging Face. Request access on the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), then log in:

```bash
hf auth login
```

## Run

```bash
python boundary_scores.py
```

For each tokenizer, it prints: name, precision, recall, F0.5.

## Status

- The pipeline runs end to end.
- The word list in the script is a 7-word placeholder. Scores from it are not meaningful yet; a larger evaluation dataset is in progress.
- When a word has more than one possible analysis, the script currently uses Zemberek's first one. Once the dataset records each word's intended meaning, ambiguous words (e.g. *kalem*, "pen", vs. *kale-m*, "my castle") will be scored against the right analysis.
- Part 2 (planned) moves to interpretability: looking inside Gemma-2-2B to see whether its internal states recover the morphemes its tokenizer split.

## Built by

Ece Erkmen and Claude (Anthropic)
