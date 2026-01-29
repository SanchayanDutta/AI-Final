# 100_ANIMALS OQA Dataset

This package contains a 100 item ANIMALS attribute table, prompts for running the Optimal Question Asking game, and an entropy plot bundle.

## Repository layout

```
100-animals-oqa-dataset/
├─ data/
│  ├─ 100_animals.json
│  ├─ items.txt
│  ├─ attributes.txt
│  └─ equivalence_classes.json
├─ plots/
│  ├─ 100_animals_entropy_summary.csv
│  ├─ 100_animals_entropy_summary.json
│  ├─ 100_animals_entropy_seeds.csv
│  ├─ 100_animals_entropy_plot.png
│  ├─ make_plot.py
│  └─ README.txt
├─ prompts/
│  ├─ prompt_plain.txt
│  ├─ prompt_strict_json.txt
│  └─ prompt_system.txt
├─ metadata.json
```

## Data

The file `data/100_animals.json` stores a map from animal name to a vector of 14 boolean attributes:

- amphibian
- big
- bird
- can_fly
- can_swim
- carnivore
- domestic
- fish
- has_fur
- herbivore
- insect
- lays_eggs
- mammal
- reptile

`data/items.txt` lists the 100 animals in canonical order, one per line.

`data/attributes.txt` lists the 14 attributes in canonical order.

`data/equivalence_classes.json` groups animals that share the same attribute vector and records summary counts for the classes.

## Prompts

The three files in `prompts` are ready to drop into an API call. The plain prompt sets up the game in natural language. The strict JSON prompt asks the model to respond using a machine readable object and refers to the 100_ANIMALS dataset. The system prompt encourages informative questions and compact outputs.

## Plot bundle

The `plots` folder contains tidy entropy tables and a figure for five series: GPT 5, Gemini 2.5 Pro, Claude Sonnet 4.5, Grok 4, and Oracle. The seeds file stores per run posterior entropies. The summary file aggregates these runs into means, standard deviations, and extrema for each step.

## Technical note

For the API experiments, the APIs were accessed on 16 November 2025.
