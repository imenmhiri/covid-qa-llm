# Medical-Llama3-8B — Fine-tuning QLoRA pour le Q&A COVID-19 en français

Fine-tuning d'un LLM médical de 8 milliards de paramètres pour répondre à des questions
sur le COVID-19 **en français**, sur un seul GPU grand public, grâce à la quantification
4-bit et aux adaptateurs LoRA (QLoRA).

Le notebook [`covid-qa-llm.ipynb`](covid-qa-llm.ipynb) contient l'intégralité du pipeline :
chargement du modèle quantifié, préparation du jeu de données, entraînement et journalisation.

---

## Contexte et motivation

Les grands modèles médicaux disponibles publiquement sont majoritairement entraînés en
anglais. Ce projet adapte l'un d'eux au français sur un domaine précis — les questions
liées au COVID-19 — sans réentraîner les 8 milliards de poids d'origine.

Deux contraintes dictent les choix techniques :

1. **Mémoire GPU limitée.** Un modèle 8B en précision 16 bits demande ~16 Go rien que pour
   les poids. La quantification 4-bit NF4 ramène cette empreinte autour de 5 Go.
2. **Budget de calcul limité.** LoRA n'entraîne qu'un petit nombre de matrices de rang
   faible injectées dans les couches d'attention, au lieu de la totalité du modèle.

## Approche technique

| Composant | Choix | Pourquoi |
|---|---|---|
| Modèle de base | [`PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed`](https://huggingface.co/PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed) | Llama 3 8B déjà spécialisé sur le domaine médical, distribué pré-quantifié en 4-bit |
| Tokenizer | [`ruslanmv/Medical-Llama3-8B`](https://huggingface.co/ruslanmv/Medical-Llama3-8B) | Tokenizer du modèle non quantifié d'origine ; `pad_token = eos_token` |
| Quantification | NF4, compute dtype `bfloat16`, via `bitsandbytes` | Réduit l'empreinte mémoire des poids gelés |
| Adaptateurs | LoRA `r=16`, `alpha=32`, `dropout=0.05` | Compromis capacité / nombre de paramètres entraînables |
| Couches ciblées | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Projections d'attention : le meilleur rapport effet/coût pour une adaptation de style et de langue |
| Objectif | Modélisation du langage causale (`CAUSAL_LM`) | Le modèle apprend à générer la réponse à la suite de la question |

`prepare_model_for_kbit_training()` est appelé avant l'injection LoRA : cette fonction
remet les normalisations en précision 32 bits, active le gradient checkpointing et rend
les entrées différentiables — indispensable pour que les gradients traversent un modèle
quantifié.

## Jeu de données

`merged_covid_dataset_french.csv` — paires question/réponse en français sur le COVID-19.

| | |
|---|---|
| Paires totales | 11 181 (aucune valeur manquante) |
| Colonnes | `Question`, `Answer` |
| Entraînement / test | 8 944 / 2 237 (split 80-20, `random_state=42`) |

Chaque exemple est mis en forme selon un format de dialogue minimal, puis tokenisé :

```
<Human>: {Question}
<AI>: {Answer}
```

> **Note :** le jeu de données n'est pas versionné dans ce dépôt. Le notebook le lit depuis
> `/kaggle/input/frenchdataset/merged_covid_dataset_french.csv`. Adaptez ce chemin à votre
> environnement.

## Hyperparamètres d'entraînement

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 8    # taille de batch effective : 8
num_train_epochs             = 1
learning_rate                = 2e-4
warmup_ratio                 = 0.1
fp16                         = True
eval_strategy                = "steps"   # évaluation tous les 100 pas
save_strategy                = "epoch"
```

La taille de batch de 1 compensée par 8 pas d'accumulation permet de tenir dans la mémoire
du GPU tout en conservant un gradient stable. Les métriques sont exportées vers TensorBoard
dans `./logs`.

## Résultats

| Métrique | Valeur |
|---|---|
| Pas d'entraînement | 1 118 |
| Perte d'entraînement finale | **1,658** |
| Durée totale | ~10 h 55 min (39 323 s) |
| Débit | 0,23 exemple/s |
| FLOPs totaux | 7,47 × 10¹⁶ |

Les adaptateurs entraînés sont écrits dans `medical-llama-covid-french-finetuned/`.

## Reproduire l'entraînement

### Prérequis

- GPU NVIDIA avec ~16 Go de VRAM (le notebook a été exécuté sur Kaggle, Python 3.10)
- Un compte Hugging Face si vous accédez à des modèles sous licence restreinte

### Installation

```bash
pip install -qU bitsandbytes transformers datasets accelerate loralib einops xformers
pip install -q -U git+https://github.com/huggingface/peft.git
pip install rouge-score nltk evaluate
```

### Exécution

1. Ouvrez `covid-qa-llm.ipynb` (Kaggle, Colab ou Jupyter local).
2. Remplacez le chemin du CSV dans la cellule « Loading and Preparing the Dataset ».
3. Exécutez les cellules dans l'ordre.

Pour une exécution locale, retirez `fp16=True` si votre GPU ne gère pas la précision mixte,
et réduisez `gradient_accumulation_steps` si la mémoire le permet.

## Inférence

Les poids LoRA se rechargent par-dessus le modèle de base quantifié :

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed",
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "medical-llama-covid-french-finetuned")
tokenizer = AutoTokenizer.from_pretrained("ruslanmv/Medical-Llama3-8B")

prompt = "<Human>: Quels sont les symptômes du COVID-19 ?\n<AI>:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limites et pistes d'amélioration

- **Pas d'évaluation quantitative de la génération.** Les bibliothèques ROUGE et BLEU sont
  importées mais le calcul des scores n'est pas encore implémenté. C'est la prochaine étape
  logique pour mesurer l'apport du fine-tuning par rapport au modèle de base.
- **Une seule époque.** La perte descend encore en fin d'entraînement ; un budget plus large
  laisserait de la marge de progression.
- **Masquage des étiquettes.** La perte est calculée sur l'ensemble du prompt, question
  comprise. Ne la calculer que sur la réponse concentrerait l'apprentissage sur la
  génération plutôt que sur la reformulation des questions.
- **Longueur de séquence.** La tokenisation utilise `truncation=True` sans `max_length`
  explicite : les paires très longues sont coupées à la limite par défaut du tokenizer.

## Avertissement

Ce modèle est un travail de recherche et d'apprentissage. Ses réponses **ne constituent pas
un avis médical** et ne doivent pas servir à poser un diagnostic ou à guider un traitement.
Les informations sur le COVID-19 évoluent : les réponses du modèle reflètent son corpus
d'entraînement, pas l'état actuel des connaissances.

## Pile technique

`PyTorch` · `Transformers` · `PEFT` · `bitsandbytes` · `Datasets` · `Accelerate` ·
`scikit-learn` · `TensorBoard`
