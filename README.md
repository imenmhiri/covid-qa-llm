# Assistant COVID-19 en français — traduction, fine-tuning QLoRA et déploiement RAG

Chaîne complète pour obtenir un assistant de questions-réponses sur le COVID-19 **en
français**, à partir d'un LLM médical anglophone de 8 milliards de paramètres, sur un seul
GPU grand public.

Le projet se décompose en quatre étapes, chacune portée par un notebook :

```
  covid_dataset.csv            ①  Traduction          covid_dataset_french.csv
  (Q/R, anglais)          ──────────────────────►     (11 181 paires, français)
                             MarianMT opus-mt-en-fr
                                                                │
                                                                ▼
                                                     ②  Fine-tuning QLoRA
                                              Medical-Llama3-8B 4-bit + adaptateurs LoRA
                                                 ├─ Part 1 : sous-ensemble + évaluation
                                                 └─ Jeu complet : checkpoint-1118
                                                                │
                                                                ▼
  rag_data/*.pdf              ③  Indexation            ④  Application Streamlit
  (rapports officiels FR) ──────────────────────►    RAG (FAISS) + modèle finetuné
                             FAISS + embeddings              exposée via localtunnel
```

---

## Table des matières

- [Structure du dépôt](#structure-du-dépôt)
- [Étape 1 — Traduction du jeu de données (EN → FR)](#étape-1--traduction-du-jeu-de-données-en--fr)
- [Étape 2 — Fine-tuning QLoRA](#étape-2--fine-tuning-qlora)
- [Étape 3 — Évaluation](#étape-3--évaluation)
- [Étape 4 — Déploiement : RAG + Streamlit](#étape-4--déploiement--rag--streamlit)
- [Reproduire le pipeline](#reproduire-le-pipeline)
- [Inférence sans l'interface](#inférence-sans-linterface)
- [Limites connues et pistes d'amélioration](#limites-connues-et-pistes-damélioration)
- [Avertissement](#avertissement)
- [Pile technique](#pile-technique)

---

## Structure du dépôt

| Chemin | Rôle |
|---|---|
| `notebooks/Traduction_ENFR.ipynb` | ① Traduction du jeu de données anglais vers le français (MarianMT, Colab) |
| `notebooks/FineTuning__Part1.ipynb` | ② Fine-tuning QLoRA sur un sous-ensemble **+ ③ évaluation** (BLEU, ROUGE, perplexité) |
| `notebooks/covid-qa-llm.ipynb` | ② Fine-tuning QLoRA sur le **jeu complet** — produit le checkpoint servi en production |
| `notebooks/Deploiement.ipynb` | ④ Récupération du checkpoint et lancement de l'application sur Colab |
| `app/app.py` | Application Streamlit : RAG sur PDF + génération avec le modèle finetuné |
| `app/launch.sh` | Démarre Streamlit et expose le port 8501 via localtunnel |
| `app/rag_data/` | Deux rapports officiels français utilisés comme base de connaissances RAG |
| `data/covid_dataset.csv` | Jeu de données source, questions/réponses COVID-19 en anglais |
| `data/covid_dataset_french.csv` | Résultat de l'étape ① : 11 181 paires en français |
| `requirements.txt` | Dépendances des quatre étapes, regroupées par phase |

Les checkpoints d'entraînement (`model_output/`, `medical-llama-covid-french-finetuned/`)
ne sont pas versionnés : ils pèsent plusieurs centaines de Mo et sont régénérables.

---

## Étape 1 — Traduction du jeu de données (EN → FR)

> `notebooks/Traduction_ENFR.ipynb` — exécuté sur Google Colab (GPU Tesla T4)

Les corpus de questions-réponses médicales disponibles sont presque tous en anglais. Plutôt
que de constituer un jeu de données français à la main, le projet traduit automatiquement le
corpus anglais avec un modèle de traduction neuronale dédié.

| Paramètre | Valeur |
|---|---|
| Modèle | [`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) (MarianMT) |
| Sens de traduction | **anglais → français** |
| Colonnes traduites | `Question` **et** `Answer` — les deux faces de la paire doivent être dans la langue cible |
| Longueur maximale | 512 tokens par segment (`truncation=True`) |
| Entrée / sortie | `covid_dataset.csv` → `covid_dataset_french.csv` |
| Robustesse | Lecture du CSV en essayant successivement `utf-8`, `latin1`, `iso-8859-1`, `cp1252` ; écriture en `utf-8` |

La classe `CovidDataProcessor` encapsule le chargement du modèle, la traduction ligne par
ligne (avec repli sur le texte d'origine en cas d'erreur) puis une passe d'analyse
exploratoire : comptages, valeurs uniques et distributions des longueurs de questions et de
réponses (histogrammes sauvegardés dans Drive).

**Jeu de données obtenu** — `data/covid_dataset_french.csv`

| | |
|---|---|
| Paires totales | 11 181 (aucune valeur manquante) |
| Colonnes | `Question`, `Answer` |

> La traduction normalise au passage les réponses multi-lignes du corpus anglais en un seul
> paragraphe, ce qui explique l'écart de nombre de lignes entre les deux fichiers CSV.

---

## Étape 2 — Fine-tuning QLoRA

> `notebooks/FineTuning__Part1.ipynb` (sous-ensemble) et `notebooks/covid-qa-llm.ipynb` (jeu complet)

Deux contraintes dictent les choix techniques :

1. **Mémoire GPU limitée.** Un modèle 8B en précision 16 bits demande ~16 Go rien que pour
   les poids. La quantification 4-bit NF4 ramène cette empreinte autour de 5 Go.
2. **Budget de calcul limité.** LoRA n'entraîne qu'un petit nombre de matrices de rang faible
   injectées dans les couches d'attention, au lieu de la totalité du modèle.

### Configuration

| Composant | Choix | Pourquoi |
|---|---|---|
| Modèle de base | [`PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed`](https://huggingface.co/PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed) | Llama 3 8B déjà spécialisé sur le domaine médical, distribué pré-quantifié en 4-bit |
| Tokenizer | [`ruslanmv/Medical-Llama3-8B`](https://huggingface.co/ruslanmv/Medical-Llama3-8B) | Tokenizer du modèle non quantifié d'origine ; `pad_token = eos_token` |
| Quantification | NF4, compute dtype `bfloat16`, via `bitsandbytes` | Réduit l'empreinte mémoire des poids gelés |
| Adaptateurs | LoRA `r=16`, `alpha=32`, `dropout=0.05`, `bias="none"` | Compromis capacité / nombre de paramètres entraînables |
| Couches ciblées | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Projections d'attention : meilleur rapport effet/coût pour une adaptation de style et de langue |
| Objectif | Modélisation du langage causale (`CAUSAL_LM`) | Le modèle apprend à générer la réponse à la suite de la question |

`prepare_model_for_kbit_training()` est appelé avant l'injection LoRA : cette fonction remet
les normalisations en précision 32 bits, active le gradient checkpointing et rend les entrées
différentiables — indispensable pour que les gradients traversent un modèle quantifié.

Chaque exemple est mis en forme selon un format de dialogue minimal, puis tokenisé :

```
<Human>: {Question}
<AI>: {Answer}
```

### Hyperparamètres (identiques dans les deux notebooks)

```python
per_device_train_batch_size  = 1
per_device_eval_batch_size   = 1
gradient_accumulation_steps  = 8      # taille de batch effective : 8
num_train_epochs             = 1
learning_rate                = 2e-4
warmup_ratio                 = 0.1
fp16                         = True
eval_strategy                = "steps"   # évaluation tous les 100 pas
save_strategy                = "epoch"
save_total_limit             = 4
report_to                    = "tensorboard"   # journaux dans ./logs
```

Une taille de batch de 1 compensée par 8 pas d'accumulation permet de tenir dans la mémoire
du GPU tout en conservant un gradient stable.

### Deux entraînements, deux usages

| | **Part 1** (`FineTuning__Part1.ipynb`) | **Jeu complet** (`covid-qa-llm.ipynb`) |
|---|---|---|
| Rôle | Itération rapide, mise au point et **évaluation** | Entraînement final servi en production |
| Paires | 2 236 | 11 181 |
| Entraînement / test | 1 788 / 448 | 8 944 / 2 237 |
| Split | 80-20, `random_state=42` | 80-20, `random_state=42` |
| Pas d'entraînement | 223 | 1 118 |
| Perte d'entraînement finale | **1,611** | **1,658** |
| Durée | ~1 h 14 (4 466 s) | ~10 h 55 (39 323 s) |
| Débit | 0,40 exemple/s | 0,23 exemple/s |
| FLOPs totaux | 1,41 × 10¹⁶ | 7,47 × 10¹⁶ |
| Sortie | `medical-llama-covid-french-finetuned/` | `medical-llama-covid-french-finetuned/checkpoint-1118` ← **servi par `app.py`** |

Entraîner d'abord sur un cinquième du corpus permet de valider la boucle complète
(chargement 4-bit, injection LoRA, tokenisation, évaluation) en une heure au lieu de onze,
avant d'engager l'entraînement long.

> Les notebooks lisent leurs CSV depuis des chemins Kaggle/Colab
> (`/kaggle/input/...`, `/content/drive/My Drive/...`). Adaptez ces chemins, ou pointez-les
> vers `data/covid_dataset_french.csv` de ce dépôt. Le sous-ensemble utilisé par Part 1
> (`covid_dataset_part_1_frensh.csv`, 2 236 paires) n'est pas versionné : c'est une tranche
> du jeu français complet.

---

## Étape 3 — Évaluation

> `notebooks/FineTuning__Part1.ipynb`, fonction `compute_metrics()`

L'évaluation génère une réponse pour chacune des 448 questions du jeu de test (décodage
glouton, `do_sample=False`, `no_repeat_ngram_size=2`, 512 tokens), puis la compare à la
réponse de référence.

| Métrique | Valeur | Lecture |
|---|---|---|
| BLEU (lissage method1) | 0,0168 | Recouvrement n-gramme exact, très sévère sur des réponses longues reformulées |
| ROUGE-1 F1 | 0,2332 | ~23 % des unigrammes de la référence retrouvés |
| ROUGE-2 F1 | 0,0530 | Enchaînements de deux mots |
| ROUGE-L F1 | 0,1152 | Plus longue sous-séquence commune |
| Perplexité | 4,5244 | Calculée sur le prompt complet du jeu de test |
| Échantillons traités / ignorés | 448 / 0 | Aucune erreur de génération |

**Comment lire ces chiffres.** BLEU et ROUGE mesurent un recouvrement lexical : sur des
réponses libres de plusieurs phrases, un modèle qui répond correctement mais avec d'autres
mots est fortement pénalisé. Un BLEU de 0,017 ne signifie donc pas que le modèle répond
faux. La perplexité de 4,52 est ici le signal le plus fiable : le modèle attribue une
probabilité élevée aux réponses de référence, donc il a bien assimilé le registre et la
langue du corpus.

**Ce qui manque pour conclure.** Aucune mesure n'a été faite sur le modèle de base *avant*
fine-tuning, et le jeu complet n'a pas été évalué. Sans ces deux points de comparaison, les
scores décrivent l'état du modèle mais ne quantifient pas l'apport du fine-tuning.

---

## Étape 4 — Déploiement : RAG + Streamlit

> `app/app.py`, `app/launch.sh`, `notebooks/Deploiement.ipynb`

Le fine-tuning fixe le **style** et la **langue** des réponses, mais les connaissances
restent figées à la date du corpus d'entraînement. Le RAG
(*Retrieval-Augmented Generation*) complète le dispositif : à chaque question, les passages
pertinents de documents officiels français sont retrouvés et injectés dans le prompt.

### Chaîne de récupération

| Étape | Implémentation |
|---|---|
| Chargement | `DirectoryLoader` + `PyPDFLoader` sur `app/rag_data/**/*.pdf` |
| Découpage | `RecursiveCharacterTextSplitter`, `chunk_size=500`, `chunk_overlap=50` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (GPU si disponible) |
| Index vectoriel | FAISS, construit en mémoire au premier lancement |
| Récupération | `similarity_search`, `k=2` passages par question |

Le modèle et l'index sont tous deux mis en cache via `@st.cache_resource` : ils ne sont
construits qu'une fois par session Streamlit.

### Base de connaissances (`app/rag_data/`)

- `avis_du_covars_du_20.10.2022_sur_le_covid-19.pdf` — avis du COVARS du 20 octobre 2022
- `final-report-rc-srs-covid-19-fr.pdf` — rapport final sur le SRAS-CoV-2

Un découpage à 500 caractères avec seulement 2 passages récupérés est un choix volontaire :
le prompt total est plafonné à 512 tokens, il faut donc laisser de la place à la question et
à la génération.

### Génération

```python
max_new_tokens = 256
do_sample      = True
temperature    = 0.7
top_p          = 0.9
no_repeat_ngram_size = 2
```

Le contexte est tronqué par `truncate_text()` avant l'appel au modèle, avec une marge de
100 tokens réservée à la réponse.

### Interface

Interface Streamlit en français : titre, barre latérale (mission, sources, avertissement
médical, contact), historique de conversation conservé dans `st.session_state`, et bulles de
discussion stylées en CSS.

### Mise en ligne depuis Colab

`notebooks/Deploiement.ipynb` enchaîne :

1. Installation des dépendances (`streamlit`, `langchain-community`, `faiss-cpu`, `pypdf`,
   `sentence-transformers`, `bitsandbytes`, `peft`, `localtunnel`).
2. Récupération du checkpoint — soit depuis la sortie du kernel Kaggle
   (`kaggle kernels output ... -p ./model_output`), soit par copie depuis Google Drive vers
   `/content/model_output/`.
3. `./launch.sh` : démarre Streamlit sur le port 8501 et l'expose par localtunnel, qui
   renvoie une URL publique temporaire (`https://*.loca.lt`).

---

## Reproduire le pipeline

### Prérequis

- GPU NVIDIA avec ~16 Go de VRAM — les notebooks ont été exécutés sur Kaggle et Colab
  (Tesla T4), Python 3.10
- Un compte Hugging Face si vous accédez à des modèles sous licence restreinte
- Un compte Kaggle (`kaggle.json`) uniquement pour récupérer le checkpoint depuis un kernel

### Installation

```bash
pip install -r requirements.txt
# PEFT depuis les sources, comme dans les notebooks :
pip install -q -U git+https://github.com/huggingface/peft.git
```

### Déroulé

```bash
# 1. Traduction — produit data/covid_dataset_french.csv (déjà fourni dans le dépôt)
jupyter notebook notebooks/Traduction_ENFR.ipynb

# 2. Fine-tuning — commencez par Part 1 pour valider la boucle en ~1 h
jupyter notebook notebooks/FineTuning__Part1.ipynb
jupyter notebook notebooks/covid-qa-llm.ipynb

# 3. Application — le checkpoint doit être présent sous app/model_output/
cd app && chmod +x launch.sh && ./launch.sh
```

Pensez à adapter, dans chaque notebook, le chemin du CSV et celui du checkpoint à votre
environnement. En local, retirez `fp16=True` si votre GPU ne gère pas la précision mixte.

---

## Inférence sans l'interface

Les poids LoRA se rechargent par-dessus le modèle de base quantifié. Respectez le format de
prompt utilisé à l'entraînement :

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "PrunaAI/ruslanmv-Medical-Llama3-8B-bnb-4bit-smashed",
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "medical-llama-covid-french-finetuned/checkpoint-1118")
tokenizer = AutoTokenizer.from_pretrained("ruslanmv/Medical-Llama3-8B")
tokenizer.pad_token = tokenizer.eos_token

prompt = "<Human>: Quels sont les symptômes du COVID-19 ?\n<AI>:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Limites connues et pistes d'amélioration

Les points ci-dessous sont des constats de lecture du code, classés par impact.

1. **Format de prompt incohérent entre entraînement et application.** Le modèle a été
   entraîné sur `<Human>: ... \n<AI>: ...`, mais `app/app.py` l'interroge avec
   `Question: ... Réponse basée sur le contexte suivant: ... Réponse:`. Le modèle est donc
   servi hors de la forme qu'il a apprise, ce qui dégrade la qualité des réponses. Aligner
   le prompt de `generate_response()` sur le format d'entraînement est le correctif au
   meilleur rapport effort/gain du projet.
2. **Embeddings anglophones sur des documents français.** `all-MiniLM-L6-v2` est entraîné
   sur de l'anglais ; l'index FAISS porte sur des PDF français interrogés par des questions
   françaises. Un modèle multilingue
   (`paraphrase-multilingual-MiniLM-L12-v2`, `intfloat/multilingual-e5-base`) améliorerait
   nettement la pertinence des passages récupérés, à coût comparable.
3. **Pas de baseline d'évaluation.** Les métriques de l'étape 3 ne sont pas comparées au
   modèle de base, ni calculées sur l'entraînement complet. Évaluer les deux dans les mêmes
   conditions est nécessaire pour attribuer un gain au fine-tuning.
4. **Masquage des étiquettes absent.** La perte est calculée sur l'ensemble du prompt,
   question comprise. Ne la calculer que sur la réponse concentrerait l'apprentissage sur la
   génération plutôt que sur la reformulation des questions.
5. **Une seule époque.** La perte descend encore en fin d'entraînement ; un budget plus large
   laisserait de la marge de progression.
6. **Longueur de séquence implicite.** La tokenisation d'entraînement utilise
   `truncation=True` sans `max_length` explicite : les paires très longues sont coupées à la
   limite par défaut du tokenizer, sans trace dans les journaux.
7. **Configuration de quantification partiellement ignorée.** Le modèle de base étant déjà
   pré-quantifié, `transformers` ignore le `BitsAndBytesConfig` passé à `from_pretrained`
   (avertissement visible dans les sorties). De plus, `load_4bit_use_double_quant` est une
   clé invalide — le nom attendu est `bnb_4bit_use_double_quant`. Sans effet ici, mais à
   corriger pour éviter de croire la double quantification active.
8. **Chemin de checkpoint codé en dur.** `app/app.py` pointe vers
   `./model_output/medical-llama-covid-french-finetuned/checkpoint-1118`. À externaliser en
   variable d'environnement pour pouvoir servir un autre checkpoint sans modifier le code.
9. **Qualité de la traduction non contrôlée.** Le corpus français est entièrement produit par
   machine, sans relecture ni mesure. Les erreurs de traduction sur le vocabulaire médical se
   propagent directement dans le modèle finetuné.
10. **Exposition par localtunnel.** Pratique pour une démonstration, mais l'URL est publique,
    temporaire et sans authentification : à remplacer par un hébergement applicatif pour tout
    usage durable.
11. **Index FAISS reconstruit à chaque démarrage.** Le persister sur disque
    (`FAISS.save_local` / `load_local`) supprimerait plusieurs dizaines de secondes au
    lancement.

---

## Avertissement

Ce projet est un travail de recherche et d'apprentissage. Les réponses du modèle **ne
constituent pas un avis médical** et ne doivent pas servir à poser un diagnostic ou à guider
un traitement. Les informations sur le COVID-19 évoluent : les réponses reflètent le corpus
d'entraînement et les documents indexés, pas l'état actuel des connaissances.

---

## Pile technique

**Traduction** — `MarianMT` · `Transformers`

**Entraînement** — `PyTorch` · `Transformers` · `PEFT` · `bitsandbytes` · `Datasets` ·
`Accelerate` · `scikit-learn` · `TensorBoard`

**Évaluation** — `NLTK` (BLEU) · `rouge-score` · `evaluate`

**Déploiement** — `Streamlit` · `LangChain` · `FAISS` · `sentence-transformers` · `pypdf` ·
`localtunnel`
