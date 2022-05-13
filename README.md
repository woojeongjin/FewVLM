## A Good Prompt Is Worth Millions of Parameters: Low-resource Prompt-based Learning for Vision-Language Models

[Paper](https://arxiv.org/abs/2110.08484) (ACL 2022)

This repository contains the implementation of FewVLM described in the paper.
Codes are based on [VL-T5](https://github.com/j-min/VL-T5)

## Installation

```bash
pip install -r requirements.txt
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Datasets

- Datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1T8x5O3sZp83_x9XX2Yg0HgCP8P_Bh8KD?usp=sharing)
- For other datasets, please refer to [VL-T5 repository](https://github.com/j-min/VL-T5/tree/main/feature_extraction).

## Pre-trained checkpoints

- [Base](https://drive.google.com/file/d/17B-3TcXJ1tumNPYAQpg2MisSi9-7Dz-Y/view?usp=sharing), [Large](https://drive.google.com/file/d/1Y80nD_FYAs5Gs2mbR_xcshEBTowh6t8I/view?usp=sharing)

## Pre-training

```bash
# Pre-train with 8 GPUs
bash scripts/pretrain.sh 8 
```

## Zero/few-shot Learning

All commands are runnable on a single GPU.

### VQA

```bash
# for few-shot
bash scripts/VQA.sh 0 VQA --subsample --dataseed 42 --num_data 16 --test_only --prompt 3

# for zero-shot 
bash scripts/VQA.sh 0 VQA --test_only --prompt 3
```

### OKVQA

```bash
# for few-shot
bash scripts/OKVQA.sh 0 OKVQA --subsample --dataseed 42 --num_data 16 --test_only --prompt 3

# for zero-shot 
bash scripts/OKVQA.sh 0 OKVQA --test_only --prompt 3
```

### GQA

```bash
# for few-shot
bash scripts/GQA.sh 0 GQA --subsample --dataseed 42 --num_data 16 --test_only --prompt 3

# for zero-shot 
bash scripts/GQA.sh 0 GQA --test_only --prompt 3
```

### Flickr30k

```bash
# for few-shot
bash scripts/flickr.sh 0 flickr --subsample --dataseed 42 --num_data 16 --prefix image 

# for zero-shot 
bash scripts/flickr.sh 0 flickr --prefix image --test_only 
```

### Nocaps

```bash
# for few-shot
bash scripts/nocaps.sh 0 nocaps --subsample --dataseed 42 --num_data 16 --prefix image 

# for zero-shot 
bash scripts/nocaps.sh 0 nocaps --prefix image --test_only 
```

Some important command line arguments are listed as follows:

| Arg                             | Values                                                     | Description                      | Notes                                                        |
| ------------------------------- | ---------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `--load`                        | path for trained checkpoints                               | load a checkpoint                |                                                              |
| `--dataseed`                    | {0, 42, 9595,...}                                          | Random seed for data shuffling   | default=42                                                   |
| `--seed`                        | {0, 42, 9595,...}                                          | Random seed for parameter shuffling | default=9595                                              |
| `--subsample`                   | store_true                                                 | Subsample train and val sets for few-shot learning  |                                           |
| `--num_data`                    | {16, 40, ...}                                              | Number of subsamples for train and val sets | default=16                                        |
| `--test_only`                   | store_true                                                 | Run test without training        |                                                              |
| `--prompt`                      | {0, 1, 2, 3}                                               | Prompts for VQA                  | default=0, 0: no prompt, 1: '[Q] <text_1>', 2: 'question: [Q] answer:', 3: 'question: [Q] answer: <text_1>'         |
| `--prefix`                      | {None, 'image', 'picture', 'photo'}                        | Prompts for captioning           | Default=None, 'image': 'an image of', 'picture': 'a picture of', 'photo': 'a photo of' |
| `--backbone`                    | {'t5-base', 't5-large'}                                    | Backbone architecture            | default='t5-base'                                            |

