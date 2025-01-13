# D3MES: Diffusion Transformer with Multihead Equivariant Self-Attention for 3D Molecule Generation

Understanding and predicting the diverse conformational states of molecules is crucial for advancing fields such as chemistry, material science, and drug development. Despite significant progress in generative models, accurately generating complex and biologically or material-relevant molecular structures remains a major challenge. 

In this project, we introduce **D3MES**, a diffusion model for three-dimensional (3D) molecule generation that combines a classifiable diffusion model, **Diffusion Transformer**, with **multihead equivariant self-attention**. Our method addresses two key challenges:  
1. Correctly attaching hydrogen atoms in generated molecules by learning representations of molecules after hydrogen atoms are removed.  
2. Generating molecules across multiple classes simultaneously, overcoming the limitations of existing models.
![D3MES Architecture](pictures/etc.png)

## Key Features
- State-of-the-art performance on molecular generation tasks.
- Robustness and versatility for large-scale generation processes.
- Highly suitable for early-stage molecule design, with further validation and screening capabilities.

## Table of Contents
1. [Randomized Generation Based on QM9 Dataset](#1-randomized-generation-based-on-qm9-dataset)
2. [Randomized Generation Based on Drugs Dataset](#2-randomized-generation-based-on-drugs-dataset)
3. [Classification-Based Molecule Generation](#3-classification-based-molecule-generation)

---

## 1. Randomized Generation Based on QM9 Dataset

### Training
To train the model on the QM9 dataset, run the following command:
```bash
python train_random_qm9.py --global-batch-size=1024 --epochs=5000 --num-workers=0 --ckpt-every=20000
```
### Generation
After training, the address of the generated model is filled in the corresponding location and the following code is run to generate it:
```bash
python sample_random_qm9.py
```
### Evaluation
For evaluation related please run the following code:
```bash
python eval_qm9.py
```

## 2. Randomized generation based on Drugs dataset
### Training
To train the model, please run:
```bash
python train_random_drugs.py --global-batch-size=1024 --epochs=5000 --num-workers=0 --ckpt-every=20000
```
### Generation
After training, the address of the generated model is filled in the corresponding location and the following code is run to generate it:
```bash
python sample_random_drugs.py
```
### Evaluation
For evaluation related please run the following code:

```bash
python eval_drugs.py
```
## 3. Classification generates molecules
### Training
To train the model, please run:
```bash
python train_cla.py --global-batch-size=256 --epochs=1000 --num-workers=0 --ckpt-every=20000
```
### Generation
After training, the address of the generated model is filled in the corresponding location. Generate and evaluate according to the categories that need to be generated and then the following code is run to generate it:
```bash
python sample_cla.py
```
### Evaluation
For evaluation related please run the following code:
```bash
python eval_drugs.py
```
