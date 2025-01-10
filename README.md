# D3MES

To facilitate data uniformity, we uploaded our processed data in Kaggle, please download the corresponding three datasets from xx first.

## 1. Randomized generation based on qm9 dataset

To train the model, please run:

```bash
python train_random_qm9.py --global-batch-size=1024 --epochs=5000 --num-workers=0 --ckpt-every=20000
```
After training, the address of the generated model is filled in the corresponding location and the following code is run to generate it:
```bash
python sample_random_qm9.py
```
For evaluation related please run the following code:
```bash
python eval_qm9.py
```
## 2. Randomized generation based on Drugs dataset
To train the model, please run:
```bash
python train_random_drugs.py --global-batch-size=1024 --epochs=5000 --num-workers=0 --ckpt-every=20000
```
After training, the address of the generated model is filled in the corresponding location and the following code is run to generate it:
```bash
python sample_random_drugs.py
```
For evaluation related please run the following code:

```bash
python eval_drugs.py
```
## 3. Classification generates molecules
To train the model, please run:
```bash
python train_cla.py --global-batch-size=256 --epochs=1000 --num-workers=0 --ckpt-every=20000
```
After training, the address of the generated model is filled in the corresponding location. Generate and evaluate according to the categories that need to be generated and then the following code is run to generate it:
```bash
python sample_cla.py
```
For evaluation related please run the following code:
```bash
python eval_drugs.py
```
