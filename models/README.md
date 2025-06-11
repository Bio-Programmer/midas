# CS231N Final Project 

Improving deep learning-based diagnosis of skin malignancies using the Melanoma Research Alliance Multimodal Image Dataset for AI-based Skin Cancer (MIDAS) diagnosis

Sophia Longo

Spring 2025

## Notebooks
1. `starter.ipynb`: initializing codebase and importing dataests (ISIC, MIDAS)
2. `dev.ipynb`: training the EfficientNet-B3 model as our "baseline" on the dev set (1,000 ISIC training images)
3. `model_experiments.ipynb`: testing out differnt models (EfficientNet-B3, Swin) and architecutres (including ensembles) on the dev set (table of results at end)
4. `MIDAS.ipynb`: after selecting highest performing model from dev experiments in (3), trained on full ISIC training set, fine-tuned on midas
5. `plots.ipynb`: code for creating plots, confusion matrices from results logs 

! [Model Training Scheme] (training_procedure.png)

## Directories
1. All models designed and tested can be found in `modeles` as separate scripts
2. Preprocessing, loading, benchmarking, and evaluation scripts found in `utils`
3. Training script in `src`
4. ISIC 2019 data found in `data` and MIDAS in `data_midas`; omitted dataset images for upload
5. For each experiment/training round, a training log with metrics, loss curve, and best model weights are saved within a directory in `results`


