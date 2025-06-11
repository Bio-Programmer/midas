import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score
from tqdm.auto import tqdm  

def ensemble_predict(models, dataloaders, device, weights=None, num_classes=8):
    """
    Performs soft-voting ensemble using separate dataloaders (e.g., for different image sizes).
    Returns: predictions, labels, accuracy, balanced accuracy (BMA)
    """
    assert len(models) == len(dataloaders)
    if weights is None:
        weights = [1 / len(models)] * len(models)

    all_probs = []
    all_labels = None
    individual_preds = []

    for model, loader, w in zip(models, dataloaders, weights):
        model = model.to(device).eval()
        model_probs = []
        model_preds = [] 
        current_labels = []

        with torch.no_grad():
            for batch in tqdm(loader,
                              desc=f"Infer {type(model).__name__}",
                              unit="batch"):       # ← progress bar
                if len(batch) == 2:  # Swin-style: (images, labels)
                    images, labels = batch
                    images = images.to(device)
                    probs = F.softmax(model(images), dim=1)
                else:  # MLPFusion-style: (images, metadata, labels)
                    images, metadata, labels = batch
                    images = images.to(device)
                    metadata = metadata.to(device)
                    probs = F.softmax(model(images, metadata), dim=1)

                model_probs.append(probs.cpu())
                if all_labels is None:
                    current_labels.extend(labels.cpu().numpy())

        model_probs = torch.cat(model_probs, dim=0).numpy()
        all_probs.append(w * model_probs)

    final_probs = np.sum(all_probs, axis=0)
    final_preds = np.argmax(final_probs, axis=1)

    acc = accuracy_score(current_labels, final_preds)
    bma = recall_score(current_labels, final_preds, average='macro')

    return final_preds, current_labels, acc, bma

  
def get_model_predictions(model, dataloader, device, is_fusion=False):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            if is_fusion:
                x, meta, y = [b.to(device) for b in batch]
                out = model(x, meta)
            else:
                x, y = [b.to(device) for b in batch]
                out = model(x)

            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    return np.array(preds), np.array(labels)
