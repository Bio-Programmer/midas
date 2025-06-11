import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_class_weights(y):
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return torch.tensor(weights, dtype=torch.float)

def train_one_epoch(model, train_loader, optimizer, criterion, device, forward_fn):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Train"):
        batch = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = forward_fn(model, batch)
        loss = criterion(outputs, batch[-1].long())
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        if not np.isfinite(batch_loss):
            raise ValueError(f"Non-finite loss: {batch_loss:.4f}")

        running_loss += batch_loss * batch[0].size(0)
    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device, forward_fn):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val"):
            batch = [x.to(device) for x in batch]
            outputs = forward_fn(model, batch)
            loss = criterion(outputs, batch[-1].long())
            running_loss += loss.item() * batch[0].size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch[-1].cpu().numpy())
    avg_loss = running_loss / len(val_loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    bma = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, acc, bma, all_preds, all_labels

def train_model(model, train_loader, val_loader, train_df, index_to_label, forward_fn, results_root,
                criterion=None, optimizer=None, scheduler=None,
                experiment_name="experiment", num_epochs=10, lr=1e-4, patience=3):
    device = get_device()
    model = model.to(device)
    if criterion is None:
        class_weights = prepare_class_weights(train_df["label"].values).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("MyDrive/midas/results", f"{experiment_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_log.txt")
    log_file = open(log_path, "w")

    train_losses, val_losses = [], []
    # best_val_loss = float('inf')
    best_val_bma = 0.0
    patience_counter = 0
    best_model_path = os.path.join(results_dir, "best_model.pt")

    for epoch in range(num_epochs):
        print("CUDA available:", torch.cuda.is_available())
        print("Current device:", torch.cuda.current_device())

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, forward_fn)
        val_loss, val_acc, val_bma, all_preds, all_labels = validate(model, val_loader, criterion, device, forward_fn)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log_str = (f"\nEpoch {epoch+1}: Train Loss = {train_loss:.4f}\n"
                   f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}\n"
                   f"Val Acc = {val_acc:.4f}, Val BMA = {val_bma:.4f}\n"
                   f"Classification Report:\n{classification_report(all_labels, all_preds, target_names=[index_to_label[i] for i in range(8)])}\n"
                   f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}\n")

        print(log_str)
        log_file.write(log_str + "\n")

        if val_bma > best_val_bma:
            best_val_bma = val_bma
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best validation BMA! We saved the model in experiment results directory.")
        else:
            patience_counter += 1
            print(f"EarlyStopping: No improvement in val BMA for {patience_counter} epoch(s).")
            log_file.write(f"EarlyStopping: No improvement in BMA for {patience_counter} epoch(s).\n")
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch+1}.")
                log_file.write(f"Stopping early at epoch {epoch+1}.\n")
                break

        # Plot loss curve after each epoch
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, "loss_curve.png"))
        plt.close()

    log_file.close()
    print(f"Training complete. Logs, model, and plots saved to: {results_dir}")
    return train_losses, val_losses, best_model_path
