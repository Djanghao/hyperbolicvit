import os
import random
import numpy as np
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Net
from torch.cuda.amp import GradScaler, autocast
from geoopt.optim import RiemannianAdam
import torch.nn.utils
import gc

gc.collect()
# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to save the model
def save_model(output_dir, model, epoch):
    model_checkpoint = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_checkpoint)
    logger.info(f"Model checkpoint saved at {model_checkpoint}")

# Function to calculate accuracy
def simple_accuracy(preds, labels):
    return (preds == labels).mean() * 100

def top_5_accuracy(output, target):
    """
    Compute the top-5 accuracy for classification tasks using model outputs.
    """
    # Convert output and target to tensors if they are not already
    if isinstance(output, torch.Tensor):
        logits = output.detach()
    else:
        logits = torch.tensor(output)

    if isinstance(target, torch.Tensor):
        targets = target.detach()
    else:
        targets = torch.tensor(target)

    # Ensure logits are floating-point tensors
    if not logits.is_floating_point():
        logits = logits.float()

    # Ensure targets are of type Long for comparison
    if targets.dtype != torch.long:
        targets = targets.long()

    with torch.no_grad():
        # Handle 1D output (e.g., batch size of 1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # Shape: [1, n_classes]
            targets = targets.unsqueeze(0)  # Shape: [1]

        # Ensure logits have two dimensions: [batch_size, n_classes]
        if logits.dim() != 2:
            raise ValueError(f"Expected logits to be 2D, but got {logits.dim()}D")

        # Get the indices of the top 5 predictions for each sample
        top5_preds = torch.topk(logits, k=5, dim=1).indices  # Shape: [batch_size, 5]

        # Expand targets to compare with top5_preds
        targets_expanded = targets.view(-1, 1)  # Shape: [batch_size, 1]

        # Check if the true label is among the top 5 predictions
        correct = top5_preds.eq(targets_expanded)  # Shape: [batch_size, 5]

        # For each sample, check if any of the top 5 predictions is correct
        correct_any = correct.any(dim=1).float()  # Shape: [batch_size]

        # Compute the top-5 accuracy as the mean of correct predictions
        top5_accuracy = correct_any.mean().item() * 100.0

        return top5_accuracy

    
class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Geodesic regularization function
def geodesic_regularization(outputs, labels, manifold, lambda_reg=0.01):
    """
    Compute the geodesic regularization loss based on the output embeddings.
    """
    dist_matrix = manifold.dist(outputs.unsqueeze(1), outputs.unsqueeze(0))  # Pairwise geodesic distances
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # 1 if labels are the same, else 0
    reg_loss = ((1 - label_matrix) * dist_matrix).mean()  # Penalize distances between different-class points
    return lambda_reg * reg_loss


# Validation function
def validate(model, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_losses.update(loss.item(), inputs.size(0))

            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = simple_accuracy(all_preds, all_labels)
    top5 = top_5_accuracy(all_preds, all_labels)
    logger.info(f"Validation Accuracy: {accuracy:.4f}%, Validation Loss: {eval_losses.avg:.4f}, Top-5 Accuracy: {top5:.4f}%")
    return accuracy, top5

# Training function
def train_single_gpu(dataset_name='cifar10', subset_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    set_seed(42)  # Set a fixed seed for reproducibility

    # Set hyperparameters
    batch_size = 30
    num_epochs = 350       
    learning_rate = 5e-3  
    output_dir = './output'
    
    # Set dataset-specific parameters
    if dataset_name.lower() == 'cifar10':
        num_classes = 10
        img_size = 32
        patch_size = 4
    else:  # 'imagenet'
        num_classes = 1000
        img_size = 224
        patch_size = 16

    # Model setup
    model = Net(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes
    ).to(device)
    
    # Data augmentation and normalization for ImageNet
    transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_imagenet_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR-10 transforms
    transform_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Choose dataset based on parameter
    if dataset_name.lower() == 'cifar10':
        full_train_dataset = datasets.CIFAR10(root='/home/tmp/workspace/hyperbolicvit/data/cifar10', 
                                      train=True, download=True, transform=transform_cifar)
        full_val_dataset = datasets.CIFAR10(root='/home/tmp/workspace/hyperbolicvit/data/cifar10', 
                                     train=False, download=True, transform=transform_cifar_test)
        logger.info(f"Using CIFAR-10 dataset with image size {img_size} and patch size {patch_size}")
    else:  # 'imagenet'
        full_train_dataset = datasets.ImageNet(root='/home/tmp/workspace/hyperbolicvit/data/imagenet10', 
                                        split='train', transform=transform_imagenet)
        full_val_dataset = datasets.ImageNet(root='/home/tmp/workspace/hyperbolicvit/data/imagenet10', 
                                      split='val', transform=transform_imagenet_test)
        logger.info(f"Using ImageNet dataset with image size {img_size} and patch size {patch_size}")
    
    # Create subset of datasets if subset_ratio < 1.0
    if subset_ratio < 1.0:
        # Calculate subset sizes
        train_size = int(len(full_train_dataset) * subset_ratio)
        val_size = int(len(full_val_dataset) * subset_ratio)
        
        # Create random subsets
        indices = torch.randperm(len(full_train_dataset))
        train_indices = indices[:train_size]
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        
        indices = torch.randperm(len(full_val_dataset))
        val_indices = indices[:val_size]
        val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
        
        logger.info(f"Using {subset_ratio:.1%} of the dataset: {train_size} training samples, {val_size} validation samples")
    else:
        train_dataset = full_train_dataset
        val_dataset = full_val_dataset
        logger.info(f"Using full dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=f"./runs/{dataset_name}_tensorboard")

    # Set up optimizer and scheduler
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Optional: add geodesic regularization
                geo_reg_loss = model.geodesic_regularization(outputs, labels, margin=1.0)
                total_loss = loss + geo_reg_loss
            
            # Backward and optimize with gradient scaling
            scaler.scale(total_loss).backward()
            
            # Clip gradients (often needed for stability in hyperbolic space)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step the optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Moved after optimizer step

            running_loss += total_loss.item()

            # Clear cache to reduce memory fragmentation
            torch.cuda.empty_cache()

        # Log training loss
        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss}")

        # Validation
        accuracy, top5 = validate(model, val_loader, device)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Top5Accuracy/val', top5, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {accuracy}, Top5: {top5}")
        
        # Save model checkpoint
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(output_dir, model, epoch + 1)
            logger.info(f"New best accuracy: {best_acc}")
        else:
            save_model(output_dir, model, epoch + 1)
        
        # Early stopping (optional)
        if epoch > 10 and epoch_loss < 0.01:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    logger.info(f"Best validation accuracy: {best_acc}")
    return model

def main():
    # Get dataset from environment variable, default to CIFAR-10
    dataset_name = os.environ.get('DATASET', 'cifar10')
    
    # Get subset ratio from environment variable
    subset_ratio = float(os.environ.get('SUBSET_RATIO', '0.1'))
    
    train_single_gpu(dataset_name, subset_ratio)

if __name__ == "__main__":
    main()
