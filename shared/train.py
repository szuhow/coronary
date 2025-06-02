import copy
import time
from pathlib import Path
from typing import Dict, Tuple, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv
import argparse
import itertools
from torchvision import transforms
import json
from unet import UNet
from datasets.coronary_dataset import CoronaryDataset
from utils.config import Config
from utils.loss import calc_loss, Dice, IoU, DebugDice

# class TrainingManager:
#     def __init__(self, config: Config, transform=None):
#         self.config = config
#         self.transform = transform
#         self.device = self._setup_device()
#         self.model = self._setup_model()
#         self.dataloaders = self._setup_dataloaders()
#         self.optimizer = self._setup_optimizer()
#         self.scheduler = self._setup_scheduler()
#         self.writer = self._setup_tensorboard()
#         self.loss_function = self._setup_loss_function()
        
#     def _setup_device(self) -> torch.device:
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             print(f'Using GPU: {torch.cuda.get_device_name(0)}')
#         elif torch.backends.mps.is_available():
#             device = torch.device("mps")
#             print("Using MPS")
#         else:
#             device = torch.device("cpu")
#             print("Using CPU")
#         return device
    
#     def _setup_model(self) -> nn.Module:
#         model = UNet(1).to(self.device)
#         return model
    
#     def _print_epoch_summary(self, epoch_start: float, epoch_loss: float, best_loss: float):
#         epoch_time = time.time() - epoch_start
#         print(f'Epoch completed in {epoch_time:.0f}s')
#         print(f'Current loss: {epoch_loss:.4f}, Best loss so far: {best_loss:.4f}')


#     def _setup_dataloaders(self) -> Dict[str, DataLoader]:
#         train_dataset, val_dataset = self._split_dataset()
#         return {
#             'train': DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
#             'val': DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
#         }
    
#     def _split_dataset(self) -> Tuple[CoronaryDataset, CoronaryDataset]:
#         path = Path(self.config.path)
#         images = sorted(list(path.glob('JPEGImages/*')))
#         masks = sorted(list(path.glob('SegmentationClassPNG/*')))
        
#         split_idx = int(len(images) * 0.8)
#         train_images, val_images = images[:split_idx], images[split_idx:]
#         train_masks, val_masks = masks[:split_idx], masks[split_idx:]
        
#         train_dataset = CoronaryDataset(train_images, train_masks, train=True, transform=self.transform)
#         val_dataset = CoronaryDataset(val_images, val_masks, train=False, transform=self.transform)
        
#         return train_dataset, val_dataset
    
#     def _setup_optimizer(self) -> optim.Optimizer:
#         optimizers = {
#             'adam': optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5),
#             'sgd': optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=1e-5)
#         }
#         if self.config.optimizer_name not in optimizers:
#             raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")
#         return optimizers[self.config.optimizer_name]
    
#     def _setup_scheduler(self) -> optim.lr_scheduler.ReduceLROnPlateau:
#         return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
    
#     def _setup_tensorboard(self) -> SummaryWriter:
#         return SummaryWriter(comment=self._get_tensorboard_comment())
    
#     def _get_tensorboard_comment(self) -> str:
#         return (f" LR_{self.config.lr}_optimizer_{self.config.optimizer_name}"
#                 f"_batch_size_{self.config.batch_size}_epochs_{self.config.epochs}"
#                 f"_bce_weight_{self.config.bce_weight}")
    
#     def _setup_loss_function(self):
#         loss_functions = {
#             'dice': Dice(),
#             'iou': IoU(),
#             'debugdice': DebugDice()
#         }
#         if self.config.loss_type not in loss_functions:
#             raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
#         return loss_functions[self.config.loss_type]

#     def train(self) -> nn.Module:
#         """
#         Train the model and return the best performing version.
#         """
#         best_model_wts = copy.deepcopy(self.model.state_dict())
#         best_loss = float('inf')
#         patience_counter = 0
#         max_patience = 10  # Early stopping patience

#         for epoch in range(self.config.epochs):
#             print(f'Epoch {epoch}/{self.config.epochs - 1}')
#             print('-' * 10)
#             epoch_start = time.time()
            
#             # Track validation loss for early stopping
#             current_val_loss = float('inf')

#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     self.model.train()
#                 else:
#                     self.model.eval()

#                 metrics = self._run_epoch(phase, epoch)
#                 epoch_loss = self._log_metrics(metrics, phase, epoch)

#                 if phase == 'train':
#                     print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
#                 else:  # Validation phase
#                     current_val_loss = epoch_loss
#                     self.scheduler.step(epoch_loss)
                    
#                     # Save best model if validation loss improved
#                     if epoch_loss < best_loss:
#                         print(f"Validation loss decreased from {best_loss:.6f} to {epoch_loss:.6f}")
#                         best_loss = epoch_loss
#                         best_model_wts = self._save_if_best(epoch_loss, epoch, best_loss, 
#                                                         copy.deepcopy(self.model.state_dict()))
#                         patience_counter = 0
#                     else:
#                         patience_counter += 1

#             self._print_epoch_summary(epoch_start, current_val_loss, best_loss)

#             # Early stopping check
#             if patience_counter >= max_patience:
#                 print(f"Early stopping triggered after {patience_counter} epochs without improvement")
#                 break

#         # Load best weights and save final model
#         self.model.load_state_dict(best_model_wts)
#         self._save_final_model(best_model_wts, best_loss)
#         return self.model

#     # def _run_epoch(self, phase: str, epoch: int) -> defaultdict:
#     #     """
#     #     Run one epoch of training or validation.
#     #     Returns metrics averaged over the epoch.
#     #     """
#     #     metrics = defaultdict(float)
#     #     total_samples = 0

#     #     with torch.set_grad_enabled(phase == 'train'):
#     #         for inputs, labels in tqdm(self.dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=True):
#     #             inputs = inputs.to(self.device)
#     #             labels = labels.to(self.device)
#     #             batch_size = inputs.size(0)

#     #             # Zero gradients only for training phase
#     #             if phase == 'train':
#     #                 self.optimizer.zero_grad()

#     #             # Forward pass
#     #             outputs = self.model(inputs)

#     #             # Calculate loss
#     #             loss = calc_loss(outputs, labels, metrics, 
#     #                         self.config.bce_weight, 
#     #                         loss_function=self.loss_function)

#     #             # Backward pass and optimization for training phase
#     #             if phase == 'train':
#     #                 loss.backward()
#     #                 self.optimizer.step()

#     #         #     # Accumulate metrics (already scaled by batch_size in calc_loss)
#     #         #     metrics['loss'] += loss.item() * batch_size
#     #         #     total_samples += batch_size

#     #         # # Average metrics over the entire epoch
#     #         # for key in metrics:
#     #         #     metrics[key] = metrics[key] / total_samples

#     #     return metrics
    
#     def _run_epoch(self, phase: str, epoch: int) -> defaultdict:
#         self.model.train(phase == 'train')
#         metrics = defaultdict(float)
#         epoch_loss = 0.0
#         total_samples = 0

#         with torch.set_grad_enabled(phase == 'train'):
#             for i, (inputs, labels) in enumerate(tqdm(self.dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=True, position=0)):
#                 loss = self._process_batch(inputs, labels, metrics, phase)
#                 epoch_loss += loss.item() * inputs.size(0)
#                 total_samples += inputs.size(0)

#                 self._log_batch_metrics(metrics, phase, epoch, i, len(self.dataloaders[phase]))

#         # metrics['epoch_loss'] = epoch_loss / len(self.dataloaders[phase].dataset)

#         # Average metrics
#         for key in metrics:
#             metrics[key] /= total_samples
#         return metrics

#     # def _process_batch(self, inputs, labels, metrics, phase):
#     #     inputs, labels = inputs.to(self.device), labels.to(self.device)
#     #     self.optimizer.zero_grad()
#     #     outputs = self.model(inputs)
#     #     loss = calc_loss(outputs, labels, metrics)
#     #     if phase == 'train':
#     #         loss.backward()
#     #         self.optimizer.step()
#     #     return loss

#     def _process_batch(self, inputs, labels, metrics, phase):
#         """
#         Process a single batch of data: forward pass, loss computation, and backpropagation (if training).
#         """
#         self.optimizer.zero_grad()
#         outputs = self.model(inputs)
#         loss = self.loss_function(outputs, labels)

#         if phase == 'train':
#             loss.backward()
#             self.optimizer.step()

#         metrics['loss'] += loss.item() * inputs.size(0)  # Accumulate loss scaled by batch size
#         return loss

#     def _log_batch_metrics(self, metrics: defaultdict, phase: str, 
#                           epoch: int, batch_idx: int, epoch_len: int):
#         self.writer.add_scalar(f'Batch loss/{phase}', 
#                              metrics['loss'] / (batch_idx + 1), 
#                              epoch * epoch_len + batch_idx)
#         self.writer.add_scalar(f'Batch BCE/{phase}', 
#                              metrics['bce'] / (batch_idx + 1), 
#                              epoch * epoch_len + batch_idx)
#         self.writer.add_scalar(f'Batch {self.config.loss_type}/{phase}', 
#                              metrics[self.config.loss_type] / (batch_idx + 1), 
#                              epoch * epoch_len + batch_idx)
    
#     def _log_metrics(self, metrics: defaultdict, phase: str, epoch: int) -> float:
#         """
#         Log metrics for the current epoch to TensorBoard and MLflow.
#         The metrics are already averaged over the epoch in _run_epoch.
#         """
#         # Metrics are already averaged in _run_epoch, no need to divide by samples
#         epoch_loss = metrics['loss']
#         epoch_custom = metrics[self.config.loss_type]
#         epoch_bce = metrics['bce']

#         # Log to TensorBoard
#         self.writer.add_scalar(f'Epoch loss/{phase}', epoch_loss, epoch)
#         self.writer.add_scalar(f'Epoch BCE/{phase}', epoch_bce, epoch)
#         self.writer.add_scalar(f'Epoch {self.config.loss_type}/{phase}', epoch_custom, epoch)
#         self.writer.add_scalar(f'Learning_rate/{phase}', self.optimizer.param_groups[0]['lr'], epoch)

#         # Log to MLflow
#         mlflow.log_metric(key=f"{phase}_epoch_loss", value=epoch_loss, step=epoch)
#         mlflow.log_metric(key=f"{phase}_epoch_bce", value=epoch_bce, step=epoch)
#         mlflow.log_metric(key=f"{phase}_epoch_{self.config.loss_type}", value=epoch_custom, step=epoch)

#         return epoch_loss
    
#     def _save_if_best(self, epoch_loss: float, epoch: int, 
#                      best_loss: float, best_weights: Dict) -> Dict:
#         if epoch_loss < best_loss:
#             print(f"Saving best model from epoch {epoch} with loss: {epoch_loss:.4f}")
#             best_weights = copy.deepcopy(self.model.state_dict())
#             mlflow.log_metric(key="best_loss", value=epoch_loss, step=epoch)
#             self._save_checkpoint(epoch, epoch_loss)
#         return best_weights
    
#     def _save_checkpoint(self, epoch: int, loss: float):
#         os.makedirs('models', exist_ok=True)
#         save_path = f'models/best_model_epoch_{epoch}_loss_{loss:.4f}.pth'
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': loss,
#         }, save_path)
    
#     def _save_final_model(self, best_weights: Dict, best_loss: float):
#         self.model.load_state_dict(best_weights)
#         save_path = f'models/final_model_epochs_{self.config.epochs}_loss_{best_loss:.4f}.pth'
#         torch.save({
#             'epochs_completed': self.config.epochs,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'final_loss': best_loss,
#         }, save_path)
        
#         timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
#         mlflow.pytorch.log_model(self.model, f"unet-{timestamp}")
#         self.writer.close()
# class TrainingManager:
#     def __init__(self, config: Config, transform=None):
#         self.config = config
#         self.transform = transform
#         self.device = self._setup_device()
#         self.model = self._setup_model()
#         self.dataloaders = self._setup_dataloaders()
#         self.optimizer = self._setup_optimizer()
#         self.scheduler = self._setup_scheduler()
#         self.writer = self._setup_tensorboard()
#         self.loss_function = self._setup_loss_function()
        
#     def _setup_device(self) -> torch.device:
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             print(f'Using GPU: {torch.cuda.get_device_name(0)}')
#         elif torch.backends.mps.is_available():
#             device = torch.device("mps")
#             print("Using MPS")
#         else:
#             device = torch.device("cpu")
#             print("Using CPU")
#         return device
    
#     def _setup_model(self) -> nn.Module:
#         model = UNet(1).to(self.device)
#         return model
    
#     def _print_epoch_summary(self, epoch_start: float, epoch_loss: float, best_loss: float):
#         epoch_time = time.time() - epoch_start
#         print(f'Epoch completed in {epoch_time:.0f}s')
#         print(f'Current loss: {epoch_loss:.4f}, Best loss so far: {best_loss:.4f}')

#     def _setup_dataloaders(self) -> Dict[str, DataLoader]:
#         train_dataset, val_dataset = self._split_dataset()
#         return {
#             'train': DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
#             'val': DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
#         }
    
#     def _split_dataset(self) -> Tuple[CoronaryDataset, CoronaryDataset]:
#         path = Path(self.config.path)
#         images = sorted(list(path.glob('imgs/*')))
#         masks = sorted(list(path.glob('masks/*')))
        
#         split_idx = int(len(images) * 0.8)
#         train_images, val_images = images[:split_idx], images[split_idx:]
#         train_masks, val_masks = masks[:split_idx], masks[split_idx:]
        
#         train_dataset = CoronaryDataset(train_images, train_masks, train=True, transform=self.transform)
#         val_dataset = CoronaryDataset(val_images, val_masks, train=False, transform=self.transform)
        
#         return train_dataset, val_dataset
    
#     def _setup_optimizer(self) -> optim.Optimizer:
#         optimizers = {
#             'adam': optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5),
#             'sgd': optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=1e-5)
#         }
#         if self.config.optimizer_name not in optimizers:
#             raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")
#         return optimizers[self.config.optimizer_name]
    
#     def _setup_scheduler(self) -> optim.lr_scheduler.ReduceLROnPlateau:
#         return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
    
#     def _setup_tensorboard(self) -> SummaryWriter:
#         return SummaryWriter(comment=self._get_tensorboard_comment())
    
#     def _get_tensorboard_comment(self) -> str:
#         return (f" LR_{self.config.lr}_optimizer_{self.config.optimizer_name}"
#                 f"_batch_size_{self.config.batch_size}_epochs_{self.config.epochs}"
#                 f"_bce_weight_{self.config.bce_weight}")
    
#     def _setup_loss_function(self):
#         loss_functions = {
#             'dice': Dice(),
#             'iou': IoU(),
#             'debugdice': DebugDice()
#         }
#         if self.config.loss_type not in loss_functions:
#             raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
#         return loss_functions[self.config.loss_type]

#     def train(self) -> nn.Module:
#         """
#         Train the model and return the best performing version.
#         """
#         best_model_wts = copy.deepcopy(self.model.state_dict())
#         best_loss = float('inf')
#         patience_counter = 0
#         max_patience = 10  # Early stopping patience

#         for epoch in range(self.config.epochs):
#             print(f'Epoch {epoch}/{self.config.epochs - 1}')
#             print('-' * 10)
#             epoch_start = time.time()
            
#             # Track validation loss for early stopping
#             current_val_loss = float('inf')

#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     self.model.train()
#                 else:
#                     self.model.eval()

#                 metrics = self._run_epoch(phase, epoch)
#                 epoch_loss = self._log_metrics(metrics, phase, epoch)

#                 if phase == 'train':
#                     print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
#                 else:  # Validation phase
#                     current_val_loss = epoch_loss
#                     self.scheduler.step(epoch_loss)
                    
#                     # Save best model if validation loss improved
#                     if epoch_loss < best_loss:
#                         print(f"Validation loss decreased from {best_loss:.6f} to {epoch_loss:.6f}")
#                         best_loss = epoch_loss
#                         best_model_wts = copy.deepcopy(self.model.state_dict())
#                         self._save_checkpoint(epoch, epoch_loss)
#                         patience_counter = 0
#                     else:
#                         patience_counter += 1

#             self._print_epoch_summary(epoch_start, current_val_loss, best_loss)

#             # Early stopping check
#             if patience_counter >= max_patience:
#                 print(f"Early stopping triggered after {patience_counter} epochs without improvement")
#                 break

#         # Load best weights and save final model
#         self.model.load_state_dict(best_model_wts)
#         self._save_final_model(best_model_wts, best_loss)
#         return self.model
    
#     def _run_epoch(self, phase: str, epoch: int) -> defaultdict:
#         """
#         Run one epoch of training or validation.
#         Returns metrics averaged over the epoch.
#         """
#         metrics = defaultdict(float)
#         total_samples = 0

#         # Set model to train or eval mode
#         self.model.train(phase == 'train')

#         with torch.set_grad_enabled(phase == 'train'):
#             for inputs, labels in tqdm(self.dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=True):
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 batch_size = inputs.size(0)

#                 # Zero gradients only for training phase
#                 if phase == 'train':
#                     self.optimizer.zero_grad()

#                 # Forward pass
#                 outputs = self.model(inputs)

#                 # Calculate loss
#                 loss = calc_loss(outputs, labels, metrics, 
#                              self.config.bce_weight, 
#                              loss_function=self.loss_function)

#                 # Backward pass and optimization for training phase
#                 if phase == 'train':
#                     loss.backward()
#                     self.optimizer.step()

#                 # Accumulate metrics (already scaled by batch_size in calc_loss)
#                 metrics['loss'] += loss.item() * batch_size
#                 total_samples += batch_size

#         # Average metrics over the entire epoch
#         for key in metrics:
#             metrics[key] = metrics[key] / total_samples

#         return metrics
    
#     def _log_metrics(self, metrics: defaultdict, phase: str, epoch: int) -> float:
#         """
#         Log metrics for the current epoch to TensorBoard and MLflow.
#         """
#         # Get metrics from the dictionary
#         epoch_loss = metrics['loss']
#         epoch_custom = metrics.get(self.config.loss_type, 0.0)
#         epoch_bce = metrics.get('bce', 0.0)

#         # Log to TensorBoard
#         self.writer.add_scalar(f'Epoch loss/{phase}', epoch_loss, epoch)
#         self.writer.add_scalar(f'Epoch BCE/{phase}', epoch_bce, epoch)
#         self.writer.add_scalar(f'Epoch {self.config.loss_type}/{phase}', epoch_custom, epoch)
#         self.writer.add_scalar(f'Learning_rate/{phase}', self.optimizer.param_groups[0]['lr'], epoch)

#         # Log to MLflow
#         mlflow.log_metric(key=f"{phase}_epoch_loss", value=epoch_loss, step=epoch)
#         mlflow.log_metric(key=f"{phase}_epoch_bce", value=epoch_bce, step=epoch)
#         mlflow.log_metric(key=f"{phase}_epoch_{self.config.loss_type}", value=epoch_custom, step=epoch)

#         return epoch_loss
    
#     def _save_if_best(self, epoch_loss: float, epoch: int, 
#                      best_loss: float, best_weights: Dict) -> Dict:
#         if epoch_loss < best_loss:
#             print(f"Saving best model from epoch {epoch} with loss: {epoch_loss:.4f}")
#             best_weights = copy.deepcopy(self.model.state_dict())
#             mlflow.log_metric(key="best_loss", value=epoch_loss, step=epoch)
#             self._save_checkpoint(epoch, epoch_loss)
#         return best_weights
    
#     def _save_checkpoint(self, epoch: int, loss: float):
#         os.makedirs('models', exist_ok=True)
#         save_path = f'models/best_model_epoch_{epoch}_loss_{loss:.4f}.pth'
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': loss,
#             'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
#         }, save_path)
#         print(f"Checkpoint saved at {save_path}")
    
#     def _save_final_model(self, best_weights: Dict, best_loss: float):
#         self.model.load_state_dict(best_weights)
#         save_path = f'models/final_model_epochs_{self.config.epochs}_loss_{best_loss:.4f}.pth'
#         torch.save({
#             'epochs_completed': self.config.epochs,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'final_loss': best_loss,
#             'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
#         }, save_path)
        
#         timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
#         mlflow.pytorch.log_model(self.model, f"unet-{timestamp}")
#         self.writer.close()
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import os
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import mlflow
from datetime import datetime
from utils.dice_score import compute_loss

class TrainingManager:
    def __init__(self, config: Config, transform=None):
        self.config = config
        self.transform = transform
        self.device = self._setup_device()
        self.model = UNet(n_channels=1, n_classes=1, bilinear=False).to(self.device)
        self.dataloaders = self._setup_dataloaders()
        self.train_loader = self.dataloaders['train']
        self.val_loader = self.dataloaders['val']
        self.global_step = 0
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.lr, steps_per_epoch=len(self.dataloaders), epochs=self.config.epochs)
        self.loss_function = self._setup_loss_function()
        self.best_loss = float('inf')
        self.best_model_wts = None
        self.global_step = 0
        self.patience_counter = 0
        self.max_patience = 10

        self.writer = SummaryWriter(comment=self._get_tensorboard_comment())
        
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using MPS")
            return torch.device("mps")
        print("Using CPU")
        return torch.device("cpu")
    
    # def _setup_dataloaders(self) -> dict:
    #     train_dataset, val_dataset = self._split_dataset()
    #     return {
    #         'train': DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
    #         'val': DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
    #     }
    
    # def _split_dataset(self):
    #     path = Path(self.config.path)
    #     images = sorted(list(path.glob('imgs/*')))
    #     masks = sorted(list(path.glob('masks/*')))
    #     split_idx = int(len(images) * 0.8)
    #     return (
    #         CoronaryDataset(images[:split_idx], masks[:split_idx], train=True, transform=self.transform),
    #         CoronaryDataset(images[split_idx:], masks[split_idx:], train=False, transform=self.transform)
    #     )

    def _setup_dataloaders(self) -> dict:
        dataset = self._load_dataset()
        
        # Split dataset
        n_val = int(len(dataset) * self.config.val_percent)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        # DataLoader settings
        num_workers = min(4, os.cpu_count() or 1)
        loader_args = dict(batch_size=self.config.batch_size, num_workers=num_workers)
        if torch.cuda.is_available():
            loader_args['pin_memory'] = True

        return {
            'train': DataLoader(train_dataset, shuffle=True, **loader_args),
            'val': DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)  # Ensure consistent batch sizes
        }
    
    def _load_dataset(self):
        """Loads dataset from config path and applies transformations."""
        path = Path(self.config.path)
        images = sorted(list(path.glob('imgs/*')))
        masks = sorted(list(path.glob('masks/*')))
        return CoronaryDataset(images, masks, train=True, transform=self.transform)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5),
            'sgd': optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=1e-5)
        }
        return optimizers.get(self.config.optimizer_name, ValueError(f"Unsupported optimizer: {self.config.optimizer_name}"))
    
    def _setup_loss_function(self):
        loss_functions = {'dice': Dice(), 'iou': IoU(), 'debugdice': DebugDice()}
        return loss_functions.get(self.config.loss_type, ValueError(f"Unsupported loss type: {self.config.loss_type}"))
    
    def _get_tensorboard_comment(self) -> str:
        return f"LR_{self.config.lr}_opt_{self.config.optimizer_name}_batch_{self.config.batch_size}_epochs_{self.config.epochs}"  
    
    def train(self):
        """Train the model and return the best performing version."""
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            epoch_start = time.time()

            # Training phase
            train_loss = self._run_epoch(phase='train', epoch=epoch)

            # Validation twice per epoch
            # if self.global_step % (len(self.dataloaders) // 2) == 0:
            #     val_loss = self._run_epoch(phase='val', epoch=epoch)
            # else:
            #     val_loss = float('inf')  # Placeholder for validation loss

            if self.global_step % (len(self.dataloaders) // 2) == 0:
                val_loss = self._run_epoch(phase='val', epoch=epoch)
                self.scheduler.step()  # OneCycleLR doesn’t need val_score
                print.info(f'Validation Dice score: {val_loss}')
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step(val_loss)  # Adjust LR based on validation loss
            else:
                self.scheduler.step()  # OneCycleLR updates per iteration

            # Save best model if validation improved
            if val_loss < self.best_loss:
                print(f"Validation loss improved: {self.best_loss:.6f} → {val_loss:.6f}")
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1

            # Print summary
            self._print_epoch_summary(epoch_start, train_loss, val_loss)

            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break

        # Load best model before returning
        self.model.load_state_dict(self.best_model_wts)
        return self.model

    def _run_epoch(self, phase, epoch):
        """Runs one epoch for either training or validation."""
        is_train = phase == 'train'
        self.model.train() if is_train else self.model.eval()
        
        loader = self.dataloaders['train'] if is_train else self.dataloaders['val']
        total_loss = 0

        with torch.inference_mode(not is_train):  # Disable gradients in validation
            with tqdm(total=len(loader.dataset), desc=f'{phase.capitalize()} Epoch {epoch}', unit='img') as pbar:
                for batch in loader:
                    if isinstance(batch, (tuple, list)):
                        images, masks = batch
                    else:
                        images, masks = batch['image'], batch['mask']  # If dataset returns a dictionary

                    images, masks = images.to(self.device), masks.to(self.device)
                    images = images.to(dtype=torch.float32, memory_format=torch.channels_last)
                    masks = masks.to(dtype=torch.long)

                    with torch.amp.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.config.amp):
                        preds = self.model(images)
                        loss = self.loss_function(preds, masks)  # Use correct loss function

                    if is_train:
                        self.optimizer.zero_grad(set_to_none=True)
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()

                    total_loss += loss.item()
                    pbar.update(images.shape[0])
                    pbar.set_postfix(loss=loss.item())

                    self.global_step += 1
                    if self.writer:
                        self.writer.add_scalar(f'{phase} loss', loss.item(), self.global_step)

        return total_loss / len(loader)

    

    def _print_epoch_summary(self, epoch_start: float, epoch_loss: float, best_loss: float):
        epoch_time = time.time() - epoch_start
        print(f'Epoch completed in {epoch_time:.0f}s')
        print(f'Current loss: {epoch_loss:.4f}, Best loss so far: {best_loss:.4f}')
    # def train(self) -> nn.Module:
    #     best_model_wts = copy.deepcopy(self.model.state_dict())
    #     best_loss, patience_counter, max_patience = float('inf'), 0, 10

    #     for epoch in range(self.config.epochs):
    #         print(f'Epoch {epoch}/{self.config.epochs - 1}\n' + '-' * 10)
    #         epoch_start = time.time()
    #         current_val_loss = float('inf')

    #         for phase in ['train', 'val']:
    #             self.model.train() if phase == 'train' else self.model.eval()
    #             metrics = self._run_epoch(phase, epoch)
    #             epoch_loss = self._log_metrics(metrics, phase, epoch)
    #             if phase == 'val':
    #                 current_val_loss = epoch_loss
    #                 self.scheduler.step(epoch_loss)
    #                 if epoch_loss < best_loss:
    #                     print(f"Validation loss improved from {best_loss:.6f} to {epoch_loss:.6f}")
    #                     best_loss, best_model_wts, patience_counter = epoch_loss, copy.deepcopy(self.model.state_dict()), 0
    #                     self._save_checkpoint(epoch, epoch_loss)
    #                 else:
    #                     patience_counter += 1
            
    #         print(f'Epoch completed in {time.time() - epoch_start:.0f}s, Current loss: {current_val_loss:.4f}, Best loss: {best_loss:.4f}')
    #         if patience_counter >= max_patience:
    #             print(f"Early stopping triggered after {patience_counter} epochs without improvement")
    #             break
        
    #     self.model.load_state_dict(best_model_wts)
    #     self._save_final_model(best_loss)
    #     return self.model
    
    # def _run_epoch(self, phase: str, epoch: int) -> defaultdict:
    #     metrics, total_samples = defaultdict(float), 0
    #     with torch.set_grad_enabled(phase == 'train'):
    #         for inputs, labels in tqdm(self.dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=True):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             batch_size = inputs.size(0)
    #             if phase == 'train': self.optimizer.zero_grad()
    #             outputs, loss = self.model(inputs), calc_loss(self.model(inputs), labels, metrics, self.config.bce_weight, loss_function=self.loss_function)
    #             if phase == 'train': loss.backward(), self.optimizer.step()
    #             metrics['loss'] += loss.item() * batch_size
    #             total_samples += batch_size
    #     for key in metrics: metrics[key] /= total_samples
    #     return metrics
    
    def _log_metrics(self, metrics: defaultdict, phase: str, epoch: int) -> float:
        epoch_loss = metrics['loss']
        self.writer.add_scalar(f'Epoch loss/{phase}', epoch_loss, epoch)
        mlflow.log_metric(key=f"{phase}_epoch_loss", value=epoch_loss, step=epoch)
        return epoch_loss
    
    def _save_checkpoint(self, epoch: int, loss: float):
        os.makedirs('models', exist_ok=True)
        save_path = f'models/best_model_epoch_{epoch}_loss_{loss:.4f}.pth'
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, save_path)
        print(f"Checkpoint saved at {save_path}")
    
    def _save_final_model(self, best_loss: float):
        save_path = f'models/final_model_epochs_{self.config.epochs}_loss_{best_loss:.4f}.pth'
        torch.save({'epochs_completed': self.config.epochs, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'final_loss': best_loss}, save_path)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
        mlflow.pytorch.log_model(self.model, f"unet-{timestamp}")
        self.writer.close()

def parse_list_or_value(value):
        try:
            # Spróbuj przekonwertować wartość na JSON
            parsed_value = json.loads(value)
            # Jeśli wartość nie jest listą, zwróć ją jako listę
            if not isinstance(parsed_value, list):
                return [parsed_value]
            return parsed_value
        except json.JSONDecodeError:
            # Jeśli wartość nie jest poprawnym JSON, zwróć ją jako listę
            return [value]
        

def parse_arguments():
    """Parse command line arguments with improved validation and help messages."""
    parser = argparse.ArgumentParser(
        description='Training script for coronary artery segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--path',
        type=parse_list_or_value,
        required=True,
        help='Dataset path(s). Can be single path or list: "/data/path" or "[/path1, /path2]"'
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        '--epochs',
        type=parse_list_or_value,
        default='10',
        help='Number of epochs. Single value or list: "10" or "[10, 20]"'
    )
    training_group.add_argument(
        '--lr',
        type=parse_list_or_value,
        default='0.001',
        help='Learning rate(s). Single value or list: "0.001" or "[0.001, 0.0001]"'
    )
    training_group.add_argument(
        '--batch_size',
        type=parse_list_or_value,
        default='4',
        help='Batch size(s). Single value or list: "4" or "[4, 8]"'
    )

    training_group.add_argument( 
        '--val_percent',
        type=float,
        default=0.2,
        help='Percentage of data to use for validation'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--optimizer',
        type=parse_list_or_value,
        default='adam',
        choices=['adam', 'sgd', '[adam,sgd]'],
        help='Optimizer choice(s). Single value or list: "adam" or "[adam, sgd]"'
    )
    model_group.add_argument(
        '--loss_type',
        type=parse_list_or_value,
        default='dice',
        choices=['dice', 'iou', 'debugdice', '[dice,iou]'],
        help='Loss function type(s). Single value or list: "dice" or "[dice, iou]"'
    )
    model_group.add_argument(
        '--bce_weight',
        type=parse_list_or_value,
        default='0.5',
        help='BCE weight in loss function. Single value or list: "0.5" or "[0.3, 0.5]"'
    )
    
    # Data handling
    data_group = parser.add_argument_group('Data Handling')
    resolution_group = data_group.add_mutually_exclusive_group(required=True)
    resolution_group.add_argument(
        '--quarterres',
        action='store_true',
        help='Use half resolution (128x128)'
    )
    resolution_group.add_argument(
        '--halfres',
        action='store_true',
        help='Use half resolution (256x256)'
    )
    resolution_group.add_argument(
        '--fullres',
        action='store_true',
        help='Use full resolution (512x512)'
    )
    data_group.add_argument(
        '--shuffle',
        type=parse_list_or_value,
        default='True',
        help='Whether to shuffle the dataset. Single value or list: "True" or "[True, False]"'
    )
    
    # Execution control
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Perform a dry run (print parameters without training)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dryrun:
        print("Dry run mode - parameters will be displayed without training")
    
    return args

def get_resolution(args):
    """Determine resolution based on command line arguments."""
    if args.quarterres:
        return (128, 128)
    elif args.halfres:
        return (256, 256)
    elif args.fullres:
        return (512, 512)
    else:
        raise ValueError("Must specify either --halfres or --fullres")

def get_training_parameters(args):
    """Convert parsed arguments into training parameters dictionary."""
    return {
        'path': args.path if isinstance(args.path, list) else [args.path],
        'lr': _ensure_list(args.lr),
        'batch_size': _ensure_list(args.batch_size),
        'shuffle': _ensure_list(args.shuffle),
        'epochs': _ensure_list(args.epochs),
        'optimizer': _ensure_list(args.optimizer),
        'bce_weight': _ensure_list(args.bce_weight),
        'loss_type': _ensure_list(args.loss_type),
        'val_percent': _ensure_list(args.val_percent)
    }

def _ensure_list(value):
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    return [value]

def run_training(parameters, resolution):
    """Run the training process with MLflow tracking."""
    mlflow.set_tracking_uri(uri=os.environ.get("MLFLOW_BACKEND"))
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    experiment_name = f"exp-coronary-{timestamp}"
    
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name=experiment_name)
    
    with mlflow.start_run(run_name=f"run-{timestamp}"):
        for params in itertools.product(*parameters.values()):
            config = Config(*params)
            print(f"Training with config: {config.to_dict}")
            trainer = TrainingManager(config, transform=transforms.Resize(resolution))
            trainer.train()

def main():
    load_dotenv()
    
    args = parse_arguments()
    resolution = get_resolution(args)
    parameters = get_training_parameters(args)
    # print("Training parameters:")
    # print(parameters)
    # print(f"Resolution: {resolution}")
    if not args.dryrun:
        run_training(parameters, resolution)

if __name__ == '__main__':
    main()