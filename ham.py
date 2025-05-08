"""
HAM10000 Skin Lesion Classification with ResNet + Transformer (BoTNet style)
Enhanced version with improved preprocessing, code organization, and documentation

This notebook implements a skin lesion classification model using the HAM10000 dataset.
It combines ResNet features with Transformer attention mechanisms for improved performance.
"""

# Standard library imports
import os
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils import class_weight

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import timm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration dictionary for experiment settings
CONFIG = {
    'data': {
        'dataset_path': '/content/HAM10000',
        'metadata_file': 'ham10000_metadata.csv',
        'images_dir': 'images',
        'img_size': 224,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 42
    },
    'training': {
        'batch_size': 32,
        'num_workers': 2,
        'epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'early_stopping_patience': 5,
        'use_class_weights': True,
        'use_weighted_sampler': True
    },
    'model': {
        'backbone': 'resnet18',
        'pretrained': True,
        'transformer_dim': 512,
        'transformer_heads': 8,
        'transformer_layers': 2,
        'dropout_rate': 0.2
    },
    'augmentation': {
        'use_augmentation': True,
        'hflip_prob': 0.5,
        'vflip_prob': 0.3,
        'rotate_degrees': 20,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
}

# Set seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value for random number generators
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seeds set to {seed} for reproducibility")

# Data exploration and visualization functions
def explore_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Explore the dataset and return statistics.
    
    Args:
        df: Pandas DataFrame containing the dataset metadata
        
    Returns:
        Dictionary containing dataset statistics
    """
    logger.info("Exploring dataset...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # Get class distribution
    class_distribution = df['dx'].value_counts()
    class_percentages = (class_distribution / len(df) * 100).round(2)
    
    # Get age and gender distribution
    age_stats = df['age'].describe()
    gender_distribution = df['sex'].value_counts()
    
    # Get localization distribution
    localization_distribution = df['localization'].value_counts()
    
    # Return statistics
    stats = {
        'total_samples': len(df),
        'missing_values': missing_values,
        'class_distribution': class_distribution,
        'class_percentages': class_percentages,
        'age_stats': age_stats,
        'gender_distribution': gender_distribution,
        'localization_distribution': localization_distribution
    }
    
    logger.info(f"Dataset exploration completed. Total samples: {len(df)}")
    return stats

def visualize_class_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the class distribution in the dataset.
    
    Args:
        df: Pandas DataFrame containing the dataset metadata
    """
    plt.figure(figsize=(12, 6))
    class_counts = df['dx'].value_counts()
    
    # Create bar plot
    ax = class_counts.plot(kind='bar', color='skyblue')
    plt.title('Class Distribution in HAM10000 Dataset', fontsize=14)
    plt.xlabel('Skin Lesion Type', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for i, count in enumerate(class_counts):
        ax.text(i, count + 5, str(count), ha='center', fontsize=10)
    
    # Add percentage labels
    percentages = (class_counts / len(df) * 100).round(1)
    for i, percentage in enumerate(percentages):
        ax.text(i, class_counts[i] / 2, f"{percentage}%", ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Create a pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05] * len(class_counts))
    plt.title('Class Distribution (%) in HAM10000 Dataset', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_sample_images(df: pd.DataFrame, image_dir: str, num_samples: int = 2) -> None:
    """
    Visualize sample images from each class.
    
    Args:
        df: Pandas DataFrame containing the dataset metadata
        image_dir: Directory containing the images
        num_samples: Number of samples to show per class
    """
    classes = df['dx'].unique()
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(num_samples*3, len(classes)*3))
    
    for i, cls in enumerate(classes):
        # Get samples from this class
        samples = df[df['dx'] == cls].sample(num_samples)
        
        for j, (_, row) in enumerate(samples.iterrows()):
            img_path = os.path.join(image_dir, row['image_id'] + '.jpg')
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{cls}: {row['dx_name']}")
                axes[i, j].axis('off')
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                axes[i, j].text(0.5, 0.5, f"Error loading image", ha='center')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Enhanced dataset class with validation
class HAM10000Dataset(Dataset):
    """
    Dataset class for the HAM10000 skin lesion dataset with enhanced preprocessing.
    
    Attributes:
        df: Pandas DataFrame containing the dataset metadata
        image_dir: Directory containing the images
        transform: Torchvision transforms to apply to the images
        label_map: Mapping from class names to indices
        reverse_map: Mapping from indices to class names
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 image_dir: str, 
                 transform: Optional[Callable] = None,
                 validate_files: bool = True) -> None:
        """
        Initialize the HAM10000Dataset.
        
        Args:
            df: Pandas DataFrame containing the dataset metadata
            image_dir: Directory containing the images
            transform: Torchvision transforms to apply to the images
            validate_files: Whether to validate that all image files exist
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
        # Create label mappings
        self.label_map = {l: i for i, l in enumerate(sorted(df['dx'].unique()))}
        self.reverse_map = {i: l for l, i in self.label_map.items()}
        
        # Store class names for reference
        self.class_names = df['dx'].unique()
        
        # Validate files if requested
        if validate_files:
            self._validate_files()
    
    def _validate_files(self) -> None:
        """
        Validate that all image files exist and are readable.
        Removes entries with missing or corrupted files from the dataframe.
        """
        valid_indices = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Validating files"):
            img_path = os.path.join(self.image_dir, row['image_id'] + ".jpg")
            if os.path.exists(img_path):
                try:
                    # Try to open the image to check if it's valid
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's a valid image
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Corrupted image found at {img_path}: {e}")
            else:
                logger.warning(f"Image not found: {img_path}")
        
        # Update dataframe to include only valid images
        original_len = len(self.df)
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        if len(self.df) < original_len:
            logger.warning(f"Removed {original_len - len(self.df)} invalid images from dataset")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple containing the image tensor and class label
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'] + ".jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.label_map[row['dx']]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image and the label in case of error
            if self.transform:
                black_image = torch.zeros((3, CONFIG['data']['img_size'], CONFIG['data']['img_size']))
            else:
                black_image = Image.new('RGB', (CONFIG['data']['img_size'], CONFIG['data']['img_size']))
                
            return black_image, self.label_map[row['dx']]

# Enhanced data transformations with advanced augmentation
def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get data transformations for training with advanced augmentation.
    
    Args:
        config: Configuration dictionary containing augmentation settings
        
    Returns:
        Composed transforms for training
    """
    img_size = config['data']['img_size']
    aug_config = config['augmentation']
    
    if not aug_config['use_augmentation']:
        # Basic transforms without augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    # Advanced augmentation pipeline
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=aug_config['hflip_prob']),
        transforms.RandomVerticalFlip(p=aug_config['vflip_prob']),
        transforms.RandomRotation(aug_config['rotate_degrees']),
        transforms.ColorJitter(
            brightness=aug_config['brightness'],
            contrast=aug_config['contrast'],
            saturation=aug_config['saturation'],
            hue=aug_config['hue']
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])

def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get data transformations for validation/testing.
    
    Args:
        config: Configuration dictionary containing data settings
        
    Returns:
        Composed transforms for validation/testing
    """
    img_size = config['data']['img_size']
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])

# Data preparation functions
def prepare_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data by loading and splitting into train, validation, and test sets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of train, validation, and test DataFrames
    """
    # Load metadata
    data_path = config['data']['dataset_path']
    metadata_file = os.path.join(data_path, config['data']['metadata_file'])
    
    logger.info(f"Loading metadata from {metadata_file}")
    df = pd.read_csv(metadata_file)
    
    # Clean data
    df = df[df['dx'].notna()].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} valid samples")
    
    # Create stratified train/val/test split
    val_size = config['data']['val_size']
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    
    # First split: train+val and test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        stratify=df['dx'],
        random_state=random_state
    )
    
    # Second split: train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),  # Adjusted to get the right proportion
        stratify=train_val_df['dx'],
        random_state=random_state
    )
    
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
    
    return train_df, val_df, test_df

def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of train, validation, and test DataLoaders, plus dataset info
    """
    # Get paths and parameters
    data_path = config['data']['dataset_path']
    image_dir = os.path.join(data_path, config['data']['images_dir'])
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    # Create datasets
    train_dataset = HAM10000Dataset(
        train_df, 
        image_dir, 
        transform=get_train_transforms(config),
        validate_files=True
    )
    
    val_dataset = HAM10000Dataset(
        val_df, 
        image_dir, 
        transform=get_val_transforms(config),
        validate_files=True
    )
    
    test_dataset = HAM10000Dataset(
        test_df, 
        image_dir, 
        transform=get_val_transforms(config),
        validate_files=True
    )
    
    # Create samplers for handling class imbalance
    sampler = None
    if config['training']['use_weighted_sampler']:
        # Calculate class weights
        class_counts = train_df['dx'].value_counts().sort_index()
        weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
        
        # Assign weight to each sample
        sample_weights = torch.tensor([
            weights[train_dataset.label_map[train_df.iloc[i]['dx']]] 
            for i in range(len(train_df))
        ])
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_df),
            replacement=True
        )
        logger.info("Using weighted sampler for training")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,  # Only shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create dataset info dictionary
    dataset_info = {
        'num_classes': len(train_df['dx'].unique()),
        'class_names': sorted(train_df['dx'].unique()),
        'class_to_idx': train_dataset.label_map,
        'idx_to_class': train_dataset.reverse_map,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    }
    
    logger.info(f"Created data loaders with {len(train_loader)} training batches")
    return train_loader, val_loader, test_loader, dataset_info

# Enhanced model architecture
class ResNetTransformer(nn.Module):
    """
    Enhanced ResNet + Transformer model for skin lesion classification.
    
    This model combines ResNet features with Transformer attention mechanisms
    in a BoTNet-style architecture for improved performance.
    
    Attributes:
        backbone: ResNet backbone for feature extraction
        transformer: Transformer encoder for attention-based feature refinement
        fc: Fully connected layer for classification
        dropout: Dropout layer for regularization
    """
    
    def __init__(self, 
                 num_classes: int, 
                 config: Dict[str, Any]) -> None:
        """
        Initialize the ResNetTransformer model.
        
        Args:
            num_classes: Number of output classes
            config: Model configuration dictionary
        """
        super().__init__()
        
        # Get configuration parameters
        backbone_name = config['model']['backbone']
        pretrained = config['model']['pretrained']
        transformer_dim = config['model']['transformer_dim']
        transformer_heads = config['model']['transformer_heads']
        transformer_layers = config['model']['transformer_layers']
        dropout_rate = config['model']['dropout_rate']
        
        # Initialize backbone
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract backbone layers (everything except the final FC layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=transformer_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=False
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers
        )
        
        # Add classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized {backbone_name} + Transformer model with {num_classes} output classes")

    def _init_weights(self) -> None:
        """Initialize the weights of the transformer and FC layer."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using ResNet backbone
        x = self.backbone(x)  # Shape: B x C x H x W
        
        # Reshape for transformer
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)  # Shape: (H*W, B, C)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling over spatial dimensions
        x = x.mean(0)  # Shape: (B, C)
        
        # Apply dropout and classification head
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Training and evaluation functions
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to use for training
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # Step scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    # Calculate final metrics
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy
    }
    
    return metrics

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use for evaluation
        dataset_info: Dictionary containing dataset information
        
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds) * 100
    
    # Generate classification report
    class_names = dataset_info['class_names']
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, 
        all_preds, 
        average='weighted'
    )
    
    # Compile results
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results

# Visualization functions
def plot_loss_curve(train_losses: List[float], val_losses: List[float]) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', label='Training Loss')
    plt.plot(val_losses, marker='o', label='Validation Loss')
    plt.title("Loss Curves", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy_curve(train_accs: List[float], val_accs: List[float]) -> None:
    """
    Plot training and validation accuracy curves.
    
    Args:
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, marker='o', label='Training Accuracy')
    plt.plot(val_accs, marker='o', label='Validation Accuracy')
    plt.title("Accuracy Curves", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix", fontsize=14)
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()

# Model training with early stopping
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    dataset_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train the model with early stopping.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        dataset_info: Dictionary containing dataset information
        
    Returns:
        Dictionary containing training history and best model state
    """
    # Get training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    patience = config['training']['early_stopping_patience']
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function with class weights if specified
    if config['training']['use_class_weights']:
        # Calculate class weights
        class_counts = np.bincount([dataset_info['class_to_idx'][cls] for cls in dataset_info['class_names']])
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(np.arange(len(dataset_info['class_names']))),
            y=np.argmax(np.eye(len(dataset_info['class_names']))[np.arange(len(dataset_info['class_names']))], axis=1)
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using weighted loss function")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience // 2,
        verbose=True
    )
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs on {device}")
    for epoch in range(epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device, dataset_info)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print progress
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        
        # Check for improvement
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Return history and best model state
    return {
        'history': history,
        'best_model_state': best_model_state
    }

# Main experiment function
def run_experiment(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline.
    
    Args:
        config: Configuration dictionary (optional, uses default if None)
        
    Returns:
        Dictionary containing experiment results
    """
    # Use default config if none provided
    if config is None:
        config = CONFIG
    
    # Set random seeds for reproducibility
    set_seed(config['data']['random_state'])
    
    # Prepare data
    train_df, val_df, test_df = prepare_data(config)
    
    # Explore dataset
    stats = explore_dataset(train_df)
    visualize_class_distribution(train_df)
    
    # Create data loaders
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        train_df, val_df, test_df, config
    )
    
    # Initialize model
    model = ResNetTransformer(
        num_classes=dataset_info['num_classes'],
        config=config
    )
    
    # Train model
    training_results = train_model(
        model, train_loader, val_loader, config, dataset_info
    )
    
    # Plot training curves
    history = training_results['history']
    plot_loss_curve(history['train_loss'], history['val_loss'])
    plot_accuracy_curve(history['train_acc'], history['val_acc'])
    
    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    test_results = evaluate(model, test_loader, criterion, device, dataset_info)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'], dataset_info['class_names'])
    
    # Save model
    model_path = f"ham10000_resnet_transformer.pt"
    torch.save({
        'model_state_dict': training_results['best_model_state'],
        'config': config,
        'dataset_info': dataset_info
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Print final results
    logger.info(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    logger.info(f"Test Precision: {test_results['precision']:.2f}%")
    logger.info(f"Test Recall: {test_results['recall']:.2f}%")
    logger.info(f"Test F1 Score: {test_results['f1']:.2f}%")
    
    # Return results
    return {
        'config': config,
        'dataset_info': dataset_info,
        'training_history': history,
        'test_results': test_results,
        'model_path': model_path
    }

# Function to load a trained model
def load_model(model_path: str) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint file
        
    Returns:
        Tuple containing the model, configuration, and dataset info
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract components
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    dataset_info = checkpoint['dataset_info']
    
    # Initialize model
    model = ResNetTransformer(
        num_classes=dataset_info['num_classes'],
        config=config
    )
    
    # Load state
    model.load_state_dict(model_state)
    
    return model, config, dataset_info

# Function to make predictions on new images
def predict_image(
    model: nn.Module,
    image_path: str,
    config: Dict[str, Any],
    dataset_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        config: Configuration dictionary
        dataset_info: Dictionary containing dataset information
        
    Returns:
        Dictionary containing prediction results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = get_val_transforms(config)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Get class name and probability
    class_name = dataset_info['idx_to_class'][predicted_class]
    probability = probabilities[predicted_class].item() * 100
    
    # Get top-3 predictions
    top3_values, top3_indices = torch.topk(probabilities, 3)
    top3_predictions = [
        {
            'class': dataset_info['idx_to_class'][idx.item()],
            'probability': prob.item() * 100
        }
        for idx, prob in zip(top3_indices, top3_values)
    ]
    
    # Return results
    return {
        'predicted_class': class_name,
        'probability': probability,
        'top3_predictions': top3_predictions
    }

# Example usage
if __name__ == "__main__":
    # Run the experiment
    results = run_experiment()
    
    # Print summary
    print("\nExperiment completed successfully!")
    print(f"Final test accuracy: {results['test_results']['accuracy']:.2f}%")
    print(f"Model saved to: {results['model_path']}")
