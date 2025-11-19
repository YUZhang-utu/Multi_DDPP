import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics

from Multi_DDPP.mode import dmpnn
from features.data import MoleculeDataset, collate_fn


def parse_args():

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Model parameters
    parser.add_argument('--node_feat_dim', type=int, default=109')
    parser.add_argument('--edge_feat_dim', type=int, default=13)
    parser.add_argument('--edge_output_dim', type=int, default=400)
    parser.add_argument('--node_output_dim', type=int, default=400)
    parser.add_argument('--extra_dim', type=int, default=19)
    parser.add_argument('--num_rounds', type=int, default=7)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--moe_hid_dim', type=int, default=400)
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=11)
    
    # Cross-validation 
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=1)
    
  
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    
    # Logging parameters
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_and_standardize(df, train_indices, val_indices):
    """Split data"""
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()


    scaler = StandardScaler()
    train_df.iloc[:, 2:] = scaler.fit_transform(train_df.iloc[:, 2:])
    val_df.iloc[:, 2:] = scaler.transform(val_df.iloc[:, 2:])

    return train_df, val_df


class LitGraphModel(pl.LightningModule):
    
    def __init__(self, model, train_path, val_path, args):
        super().__init__()
        self.model = model
        self.args = args
        self.train_path = train_path
        self.val_path = val_path

        # Training metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.train_mcc = torchmetrics.MatthewsCorrCoef(task='binary', num_classes=2)

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features).view(-1)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(predictions, targets.float())
        

        preds = torch.sigmoid(predictions)
        self.train_accuracy(preds, targets.float())
        self.train_auc(preds, targets.float())
        self.train_mcc(preds.round(), targets.int())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
        self.log('train_auc', self.train_auc, on_epoch=True, prog_bar=True)
        self.log('train_mcc', self.train_mcc, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features).view(-1)
        

        loss = F.binary_cross_entropy_with_logits(predictions, targets.float())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.args.lr_factor, 
            patience=self.args.lr_patience
        )
        return {
            'optimizer': optimizer, 
            'lr_scheduler': scheduler, 
            'monitor': 'val_loss'
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)
            
            self.train_dataset = MoleculeDataset(
                train_df['SMILES'].tolist(),
                train_df['Label'].values,
                torch.tensor(train_df.iloc[:, 2:].values, dtype=torch.float32)
            )
            self.val_dataset = MoleculeDataset(
                val_df['SMILES'].tolist(),
                val_df['Label'].values,
                torch.tensor(val_df.iloc[:, 2:].values, dtype=torch.float32)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=self.args.num_workers, 
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=self.args.num_workers, 
            persistent_workers=True
        )


def train_fold(fold, train_df, val_df, args):
    # Create fold directory
    fold_dir = os.path.join(args.output_dir, f'fold{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # Save split data
    train_path = os.path.join(fold_dir, 'train.csv')
    val_path = os.path.join(fold_dir, 'val.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Initialize model
    model = dmpnn(
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        edge_output_dim=args.edge_output_dim,
        node_output_dim=args.node_output_dim,
        extra_dim=args.extra_dim,
        num_rounds=args.num_rounds,
        dropout_rate=args.dropout_rate,
        num_experts=args.num_experts,
        moe_hid_dim=args.moe_hid_dim
    )

    logger = TensorBoardLogger(fold_dir, name='logs')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(fold_dir, 'checkpoints'),
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stop_patience,
        mode='min',
        verbose=True
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='gpu',
        devices=1,
        log_every_n_steps=args.log_every_n_steps
    )

    lit_model = LitGraphModel(model, train_path, val_path, args)

    # Train
    trainer.fit(lit_model)
    
    print(f"\nFold {fold} completed. Best model saved to {checkpoint_callback.best_model_path}")


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.random_seed)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    

    num_classes = df['Label'].nunique()
    if num_classes != 2:
        raise ValueError(f"Expected 2 classes for binary classification, found {num_classes}")
    
    print(f"Dataset: {len(df)} samples, {len(df['SMILES'].unique())} unique SMILES")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_seed)
    
    print(f"\nStarting {args.num_folds}-fold cross-validation")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df['SMILES'].unique()), 1):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold}/{args.num_folds}")
        print(f"{'='*60}")
        
        # Split by SMILES to avoid data leakage
        train_smiles = df['SMILES'].unique()[train_idx]
        val_smiles = df['SMILES'].unique()[val_idx]
        
        train_indices = df[df['SMILES'].isin(train_smiles)].index
        val_indices = df[df['SMILES'].isin(val_smiles)].index
        
        print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
        
        train_df, val_df = split_and_standardize(df, train_indices, val_indices)
        
        train_fold(fold, train_df, val_df, args)
    
    print(f"\n{'='*60}")
    print("All folds completed!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()