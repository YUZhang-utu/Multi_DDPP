import os
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import AUROC, MatthewsCorrCoef, Accuracy
from Multi_DDPP.model import dmpnn
from features.date import MoleculeDataset, collate_fn


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LitGraphModel(pl.LightningModule):
    def __init__(self, model, teacher_model, train_path, val_path, learning_rate, lambda_=0.2, temperature=5.0):
        super().__init__()
        self.model = model
        self.teacher_model = teacher_model  # add teacher model
        self.learning_rate = learning_rate
        self.train_path = train_path
        self.val_path = val_path
        self.lambda_ = lambda_  # weitht of soft label 
        self.temperature = temperature  # T
        
        # freeze teacher model
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()

        
        self.train_accuracy = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.train_mcc = MatthewsCorrCoef(task='binary', num_classes=2)

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch

        
        student_logits = self(batched_graph, extra_features).view(-1)

        # hard label loss
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, targets.float())

        # KD
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(batched_graph, extra_features).view(-1)
            
            # smooth distribution of soft label
            # p(z, T) = σ(z/T)
            student_soft = torch.sigmoid(student_logits / self.temperature)
            teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
            
            
            # L_D = T^2 * MSE(p_s^soft, p_t^soft)
            soft_loss = F.mse_loss(student_soft, teacher_soft) * (self.temperature ** 2)
            
            # total：L_total = L_hard + λ * L_D
            total_loss = hard_loss + self.lambda_ * soft_loss
            
            
            self.log('train_hard_loss', hard_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train_soft_loss', soft_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            total_loss = hard_loss

        
        preds = torch.sigmoid(student_logits)
        self.train_accuracy(preds, targets.float())
        self.train_auc(preds, targets.float())
        self.train_mcc(preds.round(), targets.int())

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mcc', self.train_mcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features)
        predictions = predictions.view(-1)
        loss = F.binary_cross_entropy_with_logits(predictions, targets.float())

        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)
            self.train_dataset = MoleculeDataset(train_df['SMILES'].tolist(),
                                                 train_df['Label'].values,
                                                 torch.tensor(train_df.iloc[:, 2:].values, dtype=torch.float32))
            self.val_dataset = MoleculeDataset(val_df['SMILES'].tolist(),
                                               val_df['Label'].values,
                                               torch.tensor(val_df.iloc[:, 2:].values, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=24, shuffle=True, collate_fn=collate_fn, num_workers=11)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn, num_workers=11)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Training for Molecular Property Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--small_data_path', type=str, required=True,
                        help='Path to the small dataset CSV file for student model training')
    parser.add_argument('--teacher_model_path', type=str, default=teacher_model.ckpt)
    parser.add_argument('--output_dir', type=str, default='output')
    
    # Model  parameters
    parser.add_argument('--node_feat_dim', type=int, default=109)
    parser.add_argument('--edge_feat_dim', type=int, default=13')
    parser.add_argument('--edge_output_dim', type=int, default=400)
    parser.add_argument('--node_output_dim', type=int, default=400)
    parser.add_argument('--extra_dim', type=int, default=19)
    parser.add_argument('--num_rounds', type=int, default=7)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--moe_hid_dim', type=int, default=400)
    
   
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=11)
    parser.add_argument('--patience', type=int, default=30)
    
    # Knowledge distillation parameters
    parser.add_argument('--lambda_kd', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--teacher_dropout', type=float, default=0.2)
    parser.add_argument('--student_dropout', type=float, default=0.2')
    
    # Cross-validation parameters
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--cv_random_state', type=int, default=0)
    
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--devices', type=int, default=1)
    
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    print(f'Loading data from: {args.small_data_path}')
    student_df = pd.read_csv(args.small_data_path)
    print(f'Loaded {len(student_df)} samples with {len(student_df["SMILES"].unique())} unique molecules')

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # Determine teacher model path
    if args.teacher_model_path is None:
        teacher_model_path = os.path.join(os.path.dirname(args.small_data_path), 'teacher_model.ckpt')
    else:
        teacher_model_path = args.teacher_model_path
    
    print(f'\nTeacher model path: {teacher_model_path}')

    if os.path.exists(teacher_model_path):
        print('Loading teacher model...')
        teacher_model = dmpnn(
            node_feat_dim=args.node_feat_dim,
            edge_feat_dim=args.edge_feat_dim,
            edge_output_dim=args.edge_output_dim,
            node_output_dim=args.node_output_dim,
            extra_dim=args.extra_dim,
            num_rounds=args.num_rounds,
            dropout_rate=args.teacher_dropout,
            num_experts=args.num_experts,
            moe_hid_dim=args.moe_hid_dim
        )

        try:
            teacher_model.load_state_dict(torch.load(teacher_model_path), strict=False)
            print('✓ Teacher model loaded successfully!')
        except Exception as e:
            print(f'✗ Error loading teacher model: {e}')
            print('Training without knowledge distillation...')
            teacher_model = None
    else:
        print(f'✗ Teacher model not found at {teacher_model_path}')
        print('Training without knowledge distillation...')
        teacher_model = None

    # Cross-validation setup
    print(f'\n{"="*70}')
    print(f'Starting {args.num_folds}-Fold Cross-Validation')
    print(f'{"="*70}')
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.cv_random_state)

    for fold, (train_index, val_index) in enumerate(kf.split(student_df['SMILES'].unique()), start=1):
        print(f'\n{"="*70}')
        print(f'Training Fold {fold}/{args.num_folds}')
        print(f'{"="*70}')
        
        # Split data by unique SMILES
        train_smiles = student_df['SMILES'].unique()[train_index]
        val_smiles = student_df['SMILES'].unique()[val_index]
        train_indices = student_df[student_df['SMILES'].isin(train_smiles)].index
        val_indices = student_df[student_df['SMILES'].isin(val_smiles)].index

        train_df = student_df.iloc[train_indices]
        val_df = student_df.iloc[val_indices]
        
        print(f'Train set: {len(train_df)} samples ({len(train_smiles)} unique molecules)')
        print(f'Val set:   {len(val_df)} samples ({len(val_smiles)} unique molecules)')

        # Setup fold directory and save data splits
        fold_dir = os.path.join(args.output_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)

        # Initialize student model
        student_model = dmpnn(
            node_feat_dim=args.node_feat_dim,
            edge_feat_dim=args.edge_feat_dim,
            edge_output_dim=args.edge_output_dim,
            node_output_dim=args.node_output_dim,
            extra_dim=args.extra_dim,
            num_rounds=args.num_rounds,
            dropout_rate=args.student_dropout,
            num_experts=args.num_experts,
            moe_hid_dim=args.moe_hid_dim
        )

        student_module = LitGraphModel(
            student_model, 
            teacher_model,
            os.path.join(fold_dir, 'train.csv'),
            os.path.join(fold_dir, 'val.csv'),
            learning_rate=args.learning_rate,
            lambda_=args.lambda_kd,
            temperature=args.temperature
        )

        
        logger = TensorBoardLogger(os.path.join(fold_dir, 'student_logs'), name='logs_student')
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(fold_dir, 'student_model'),
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', 
            patience=args.patience, 
            verbose=True,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=args.max_epochs, 
            logger=logger, 
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator=args.accelerator, 
            devices=args.devices,
            deterministic=True,
            enable_progress_bar=True
        )

        print(f'\nStarting training for fold {fold}...')
        trainer.fit(student_module)
        
        print(f'\n✓ Fold {fold} completed!')
        print(f'Best model saved to: {checkpoint_callback.best_model_path}')

    print(f'\n{"="*70}')
    print(f'All {args.num_folds} folds completed successfully!')
    print(f'{"="*70}')


if __name__ == "__main__":
    main()
