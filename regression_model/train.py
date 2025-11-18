import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
from sklearn.model_selection import KFold
from regression_model.model import dmpnn
from features.data import MoleculeDataset, collate_fn

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_and_standardize(df, train_indices, val_indices):
    """
    data split
    """
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    features = train_df.iloc[:, 2:]
    scaler = StandardScaler().fit(features)
    train_df.iloc[:, 2:] = scaler.transform(features)
    val_features = val_df.iloc[:, 2:]
    val_df.iloc[:, 2:] = scaler.transform(val_features)

    return train_df, val_df

class LitGraphModel(pl.LightningModule):
    def __init__(self, model, train_path, val_path, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.train_path = train_path
        self.val_path = val_path
        # Metrics
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features).squeeze(-1)
        loss = F.mse_loss(predictions, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features).squeeze(-1)
        loss = F.mse_loss(predictions, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)
            self.train_dataset = MoleculeDataset(train_df['SMILES1'].tolist(),
                                                 train_df['PAMPA'].values,
                                                 torch.tensor(train_df.iloc[:, 2:].values, dtype=torch.float32))
            self.val_dataset = MoleculeDataset(val_df['SMILES1'].tolist(),
                                               val_df['PAMPA'].values,
                                               torch.tensor(val_df.iloc[:, 2:].values, dtype=torch.float32))
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(self.test_path)
            self.test_dataset = MoleculeDataset(test_df['SMILES1'].tolist(),
                                                test_df['PAMPA'].values,
                                                torch.tensor(test_df.iloc[:, 2:].values, dtype=torch.float32))
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=24, shuffle=True, collate_fn=collate_fn, num_workers=11,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn, num_workers=11,
                          persistent_workers=True)


def main():
    set_seed(1)
    data_path = os.environ.get('')
    base_directory_path = os.environ.get('')
    num_splits = 10

    df = pd.read_csv(data_path)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=0)

    fold = 0
    for train_index, val_index in kf.split(df['SMILES'].unique()):
        fold += 1
        train_smiles, val_smiles = df['SMILES'].unique()[train_index], df['SMILES'].unique()[val_index]
        train_indices = df[df['SMILES'].isin(train_smiles)].index
        val_indices = df[df['SMILES'].isin(val_smiles)].index

        train_df, val_df = split_and_standardize(df, train_indices, val_indices)

        fold_dir = os.path.join(base_directory_path, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        """
        parameters of model
        """
        model = dmpnn(node_feat_dim=109,
                      edge_feat_dim=13,
                      edge_output_dim=400,
                      node_output_dim=400,
                      extra_dim=24,
                      num_rounds=7,
                      dropout_rate=0.2,
                      num_experts=4,
                      moe_hid_dim =400,
                      num_heads=8
                      )

        logger = TensorBoardLogger(fold_dir, name='logs')
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(fold_dir, 'model'),
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=30,
            verbose=True
        )
        trainer = Trainer(
            max_epochs=500,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator='gpu',
            devices=1,
            log_every_n_steps=10
        )
        module = LitGraphModel(model, os.path.join(fold_dir, 'train.csv'), os.path.join(fold_dir, 'val.csv'),
                               learning_rate=0.0001)

        trainer.fit(module)


if __name__ == "__main__":
    main()
