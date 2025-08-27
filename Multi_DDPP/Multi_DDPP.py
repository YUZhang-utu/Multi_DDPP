import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import AUROC, MatthewsCorrCoef, Precision, Recall, Accuracy
from Multi_DDPP.model import dmpnn
from features.date import MoleculeDataset, collate_fn


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LitGraphModel(pl.LightningModule):
    def __init__(self, model, teacher_model, train_df, val_df, learning_rate, lambda_=):
        super().__init__()
        self.model = model
        self.teacher_model = teacher_model
        self.learning_rate = learning_rate
        self.train_df = train_df
        self.val_df = val_df
        self.lambda_ = lambda_

        # Initialize metrics
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_mcc = MatthewsCorrCoef(task='binary', num_classes=2)
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch


        with torch.no_grad():
            teacher_predictions = self.teacher_model(batched_graph, extra_features)
            teacher_predictions = torch.sigmoid(teacher_predictions).detach()


        predictions = self(batched_graph, extra_features)


        loss = F.binary_cross_entropy_with_logits(predictions.view(-1), targets.float())


        soft_loss = F.kl_div(F.log_softmax(predictions.view(-1), dim=-1),
                             F.softmax(teacher_predictions.view(-1), dim=-1),
                             reduction='batchmean')


        total_loss = loss + self.lambda_ * soft_loss

        preds = torch.sigmoid(predictions).squeeze()
        self.train_accuracy(preds, targets.float())

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features)
        predictions = predictions.view(-1)
        loss = F.binary_cross_entropy_with_logits(predictions, targets.float())

        preds = torch.sigmoid(predictions)


        self.val_accuracy(preds, targets.float())
        self.val_auc(preds, targets.float())
        self.val_mcc(preds.round(), targets.int())
        self.val_precision(preds, targets.float())
        self.val_recall(preds, targets.float())


        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mcc', self.val_mcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MoleculeDataset(self.train_df['SMILES'].tolist(),
                                                 self.train_df['Label'].values,
                                                 torch.tensor(self.train_df.iloc[:, 2:].values, dtype=torch.float32))
            self.val_dataset = MoleculeDataset(self.val_df['SMILES'].tolist(),
                                               self.val_df['Label'].values,
                                               torch.tensor(self.val_df.iloc[:, 2:].values, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=24, shuffle=True, collate_fn=collate_fn, num_workers=11)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn, num_workers=11)

def main_student():
    set_seed(1)


    small_data_path = os.environ.get('small_data_path')
    student_df = pd.read_csv(small_data_path)


    train_df, val_df = train_test_split(student_df, test_size=0.1, random_state=0, stratify=student_df['Label'])

    output_dir = os.environ.get('output_dir', 'output')



    teacher_model_path = os.path.join(os.path.dirname(small_data_path), 'teacher_model.ckpt')



    student_model = dmpnn(
        node_feat_dim=,
        edge_feat_dim=,
        edge_output_dim=,
        node_output_dim=,
        extra_dim=,
        num_rounds=,
        dropout_rate=,
        num_experts=,
        moe_hid_dim=
    )


    teacher_model = dmpnn(
        node_feat_dim=109,
        edge_feat_dim=13,
        edge_output_dim=400,
        node_output_dim=400,
        extra_dim=19,
        num_rounds=7,
        dropout_rate=0.2,
        num_experts=4,
        moe_hid_dim=400
    )


    teacher_model.load_state_dict(torch.load(teacher_model_path), strict=False)


    student_module = LitGraphModel(student_model, teacher_model, train_df, val_df, learning_rate=0.0001)


    logger = TensorBoardLogger(os.path.join(output_dir, 'student_logs'), name='logs_student')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'student_model'),
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True)

    trainer = Trainer(max_epochs=500, logger=logger, callbacks=[checkpoint_callback, early_stopping_callback],
                      accelerator='gpu', devices=1)


    trainer.fit(student_module)

if __name__ == "__main__":
    main_student()