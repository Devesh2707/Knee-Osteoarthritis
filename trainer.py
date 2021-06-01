
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import torchmetrics as tm

from model import CNN_Classifier
from config import CONFIG
from dataset import DataModule

cfg = CONFIG()

class CNNModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate = 1e-4, use_scheduler = True, pretrained = True):
        super().__init__()

        self.net = CNN_Classifier(model_name, pretrained = pretrained)

        self.loss_function = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.save_hyperparameters()

        self.train_metric = tm.Accuracy()
        self.valid_metric = tm.Accuracy()

    def forward(self, x):
        output = self.net(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=cfg.weight_decay)
        lr_scheduler = {
        "scheduler":torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose = True
            ),
        "name":"CosineAnnealingWarmRestarts",
        }

        # lr_scheduler = {
        # "scheduler":torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-4, total_steps=None, epochs=10, steps_per_epoch=len(self.train_dataloader()), pct_start=0.3, 
        #                                                 anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, 
        #                                                 final_div_factor=10000.0, three_phase=False, last_epoch=-1, verbose=True),
        # "name":"OneCycleLR",
        # "interval":"step"
        # }

        # lr_scheduler = {
        # "scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, 
        #                                                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08, verbose = True),
        # "name":"ReduceLROnPlateau",
        # "monitor":"Validation_loss_epoch",
        # "interval":"epoch"
        # }

        # lr_scheduler = {
        # "scheduler":torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=  [3,7], gamma=0.1, last_epoch=-1, verbose=True),
        # "name":"MultiStepLR",
        # }

        if self.use_scheduler:
              return [optimizer], [lr_scheduler]
        else:
              return optimizer

    def training_step(self, batch, batch_idx):
        image, targets = batch
        y_pred = self.forward(image)
        loss = self.loss_function(y_pred, targets.type_as(y_pred))
        train_acc_batch = self.train_metric(torch.sigmoid(y_pred), targets)
        self.log('train_acc_batch', train_acc_batch, prog_bar = True)
        self.log('train_loss_batch', loss)
        return {
            'loss': loss,
            'train_acc_batch': train_acc_batch,
        }

    def training_epoch_end(self, outputs):
        current_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_train_acc = self.train_metric.compute()
        self.log('Training_loss_epoch', current_train_loss)
        self.log('Training_ACC_epoch', current_train_acc)

    def validation_step(self, batch, batch_idx):
        image, targets = batch
        y_pred = self.forward(image)
        loss = self.loss_function(y_pred, targets.type_as(y_pred))
        val_acc_batch = self.valid_metric(torch.sigmoid(y_pred), targets)
        self.log('val_acc_batch', val_acc_batch)
        self.log('val_loss_batch', loss)
        return {
          'val_loss': loss,
          'val_acc_batch': val_acc_batch
          }

    def validation_epoch_end(self, outputs):
        current_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        current_val_acc = self.valid_metric.compute()
        print(f"Epoch {self.current_epoch}: Loss: {current_val_loss:4f}: Acc:{current_val_acc:4f}")
        self.log("Validation_loss_epoch", current_val_loss)
        self.log('Validation_ACC_epoch', current_val_acc)


def run(fold):

    print(f"Running Fold-{fold}")


    check_path = f"{cfg.model_folder}/{cfg.model_name}/{fold}"

    checkpointer = ModelCheckpoint(
    monitor = 'Validation_loss_epoch',
    dirpath = check_path,
    filename =  f"{cfg.model_name}" + "-{epoch:02d}-{Validation_loss_epoch:2f}",
    mode = 'min',
    save_top_k = 1,
    save_weights_only = True,
    verbose = True
  )

    early_stopping = EarlyStopping(
    monitor = 'Validation_loss_epoch',
    patience = 5,
    mode = 'min'
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval = 'epoch')

    callbacks = [checkpointer, 
                 early_stopping, 
                 learning_rate_monitor,
                 ]
    
    
    model = CNNModule(model_name = cfg.model_name, learning_rate=cfg.lr, pretrained = cfg.pretrained)
    
    dm = DataModule(fold = fold, train_batch_size = cfg.batch_size_train, valid_batch_size=cfg.batch_size_val, one_hot = True)
    
    trainer = pl.Trainer(
    callbacks = callbacks,
    max_epochs = cfg.max_epochs,
    progress_bar_refresh_rate = 20,
    accumulate_grad_batches=int(cfg.virtual_batch_size/cfg.batch_size_train),
    gpus = 1,
    precision = 16,
    move_metrics_to_cpu = True 
    )

    trainer.fit(model, dm)