import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F



class JumpDetector(nn.Module):
    def __init__(self, num_features=10, time_spacing=100):
        super(JumpDetector, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, time_spacing)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class LightningJumpDetector(pl.LightningModule):
    def __init__(self, num_features=10, time_spacing=100, lr=1e-3):
        super(LightningJumpDetector, self).__init__()
        self.model = JumpDetector(num_features, time_spacing)
        self.lr = lr

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.test_acc = []
        self.val_acc = []
        self.train_acc = []
        
        self.train_roc_auc_score = []
        self.val_roc_auc_score = []
        self.test_roc_auc_score = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.train_losses.append(loss)
        self.train_acc.append((y_hat.round() == y).float().mean())

        self.log("train_loss", loss)
        self.log("train_acc", (y_hat.round() == y).float().mean())
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.val_losses.append(loss)
        self.val_acc.append((y_hat.round() == y).float().mean())

        self.log("val_loss", loss)
        self.log("val_acc", (y_hat.round() == y).float().mean())
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.test_losses.append(loss)
        self.test_acc.append((y_hat.round() == y).float().mean())

        self.log("test_loss", loss)
        self.log("tes_acc", (y_hat.round() == y).float().mean())
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc_epoch", torch.stack(self.train_acc).mean())
        self.log("train_loss_epoch", torch.stack(self.train_losses).mean())
        
        self.train_losses = []
        self.train_acc = []

        
    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", torch.stack(self.val_acc).mean())
        self.log("val_loss_epoch", torch.stack(self.val_losses).mean())
        
        self.val_losses = []
        self.val_acc = []
        
    def on_test_epoch_end(self):
        self.log("test_acc_epoch", torch.stack(self.test_acc).mean())
        self.log("test_loss_epoch", torch.stack(self.test_losses).mean())
        
        self.test_losses = []
        self.test_acc = []
        
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
