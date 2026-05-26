
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
import torchvision.models as models
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 x 25 x 25
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # 25 -> 12
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pool 12 -> 6
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # pool 6 -> 3
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1) # Output logit
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class EfficientNetB0(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained, **kwargs)
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
class EfficientNetB1(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.efficientnet_b1(pretrained=pretrained, **kwargs)
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
class ConvNeXtTiny(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.convnext_tiny(pretrained=pretrained, **kwargs)
        if in_channels != 3:
            # ConvNeXtTiny default stem is stride 4. For ~25x25, we reduce to 2 to avoid spatial collapse.
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 96, kernel_size=4, stride=2, padding=1
            )
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
class MobileNetV3Small(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.mobilenet_v3_small(pretrained=pretrained, **kwargs)
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
class MobileNetV3Large(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=pretrained, **kwargs)
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=pretrained, **kwargs)
        if in_channels != 3:
            self.model.conv1[0] = nn.Conv2d(
                in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)
class OTrainCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, **kwargs):
        super().__init__()
        # Architecture based on O'TRAIN Section 3 + Fig. 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Adaptive pooling for shape-agnostic flatten
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class RBClassificationModule(pl.LightningModule):
    def __init__(self, model_name='simplecnn', learning_rate=1e-3, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()
        
        if model_name == 'simplecnn':
            self.model = SimpleCNN()
        elif model_name == 'resnet18':
            self.model = models.resnet18(num_classes=1)
            # Adjust for 1-channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'resnet34':
            self.model = models.resnet34(num_classes=1)
            # Adjust for 1-channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'resnet50':
            self.model = models.resnet50(num_classes=1)
            # Adjust for 1-channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'efficientnet_b0':
            self.model = EfficientNetB0(in_channels=1, num_classes=1)
        elif model_name == 'efficientnet_b1':
            self.model = EfficientNetB1(in_channels=1, num_classes=1)
        elif model_name == 'convnext_tiny':
            self.model = ConvNeXtTiny(in_channels=1, num_classes=1)
        elif model_name == 'mobilenet_v3_small':
            self.model = MobileNetV3Small(in_channels=1, num_classes=1)
        elif model_name == 'mobilenet_v3_large':
            self.model = MobileNetV3Large(in_channels=1, num_classes=1)
        elif model_name == 'shufflenet_v2':
            self.model = ShuffleNetV2(in_channels=1, num_classes=1)
        elif model_name == 'otrain':
            self.model = OTrainCNN(in_channels=1, num_classes=1)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Loss function
        # pos_weight should be a tensor for BCEWithLogitsLoss
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
             self.register_buffer('pos_weight_tensor', torch.tensor(pos_weight))
        else:
             self.pos_weight_tensor = pos_weight
             
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        
        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        
        self.val_f1 = BinaryF1Score()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.val_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_prec = BinaryPrecision()
        self.test_rec = BinaryRecall()
        self.test_auroc = BinaryAUROC()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_prec(preds, y)
        self.val_rec(preds, y)
        self.val_auroc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_prec(preds, y)
        self.test_rec(preds, y)
        self.test_auroc(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)
        self.log('test_prec', self.test_prec)
        self.log('test_rec', self.test_rec)
        self.log('test_auroc', self.test_auroc)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Optional: Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
