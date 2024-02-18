import torch, torchvision
import numpy as np
from .base import BaseTrainer
from tqdm import tqdm
import metrics
import matplotlib.pyplot as plt

RUNNING_INTERVAL = 24

class Classifier(BaseTrainer):

    def train_one_epoch(self, epoch: int):
        self.model.train()
        losses = list()
        running_loss = 0
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['train'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             position=1,
                                             leave=False)):
            inputs = batch[0].to(self.device)   # (B,C,H,W)
            labels = batch[1].to(self.device)   # (B)
            #imshow(torchvision.utils.make_grid(inputs))

            logits = self.model(inputs)                 # (B,K,1,1)
            logits = torch.flatten(logits, start_dim=1) # (B,K)
            loss = self.criterion(logits, labels)
            self.backprop(loss, self.optimizer)

            losses.append(loss.item())
            running_loss += loss.item()

            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.4f}")
                running_loss = 0

        return sum(losses)/len(losses)

    @torch.no_grad
    def validation(self, epoch: int):
        self.model.eval()
        losses = list()
        running_loss = 0
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['val'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             position=1,
                                             leave=False)):
            inputs = batch[0].to(self.device)   # (B,C,H,W)
            labels = batch[1].to(self.device)   # (B)

            logits = self.model(inputs)                 # (B,K,1,1)
            logits = torch.flatten(logits, start_dim=1) # (B,K)
            loss = self.criterion(logits, labels)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.4f}")
                running_loss = 0

        return sum(losses)/len(losses)


    @torch.no_grad
    def inference(self):
        self.model.eval()
        samples = list()
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['test'],
                                             desc='Inference (generating samples)',
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True)):
            inputs = batch[0].to(self.device)   # (B,C,H,W)
            labels = batch[1].to(self.device)   # (B,)

            logits = self.model(inputs)                 # (B,K,1,1)
            logits = torch.flatten(logits, start_dim=1) # (B,K)
            preds = logits.argmax(dim=1, keepdim=False) # (B,)

            preds = preds.detach().cpu().numpy()        # (B,)
            labels = labels.detach().cpu().numpy()      # (B,)
            samples.append((preds, labels))
        return samples


    def compute_metrics(self, samples: list):
        preds, labels = zip(*samples)   # [(B,)...], [(B,)...]
        pred = np.concatenate(preds)   # (N,)
        label = np.concatenate(labels)  # (N,)

        _metrics = dict()
        _metrics['accuracy'] = metrics.accuracy(pred, label)
        #_metrics['precision'] = metrics.precision(pred, label)
        return _metrics



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

