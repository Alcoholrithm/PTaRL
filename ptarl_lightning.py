import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import OrderedDict
import numpy as np

from tqdm import tqdm

from sklearn.cluster import KMeans

from typing import Dict, Any, Type 
from utils.misc import AccuracyScorer, MSEScorer, initialize_weights
from utils.sinkhorn import SinkhornDistance

class PTARLLightning(LightningModule):

    def __init__(self,
                 input_dims: int,
                 emb_dim: int,
                 out_dim: int,
                 gp_dl: torch.utils.data.DataLoader,
                 backbone_class,
                 backbone_hparams,
                 optimizer = "AdamW",
                 optim_hparams = {
                                    "lr" : 0.0001,
                                    "weight_decay" : 0.00005
                                },
                 scheduler = None,
                 scheduler_hparams = {},
                 task_loss = "CrossEntropyLoss",
                 task_loss_hparams = {},
                 is_regression:bool = False,
                 eps = 1e-1,
                 max_iter = 50,
                 metric = "cosine",
                 loss_weights: Dict[str, float] = {
                           "task" : 1.0,
                           "projection" : 1.0,
                           "diversifying" : 0.5,
                           "orthogonalization" : 2.5
                       },
                 random_seed: int = 0
    ) -> None:
        super().__init__()

        self.random_seed = random_seed
        pl.seed_everything(random_seed)
        
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        
        self.gp_dl = gp_dl

        
        self.backbone_class = backbone_class
        self.backbone_hparams = backbone_hparams
        
        self.K = int(np.ceil(np.log2(input_dims)))
        
        self._init_model()
        self.register_parameter("global_prototypes", torch.nn.Parameter(torch.zeros((self.K, self.emb_dim))))
        
        self.optim = getattr(optim, optimizer)
        self.sched = getattr(torch.optim.lr_scheduler, scheduler) if scheduler is not None else None

        self.optim_hparams = optim_hparams
        self.scheduler_hparams = scheduler_hparams
        
        self.sinkhorn = SinkhornDistance(eps = eps, max_iter = max_iter, metric=metric)
        if type(task_loss) == str:
            self.task_loss = getattr(torch.nn, task_loss)(**task_loss_hparams)
        else:
            self.task_loss = task_loss(**task_loss_hparams)

        self.is_regression = is_regression
        
        self.task_weight = loss_weights["task"]
        self.projection_weight = loss_weights["projection"]
        self.diversifying_weight = loss_weights["diversifying"]
        self.orthogonalization_weight = loss_weights["orthogonalization"]
        
        self.scorer = MSEScorer() if self.is_regression else AccuracyScorer()
        
        self.outputs = []
        self.set_first_phase()
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        self.optimizer = self.optim(self.parameters(), **self.optim_hparams)
        if self.sched is None:
            return [self.optimizer], []
        self.scheduler = self.sched(self.optimizer, **self.scheduler_hparams)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'} ]
    
    def _init_model(self) -> None:
        self.backbone = self.backbone_class(**self.backbone_hparams)
        initialize_weights(self.backbone)

        self.projector = nn.Sequential(
            *[
                nn.Sequential(
                    OrderedDict([
                        ("batch_norm", nn.BatchNorm1d(self.emb_dim)),
                        ("linear", nn.Linear(self.emb_dim,self.emb_dim)),
                        ("activation", nn.GELU()),
                        ("dropout", nn.Dropout(0.1))
                    ])
                )
                for _ in range(3)
            ],
            nn.Linear(self.emb_dim, self.K)
        )
        initialize_weights(self.projector)

        self.head = nn.Linear(self.emb_dim, self.out_dim)
        initialize_weights(self.head)
        
    def forward(self, x):
        return self.head(self.backbone(x))

    def second_phase_forward(self, x):
        
        x = self.projector(self.backbone(x)) # Construct coordinates
        x = x @ self.global_prototypes # Construct P-Space
        return self.head(x)
    
    def generate_emb(self, x):
        return self.backbone(x)
    
    def initialize_global_prototypes(self, embs):
        # Initialize Global Prototypes
        device = next(self.backbone.parameters()).device
        self.kmeans = KMeans(n_clusters = self.K, random_state=self.random_seed, n_init="auto").fit(embs.numpy())
        self.global_prototypes = torch.nn.Parameter(torch.tensor(self.kmeans.cluster_centers_, device = device), requires_grad=True)
        
    def set_first_phase(self) -> None:
        self.training_step = self.first_phase_step
        self.on_validation_start = self.on_first_phase_validation_start
        self.validation_step = self.first_phase_step
        self.on_validation_epoch_end = self.first_phase_validation_epoch_end
        
    def set_second_phase(self) -> None:
        self.setup = self.second_phase_intial_setup
        self.training_step = self.second_phase_step
        self.on_validation_start = self.on_second_phase_validation_start
        self.validation_step = self.second_phase_step
        self.on_validation_epoch_end = self.second_phase_validation_epoch_end
        self.forward = self.second_phase_forward
        
    def first_phase_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.task_loss(y_hat, y)

        self.outputs.append({
            "loss" : loss, 
            "y" : y,
            "y_hat" : y_hat
        })
        return {
            "loss" : loss, 
            "y" : y,
            "y_hat" : y_hat
        }
        
    def second_phase_step(self, batch, batch_idx: int):
        x, y = batch
        sample_representation = self.backbone(x)
        
        # Construct Coordinates
        coordinates = self.projector(sample_representation) 
        
        # Construct P-space
        p_space = coordinates @ self.global_prototypes 

        projection_loss = self.projection_weight * self.get_projection_loss(sample_representation, coordinates, p_space)
        
        y_hat = self.head(p_space)
        
        task_loss = self.task_weight * self.task_loss(y_hat, y)

        _diversifying_loss = 0
        _diversifying_loss = self.diversifying_weight * self.get_diversifying_loss(coordinates, y)
        
        orthogonalization_loss = 0
        orthogonalization_loss = self.orthogonalization_weight * self.get_orthogonalization_loss()
        
        total_loss = projection_loss + task_loss + _diversifying_loss + orthogonalization_loss
        
        self.outputs.append({
            "loss" : total_loss, 
            "y" : y,
            "y_hat" : y_hat,
            "projection_loss" : projection_loss,
            "task_loss" : task_loss,
            "diversifying_loss" : _diversifying_loss,
            "orthogonalization_loss" : orthogonalization_loss
        })

        return {
            "loss" : total_loss, 
            "y" : y,
            "y_hat" : y_hat,
            "projection_loss" : projection_loss,
            "task_loss" : task_loss,
            "diversifying_loss" : _diversifying_loss,
            "orthogonalization_loss" : orthogonalization_loss
        }
    

    def get_projection_loss(self, sample_representation, coordinates, p_space):
        
        # return torch.mean(self.sinkhorn(sample_representation, self.global_prototypes, coordinates))
        loss1 = torch.mean(self.sinkhorn(sample_representation, self.global_prototypes, coordinates))
        loss2 = torch.mean(F.l1_loss(sample_representation, p_space))
        return (loss1+loss2)
    
    def get_diversifying_loss(self, coordinates, y):
        random_indice = np.random.choice(len(y), len(y)//2)
        _diversifying_loss = self.diversifying_loss(coordinates[random_indice], y[random_indice])
        return _diversifying_loss

    def get_orthogonalization_loss(self):
        r_1 = torch.sqrt(torch.sum(self.global_prototypes**2,dim=1,keepdim=True))
        topic_metrix = torch.mm(self.global_prototypes, self.global_prototypes.T.float()) / torch.mm(r_1, r_1.T)
        topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)

        l1 = torch.sum(topic_metrix.abs())
        l2 = torch.sum(topic_metrix ** 2)

        loss_sparse = l1 / l2
        # loss_constraint = torch.abs(l1 - topic_metrix.shape[0]) / topic_metrix.shape[0]
        loss_constraint = torch.abs(l1 - topic_metrix.shape[0])

        # # better
        # r_loss = loss_sparse + 0.5*loss_constraint + 0.5*torch.sum((topic_metrix - torch.eye(model.topic.shape[0]).cuda())**2)
        # make sense
        # r_loss = loss_sparse + 0.5*loss_constraint + 0.5*torch.sum((torch.mm(model.topic.float(), model.topic.T.float()) - torch.eye(model.topic.shape[0]).cuda())**2)
        return loss_sparse + 0.5 * loss_constraint


    def diversifying_loss(self, coordinates, labels):
        distance = (coordinates.reshape(coordinates.shape[0],1,coordinates.shape[1])-coordinates).abs().sum(dim=2)

        if not self.is_regression:
            label_similarity = (labels.reshape(-1,1) == labels.reshape(-1,1).T).float()
        else:
            device = next(self.backbone.parameters()).device
            y_min = min(labels)
            y_max = max(labels)
            # using Sturges equation select bins in manuscripts
            num_bin = 1 + int(np.log2(labels.shape[0]))
            # num_bin = 5
            interval_width = (y_max - y_min) / num_bin
            y_assign = torch.max(torch.tensor(0).to(device),torch.min(((labels.reshape(-1,1)-y_min)/interval_width).long(),torch.tensor(num_bin-1).to(device)))
            label_similarity = (y_assign.reshape(-1,1) == y_assign.reshape(-1,1).T).float()

        positive_mask = label_similarity
        positive_loss = torch.sum(distance * positive_mask) / (torch.sum(distance)+1e-8)
        return positive_loss
    
    def on_first_phase_validation_start(self):
        """Log the training loss and the performance of the finetunning
        """
        if len(self.outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.outputs]).cpu().mean()
            y = torch.cat([out["y"] for out in self.outputs]).cpu().detach().numpy()
            y_hat = torch.cat([out["y_hat"] for out in self.outputs]).cpu().detach().numpy()
            
            train_score = self.scorer(y, y_hat)
            
            self.log("train_loss", train_loss, prog_bar = True)
            self.log("train_" + self.scorer.__name__, train_score, prog_bar = True)
            self.outputs = []   
            
        return super().on_validation_start()
    
    def first_phase_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the finetunning
        """
        val_loss = torch.Tensor([out["loss"] for out in self.outputs]).cpu().mean()

        y = torch.cat([out["y"] for out in self.outputs]).cpu().numpy()
        y_hat = torch.cat([out["y_hat"] for out in self.outputs]).cpu().numpy()
        val_score = self.scorer(y, y_hat)

        self.log("val_loss", val_loss, prog_bar = True)
        self.log("val_" + self.scorer.__name__, val_score, prog_bar = True)
        
        self.outputs = []      
        return super().on_validation_epoch_end()

    def second_phase_intial_setup(self, stage) -> None:
        if stage == "fit":
            device = next(self.backbone.parameters()).device
            embs = []
            with torch.no_grad():
                for batch in tqdm(self.gp_dl, desc = "Generating Global Prototypes", total = len(self.gp_dl)):
                    x, _ = batch
                    x = [x[0].to(device), x[1].to(device)]
                    embs.append(self.generate_emb(x).cpu())
            embs = torch.cat(embs)
            self.initialize_global_prototypes(embs)
            self._init_model()
        return super().setup(stage)
    
    def on_second_phase_validation_start(self):
        """Log the training loss and the performance of the second phase
        """
        if len(self.outputs) > 0:
            train_loss = torch.Tensor([out["loss"] for out in self.outputs]).cpu().mean()
            y = torch.cat([out["y"] for out in self.outputs]).cpu().detach().numpy()
            y_hat = torch.cat([out["y_hat"] for out in self.outputs]).cpu().detach().numpy()
            
            train_score = self.scorer(y, y_hat)
            
            self.log("train_loss", train_loss, prog_bar = True)
            self.log("train_" + self.scorer.__name__, train_score, prog_bar = True)
            
            p_loss = torch.Tensor([out["projection_loss"] for out in self.outputs]).cpu().mean()
            t_loss = torch.Tensor([out["task_loss"] for out in self.outputs]).cpu().mean()
            d_loss = torch.Tensor([out["diversifying_loss"] for out in self.outputs]).cpu().mean()
            o_loss = torch.Tensor([out["orthogonalization_loss"] for out in self.outputs]).cpu().mean()
            self.log("train_pl", p_loss, prog_bar=True)
            self.log("train_tl", t_loss, prog_bar=True)
            self.log("train_dl", d_loss, prog_bar=True)
            self.log("train_ol", o_loss, prog_bar=True)
            
            self.outputs = []   
            
        return super().on_validation_start()
    
    def second_phase_validation_epoch_end(self) -> None:
        """Log the validation loss and the performance of the finetunning
        """
        val_loss = torch.Tensor([out["loss"] for out in self.outputs]).cpu().mean()
        
        p_loss = torch.Tensor([out["projection_loss"] for out in self.outputs]).cpu().mean()
        t_loss = torch.Tensor([out["task_loss"] for out in self.outputs]).cpu().mean()
        d_loss = torch.Tensor([out["diversifying_loss"] for out in self.outputs]).cpu().mean()
        o_loss = torch.Tensor([out["orthogonalization_loss"] for out in self.outputs]).cpu().mean()

        y = torch.cat([out["y"] for out in self.outputs]).cpu().numpy()
        y_hat = torch.cat([out["y_hat"] for out in self.outputs]).detach().cpu().numpy()
        val_score = self.scorer(y, y_hat)

        self.log("val_loss", val_loss, prog_bar = True)
        self.log("val_" + self.scorer.__name__, val_score, prog_bar = True)
        self.log("val_pl", p_loss, prog_bar=True)
        self.log("val_tl", t_loss, prog_bar=True)
        self.log("val_dl", d_loss, prog_bar=True)
        self.log("val_ol", o_loss, prog_bar=True)
        self.outputs = []      
        return super().on_validation_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, _ = batch
        output = self(x)
        return output
    
    def on_predict_end(self) -> None:
        self.outputs = []
        return super().on_predict_end()