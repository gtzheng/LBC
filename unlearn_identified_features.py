# %%
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
from deep_feature_reweighting.wb_data import (
    WaterBirdsDataset,
    get_loader,
    get_transform_cub,
)
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import subprocess
import shlex
from utils import set_gpu, get_free_gpu

parser = argparse.ArgumentParser(description="Prepare the metashift dataset")
parser.add_argument(
    "--topk",
    default=5,
    type=int,
    help="select the top scoring features",
)

parser.add_argument(
    "--epoch_decoder",
    default=50,
    type=int,
    help="number of epochs to train the decoder",
)

parser.add_argument(
    "--epoch",
    default=50,
    type=int,
    help="number of epochs to train the main model",
)

parser.add_argument(
    "--alpha",
    default=0.1,
    type=float,
    help="control the unlearn strength",
)

parser.add_argument(
    "--lr",
    default=1.e-4,
    type=float,
    help="learning rate for training the main model",
)

parser.add_argument(
    "--lr_decoder",
    default=1.e-3,
    type=float,
    help="learning rate for training the decoder",
)
    
args = parser.parse_args()



class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

def get_correlated_features(model, dataloader, device="cuda:0"):
    class_wise_data = {}
    model.eval()
    with torch.no_grad():
        for idx, data, y, _, _, _ in tqdm(dataloader):
            logits = model(data.cuda())
            logits = logits.detach().cpu()
            preds = torch.argmax(logits, dim=1).numpy()
            for i in range(len(y)):
                l = y[i].item()
                if l in class_wise_data:
                    class_wise_data[l].append((idx[i].item(),int(preds[i]==l)))
                else:
                    class_wise_data[l] = [(idx[i].item(),int(preds[i]==l))]
    embeddings = dataloader.dataset.dataset.embeddings  
    class_correlated_feas = {}
    for c in class_wise_data:
        num_per_class = len(class_wise_data[c])
        counts_pos_w = np.zeros(embeddings.shape[1])
        counts_neg_w = np.zeros(embeddings.shape[1])
        
        counts_pos_wo = np.zeros(embeddings.shape[1])
        counts_neg_wo = np.zeros(embeddings.shape[1])
        for idx, pred_res in class_wise_data[c]:
            if pred_res == 1:
                counts_pos_w[embeddings[idx] == 1] += 1
                counts_pos_wo[embeddings[idx] != 1] += 1
            else:
                counts_neg_w[embeddings[idx] == 1] += 1
                counts_neg_wo[embeddings[idx] != 1] += 1
                
            
        indexes = np.arange(embeddings.shape[1])
        active_feas = indexes[(counts_pos_w + counts_neg_w) > 0]
        total_w = counts_pos_w[active_feas] + counts_neg_w[active_feas]
        probs_w = counts_pos_w[active_feas] / total_w
        
        total_wo = counts_pos_wo[active_feas] + counts_neg_wo[active_feas]
        probs_wo = counts_pos_wo[active_feas] / (total_wo + 1e-10)
        
        scores = np.tanh(abs(np.log(probs_w / (probs_wo+1e-10)+1e-10)))
        class_correlated_feas[c] = (scores, active_feas)
    return class_correlated_feas 


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
    def forward(self, x):
        return x
class UnlearnModel(nn.Module):
    def __init__(self):
        super(UnlearnModel, self).__init__()
        model = torchvision.models.resnet50(weights=None)
        d = model.fc.in_features
        identity_layer = IdentityModel()
        self.backbone = model
        self.backbone.fc = identity_layer
        
        self.classifier = nn.Linear(d, 2)
        self.fea_decoder = nn.Linear(d, 2)
    def unlearn(self, x):
        with torch.no_grad():
            fea = self.backbone(x)
        dec = self.fea_decoder(fea)
        return dec
    def forward(self, x):
        fea = self.backbone(x)
        dec = self.fea_decoder(fea)
        logits = self.classifier(fea)
        return logits, dec


class DecoderDataset(Dataset):
    def __init__(self, dataset, feature_indexes):
        self.dataset = dataset
        all_indexes = np.arange(len(dataset.embeddings))
        sel_indexes = []
        labels = []
        features = []
        for c in feature_indexes:
            features.append(feature_indexes[c])
        features = np.unique(np.concatenate(features))
        self.labels = np.zeros(len(all_indexes),dtype=np.int64)
        cond = dataset.embeddings[:,features].sum(axis=1) > 0
        indexes = all_indexes[cond]
        self.labels[indexes] = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.labels[idx]

def test_model(model, loader):
    count = 0
    acc = 0
    model.eval()
    res = []
    groups = []
    with torch.no_grad():
        for x, y, g, p, _ in loader:
            x, y = (
                x.cuda(),
                y.cuda(),
            )  # 1 for water; 0 for land. 1 for seabird; 0 for landbird.
            out, _ = model(x)
            pred = (torch.argmax(out, dim=-1) == y).detach().cpu().numpy()
            res.append(pred)
            groups.append(g.detach().cpu().numpy())
    res = np.concatenate(res)
    groups = np.concatenate(groups)
    avg_acc = res.sum() / len(res)
    acc_group = []
    group_num = []
    for g in np.unique(groups):
        gres = res[groups == g]
        acc_group.append(gres.sum() / len(gres))
        group_num.append(len(gres))
    acc_group = np.array(acc_group)
    worst_acc = acc_group.min()
    # print(group_num)
    return avg_acc, worst_acc


gpu = ",".join([str(i) for i in get_free_gpu()[0:1]])
set_gpu(gpu)

DATA_FOLDER = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2"
CONCEPT_PATH = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/img_embeddings_thre10_vocab144.pickle"
VOCAB_PATH = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/vocab_thre10_144.pickle"
batch_size = 128
train_transform = get_transform_cub(
        target_resolution=(224, 224), train=True, augment_data=True
    )
test_transform = get_transform_cub(
    target_resolution=(224, 224), train=False, augment_data=False
)
trainset = WaterBirdsDataset(
        basedir=DATA_FOLDER,
        split="train",
        transform=train_transform,
        concept_embed=CONCEPT_PATH
    )
trainset_ref = WaterBirdsDataset(
        basedir=DATA_FOLDER,
        split="train",
        transform=test_transform,
        concept_embed=CONCEPT_PATH
    )
train_idx_dataset = IdxDataset(trainset_ref)
train_loader = DataLoader(
                trainset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=4,
            )
ref_train_loader = DataLoader(
                train_idx_dataset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=4,
            )
valset = WaterBirdsDataset(
    basedir=DATA_FOLDER,
    split="val",
    transform=test_transform,
    concept_embed=CONCEPT_PATH
)
val_loader = DataLoader(
                valset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=4,
            )

testset = WaterBirdsDataset(
    basedir=DATA_FOLDER,
    split="test",
    transform=test_transform,
    concept_embed=CONCEPT_PATH
)
test_loader = DataLoader(
                testset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=4,
            )

model = torchvision.models.resnet50(weights=None)
d = model.fc.in_features
model.fc = torch.nn.Linear(d, 2)
ckpt_path = "/bigtemp/gz5hp/spurious_correlations/dfr_ckpts/waterbirds/erm_seed1/final_checkpoint.pt"
model.load_state_dict(torch.load(ckpt_path))
model.cuda()

class_correlated_feas = get_correlated_features(model, ref_train_loader)
umodel = UnlearnModel()
umodel.backbone.load_state_dict(torch.load(ckpt_path), strict=False)
umodel.cuda()
        

# learn the feature decoder
fea_indexes = {}
for c in class_correlated_feas:
    indexes = class_correlated_feas[c][1][np.argsort(class_correlated_feas[c][0])]
    fea_indexes[c] = indexes[-args.topk:]
decoder_dataset = DecoderDataset(trainset, fea_indexes)
decoder_train_loader = DataLoader(
                decoder_dataset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=4,
            )

    
umodel.eval()
umodel.fea_decoder.train()
loss_func = nn.CrossEntropyLoss()
decoder_optimizer = torch.optim.SGD(
        umodel.fea_decoder.parameters(), lr=1.e-3, momentum=0.9, weight_decay=1e-4
)
for epoch in range(args.epoch_decoder):
    loss_avg = 0.0
    acc_avg = 0.0
    counts = 0
    for data, y in tqdm(decoder_train_loader):
        data = data.cuda()
        y = y.cuda()
        dec = umodel.unlearn(data)
        loss = loss_func(dec, y)
        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()
        loss_avg += loss.item()
        counts += len(y)
        corrects = (torch.argmax(dec, dim=1) == y).sum().item()
        acc_avg += corrects
    acc_avg /= counts
    loss_avg /= counts
    print(f"Epoch {epoch} loss {loss_avg:.4f} acc_avg {acc_avg:.4f}")
    

# %%
#unlearn the selected features
umodel.train()
umodel.fea_decoder.eval()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
        list(umodel.backbone.parameters())+list(umodel.classifier.parameters()), lr=1.e-5, momentum=0.9, weight_decay=1e-4
)
for epoch in range(args.epoch):
    loss_avg = 0.0
    acc_avg = 0.0
    counts = 0
    for data, y, _, _, _ in tqdm(train_loader):
        data = data.cuda()
        y = y.cuda()
        ref_y = (torch.ones(len(y),2)/2.0).cuda()
        logits, dec = umodel(data)
        loss1 = loss_func(logits, y)
        loss2 = loss_func(dec, ref_y)
        loss = loss1 + loss2 * args.alpha
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        counts += len(y)
        corrects = (torch.argmax(logits, dim=1) == y).sum().item()
        acc_avg += corrects
    acc_avg /= counts
    loss_avg /= counts
    print(f"Epoch {epoch} loss {loss_avg:.4f} acc_avg {acc_avg:.4f}")
umodel.fea_decoder.train()
loss_func = nn.CrossEntropyLoss()



val_avg_acc, val_worst_acc = test_model(umodel, val_loader)
print(f"[val]: avg_acc {val_avg_acc:.4f} worst_acc {val_worst_acc:.4f}")

test_avg_acc, test_worst_acc = test_model(umodel, test_loader)
print(f"[test]: avg_acc {test_avg_acc:.4f} worst_acc {test_worst_acc:.4f}")
with open("unlearn_identified_features_results.txt", "a") as f:
    f.write(f"------------------\n")
    f.write(f"topk:{args.topk}, alpha:{args.alpha:.4f}, epoch:{args.epoch}, epoch_decoder:{args.epoch_decoder}\n")
    f.write(f"[val]: avg_acc {val_avg_acc:.4f} worst_acc {val_worst_acc:.4f}\n")
    f.write(f"[test]: avg_acc {test_avg_acc:.4f} worst_acc {test_worst_acc:.4f}\n")

