import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from datasets.data_utils import get_dataloader
from models.resnet import resnet18, resnet50
from utils import set_gpu, get_free_gpu, set_log_path, log
from test import test_model
from config import *

def get_correlated_features(model, dataloader, device="cuda:0"):
    class_wise_data = {}
    model.eval()
    with torch.no_grad():
        for idx, data, y, _, _, _ in tqdm(dataloader, leave=False):
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
    def __init__(self, backbone, num_classes, pretrained=True, layer_id=4):
        super(UnlearnModel, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(progress=True),strict=False)
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(progress=True),strict=False)
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim
  
        self.classifier = nn.Linear(d, num_classes)
        if layer_id == 4:
            self.fea_decoder = nn.Linear(d, 2)
    
        self.num_classes = num_classes
        self.fea_dim = d
        
    def initialize_decoder(self):
        self.fea_decoder = nn.Linear(self.fea_dim, 2).cuda()
    def decode(self, x):
        with torch.no_grad():
            fea = self.backbone(x)
        dec = self.fea_decoder(fea)
        return dec
    def unlearn(self, x):
        fea = self.backbone(x)
        dec = self.fea_decoder(fea)
        logits = self.classifier(fea)
        return logits, dec
    
    def forward(self, x):
        fea = self.backbone(x)
        logits = self.classifier(fea)
        return logits


class DecoderDataset(Dataset):
    def __init__(self, dataset, feature_indexes, indexes=None):
        self.dataset = dataset
        if indexes is None:
            all_indexes = np.arange(len(dataset.embeddings))
        else:
            all_indexes = indexes
            
        features = []
        for c in feature_indexes:
            features.append(feature_indexes[c])
        features = np.unique(np.concatenate(features))
        self.labels = np.zeros(len(all_indexes),dtype=np.int64)
        cond = dataset.embeddings[all_indexes][:,features].sum(axis=1) > 0
        self.labels[cond] = 1
        self.all_indexes = all_indexes
        

    def __len__(self):
        return len(self.all_indexes)

    def __getitem__(self, idx):
        return self.dataset[self.all_indexes[idx]][0], self.labels[idx]
    
    
class PseudoGroupDataset(Dataset):
    def __init__(self, dataset, feature_indexes):
        self.dataset = dataset
        all_indexes = np.arange(len(dataset.embeddings))
        
        features = []
        for c in feature_indexes:
            features.append(feature_indexes[c])
        features = np.unique(np.concatenate(features))
        self.groups = np.zeros(len(all_indexes),dtype=np.int64) # 0 represents the group of bias-conflicting samples
        cond = dataset.embeddings[:, features].sum(axis=1) > 0
        sel_indexes = all_indexes[cond]
        self.groups[sel_indexes] = 1 
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.groups[idx], 0, 0 # the last two are place holders

def learn_decoder(trainset, fea_indexes, umodel, args):
    """learn the feature decoder
    """
    indexes = np.arange(len(trainset.embeddings))
    np.random.shuffle(indexes)
    num_train = int(len(indexes) * 0.8)
    train_indexes = indexes[0:num_train]
    val_indexes = indexes[num_train:]
    
    umodel.initialize_decoder()
    
    # decoder_train = DecoderDataset(trainset, fea_indexes)
    
    decoder_train = DecoderDataset(trainset, fea_indexes, train_indexes)
    decoder_val = DecoderDataset(trainset, fea_indexes, val_indexes)
    decoder_train_loader = DataLoader(
                    decoder_train,
                    batch_size=args.batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    decoder_val_loader = DataLoader(
                    decoder_val,
                    batch_size=args.batch_size,
                    pin_memory=True,
                    num_workers=4,
                )

        
    umodel.eval()
    umodel.fea_decoder.train()
    loss_func = nn.CrossEntropyLoss()
    decoder_optimizer = torch.optim.SGD(
            umodel.fea_decoder.parameters(), lr=args.lr_decoder, momentum=0.9, weight_decay=1e-4
    )
    best_val_acc = 0.0
    tolerance = 5
    tolerance_count = 0
    for epoch in range(1, args.epoch_decoder+1):
        loss_avg = 0.0
        acc_avg = 0.0
        counts = 0
        for data, g in tqdm(decoder_train_loader, leave=False):
            data = data.cuda()
            g = g.cuda()
            dec = umodel.decode(data)
            loss = loss_func(dec, g)
            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()
            loss_avg += loss.item()
            counts += len(g)
            corrects = (torch.argmax(dec, dim=1) == g).sum().item()
            acc_avg += corrects
        acc_avg /= counts
        loss_avg /= counts
        
        val_acc_avg = 0.0
        counts = 0
        with torch.no_grad():
            for data, g in tqdm(decoder_val_loader, leave=False):
                data = data.cuda()
                g = g.cuda()
                dec = umodel.decode(data)
                corrects = (torch.argmax(dec, dim=1) == g).sum().item()
                val_acc_avg += corrects
                counts += len(g)
            val_acc_avg /= counts
        if best_val_acc < val_acc_avg:
            best_val_acc = val_acc_avg
            tolerance_count = 0
        else:
            tolerance_count += 1
        if epoch % 5 == 0 or epoch == args.epoch:
            log(f"[Learning decoder] Epoch {epoch} loss {loss_avg:.6f} acc_avg {acc_avg:.6f}")
        if tolerance_count == tolerance:
            log(f"[Learning decoder] Epoch {epoch} loss {loss_avg:.6f} acc_avg {acc_avg:.6f}")
            return umodel
    return umodel

def unlearn_spurious(umodel, train_loader, val_loader, pseudo_val_loader, test_loader, num_epochs, lr, alpha, save_path, tag="unlearning"):
    # unlearn the selected features
    umodel.train()
    umodel.fea_decoder.eval()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            list(umodel.backbone.parameters())+list(umodel.classifier.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    val_worst_best = 0.0
    val_pseudo_worst_best = 0.0
    val_best = 0.0
    test_acc_val_worst = ''
    test_acc_val_pseudo_worst = ''
    test_acc_val = ''
    tolerance_count = 0
    for epoch in range(1, num_epochs+1):
        loss_avg = 0.0
        acc_avg = 0.0
        counts = 0
        for data, y, _, _, _ in tqdm(train_loader, leave=False):
            data = data.cuda()
            y = y.cuda()
            ref_y = (torch.ones(len(y),2)/2).cuda()
            logits, dec = umodel.unlearn(data)
            loss1 = loss_func(logits, y)
            loss2 = loss_func(dec, ref_y)
            loss = loss1 + loss2 * alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            counts += len(y)
            corrects = (torch.argmax(logits, dim=1) == y).sum().item()
            acc_avg += corrects
        acc_avg /= counts
        loss_avg /= counts
        if epoch % 5 == 0 or epoch == num_epochs:
            log(f"[{tag}] Epoch {epoch} loss {loss_avg:.6f} acc_avg {acc_avg:.6f}")
        val_avg_acc, val_worst_acc, val_all_group_accs = test_model(umodel, val_loader)
        val_all_group_accs = ' '.join([f"G{i} {a:.6f}" for i,a in enumerate(val_all_group_accs)])
        val_str = f"[val]: avg_acc {val_avg_acc:.6f} worst_acc {val_worst_acc:.6f} {val_all_group_accs}"
        # log(val_str)
        
        pseudo_val_avg_acc, _, pseudo_val_all_group_accs = test_model(umodel, pseudo_val_loader)
        pseudo_val_worst_acc = pseudo_val_all_group_accs[0]
        pseudo_val_all_group_accs = ' '.join([f"G{i} {a:.6f}" for i,a in enumerate(pseudo_val_all_group_accs)])
        pseudo_val_str = f"[pseudo_val]: avg_acc {pseudo_val_avg_acc:.6f} bias-conflicting {pseudo_val_worst_acc:.6f} {pseudo_val_all_group_accs}"
        # log(pseudo_val_str)

        test_avg_acc, test_worst_acc, test_all_group_accs = test_model(umodel, test_loader)
        test_all_group_accs = ' '.join([f"G{i} {a:.6f}" for i,a in enumerate(test_all_group_accs)])
        test_str = f"[test]: avg_acc {test_avg_acc:.6f} worst_acc {test_worst_acc:.6f} {test_all_group_accs}"
        # log(test_str)
        if val_worst_best < val_worst_acc:
            val_worst_best = val_worst_acc
            test_acc_val_worst = test_str
        if val_pseudo_worst_best < pseudo_val_worst_acc:
            val_pseudo_worst_best = pseudo_val_worst_acc
            test_acc_val_pseudo = test_str
            tolerance_count = 0
            torch.save(umodel.state_dict(), os.path.join(save_path, "best_model.pt"))
        else:
            tolerance_count += 1
        if val_best < val_avg_acc:
            val_best = val_avg_acc
            test_acc_val = test_str
        if tolerance_count == 5:
            log(f"Early stopping at epoch {epoch}")
            break 
    log(f"val_pseudo: {val_pseudo_worst_best:.6f} "+test_acc_val_pseudo)
    log(f"val_worst {val_worst_best:.6f} "+test_acc_val_worst)
    log(f"val_avg {val_best:.6f} "+test_acc_val)
    return umodel, val_pseudo_worst_best, test_acc_val, test_acc_val_pseudo, test_acc_val_worst

def prepare_model(dataset, backbone, pretrained=True):
    if dataset == "waterbirds":
        ckpt_path = WATERBIRDS_ERM_MODEL
        umodel = UnlearnModel(backbone, 2, pretrained)
        umodel.backbone.load_state_dict(torch.load(ckpt_path), strict=False)
    if dataset == "celeba":
        ckpt_path = CELEBA_ERM_MODEL
        umodel = UnlearnModel(backbone, 2, pretrained)
        umodel.backbone.load_state_dict(torch.load(ckpt_path), strict=False)
    umodel.cuda()
    return umodel

def main(args):
    # prepare the experiment folder
    expr_name = f"{args.dataset}_{args.vlm}_Iter_{args.iter_num}_thre_{args.threshold:.2f}_B_{args.batch_size}_lr_{args.lr:.4f}_lrd_{args.lr_decoder:.4f}_alpha_{args.alpha:.2f}_epoch_{args.epoch}_epochd_{args.epoch_decoder}"
    if args.mode == "debug":
        expr_name += "_debug"
    else:
        now = datetime.now()
        timestamp = now.strftime("%m%d%Y-%H%M%S")
        expr_name += f"_{timestamp}"
    
        
    save_path = os.path.join(EXPR_PATH, expr_name)
    os.makedirs(save_path, exist_ok=True)
    set_log_path(save_path)
    
    args_str = (
            "--------Parameters--------\n"
            + "\n".join(["{}={}".format(k, args.__dict__[k]) for k in args.__dict__])
            + "\n--------------------"
        )
    log(args_str)
    
    # set gpu
    gpu = ",".join([str(i) for i in get_free_gpu()[0:1]])
    set_gpu(gpu)

    # prepare data loaders
    train_loader, idx_train_loader, val_loader, test_loader = get_dataloader(args.dataset, args.batch_size, args.vlm)
    #prepare the model
    umodel = prepare_model(args.dataset, args.backbone)
    if args.resume:
        log(f"load the pretrained model from {args.resume}")
        umodel.load_state_dict(torch.load(args.resume))
    
    val_pseudo_worst_best_iter = 0.0
    tolerance_count = 0

    for iter_idx in range(args.iter_num):
        # identify spurious features
        log(f"\n-------------- Iteration {iter_idx} --------------")
        class_correlated_feas = get_correlated_features(umodel, idx_train_loader)
        for c in class_correlated_feas:
            fig, ax = plt.subplots()
            ax.bar(np.arange(len(class_correlated_feas[c][0])), np.sort(class_correlated_feas[c][0]))
            fig.savefig(os.path.join(save_path, f"attribute_distribution_class_{c}_iter{iter_idx}.png"))
            fig.clf()
        fea_indexes = {}
        for c in class_correlated_feas:
            fea_indexes[c] = class_correlated_feas[c][1][class_correlated_feas[c][0]>args.threshold]
            
                
            # indexes = class_correlated_feas[c][1][np.argsort(class_correlated_feas[c][0])]
            # fea_indexes[c] = indexes[-args.topk:]
            
        pseudo_val_dataset = PseudoGroupDataset(val_loader.dataset, fea_indexes)
        pseudo_val_loader = DataLoader(
                    pseudo_val_dataset,
                    batch_size=args.batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
        
        umodel = learn_decoder(train_loader.dataset, fea_indexes, umodel, args)
        umodel, val_pseudo_worst_best, test_acc_val, test_acc_val_pseudo, test_acc_val_worst = unlearn_spurious(umodel, train_loader, val_loader, pseudo_val_loader, test_loader, args.epoch, args.lr, args.alpha, save_path)
        if val_pseudo_worst_best_iter < val_pseudo_worst_best:
            val_pseudo_worst_best_iter = val_pseudo_worst_best
            test_acc_val_iter, test_acc_val_pseudo_iter, test_acc_val_worst_iter = test_acc_val, test_acc_val_pseudo, test_acc_val_worst
            tolerance_count = 0
        else:
            tolerance_count += 1
        if tolerance_count == args.tolerance:
            break
            
        
    torch.save(umodel.state_dict(), os.path.join(save_path, "last_model.pt"))
    with open("unlearn_identified_features_results.txt", "a") as f:
        f.write(f"------------------\n")
        f.write(f"{expr_name}\n")
        f.write("val_worst selection: "+test_acc_val_worst_iter+"\n")
        f.write("val_pseudo_worst selection: "+test_acc_val_pseudo_iter+"\n")
        f.write("val_avg selection: "+test_acc_val_iter+"\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="unlearn spurious features")

    parser.add_argument(
        "--dataset",
        default="waterbirds",
        type=str,
        help="select dataset",
    )

    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float,
        help="select the top scoring features",
    )

    parser.add_argument(
        "--iter_num",
        default=10,
        type=int,
        help="Number of identify-unlearn iterations",
    )
    
    parser.add_argument(
        "--epoch_decoder",
        default=20,
        type=int,
        help="number of epochs to train the decoder",
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size",
    )

    parser.add_argument(
        "--epoch",
        default=10,
        type=int,
        help="number of epochs to train the main model",
    )

    parser.add_argument(
        "--alpha",
        default=0.2,
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
        default=1.e-2,
        type=float,
        help="learning rate for training the decoder",
    )
    parser.add_argument(
        "--vlm",
        default="vit-gpt2",
        type=str,
        help="type of the vision-language model used, select from [blip, vit-gpt2]",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="choose the backbone network",
    )
    parser.add_argument(
        "--mode",
        default="debug",
        type=str,
        help="training mode",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="load a saved model",
    )
    
    parser.add_argument(
        "--tolerance",
        default=3,
        type=int,
        help="stop training if the validation metric does not improve over consecutive tolerance epochs",
    )
    args = parser.parse_args()
    main(args)
    
