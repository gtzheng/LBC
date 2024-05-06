import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from datasets.data_utils import get_dataloader
from models.resnet import resnet18, resnet50
from utils import set_gpu, get_free_gpu, set_log_path, log, BestMetricGroup, Timer, time_str, AverageMeter
from test import test_model, test_model_pseudo
from config import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datasets.data_utils import IdxDataset
from torch.optim.lr_scheduler import CosineAnnealingLR



def get_correlated_features(model, dataloader, pred_func=None, score_func="default"):
    class_wise_data = {}
    model.eval()
    with torch.no_grad():
        for idx, data, y, _, _, _ in tqdm(dataloader, leave=False):
            if pred_func:
                logits = model.normal_forward(data.cuda())
            else:
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
    
    eps = 1e-10
    for c in class_wise_data:
        count_pos = 0
        num_per_class = len(class_wise_data[c])
        counts_pos_w = np.zeros(embeddings.shape[1])
        counts_neg_w = np.zeros(embeddings.shape[1])
        
        counts_pos_wo = np.zeros(embeddings.shape[1])
        counts_neg_wo = np.zeros(embeddings.shape[1])
        for idx, pred_res in class_wise_data[c]:
            if pred_res == 1:
                counts_pos_w[embeddings[idx] == 1] += 1
                counts_pos_wo[embeddings[idx] != 1] += 1
                count_pos += 1
            else:
                counts_neg_w[embeddings[idx] == 1] += 1
                counts_neg_wo[embeddings[idx] != 1] += 1

        all_indexes = np.arange(embeddings.shape[1])
        active_indexes = all_indexes[(counts_pos_w + counts_neg_w) > 0]
        p_y1 = count_pos / num_per_class 
        p_y0 = 1 - p_y1

        p_y1_w0 = counts_pos_wo[active_indexes] / (counts_pos_wo[active_indexes] + counts_neg_wo[active_indexes] + eps)
        p_y0_w0 = 1 - p_y1_w0

        p_y1_w1 = counts_pos_w[active_indexes] / (counts_pos_w[active_indexes] + counts_neg_w[active_indexes] + eps)
        p_y0_w1 = 1 - p_y1_w1

        p_w1 = (counts_pos_w[active_indexes] + counts_neg_w[active_indexes]) / num_per_class
        p_w0 = 1 - p_w1

        cond = (p_y1_w1 == 0) | (p_y1_w0 == 0)
        p_y1_w1[cond] = 1.0
        p_y1_w0[cond] = 1.0 
        if score_func == "default":
            scores = np.tanh(abs(np.log(p_y1_w1 / (p_y1_w0 + eps)+eps)))
        elif score_func == "tanh-log":
            scores = np.tanh(np.log(p_y1_w1 / (p_y1_w0 + eps)+eps))
        elif score_func == "abs-log":
            scores = abs(np.log(p_y1_w1 / (p_y1_w0 + eps)+eps))
        elif score_func == "log":
            scores = np.log(p_y1_w1 / (p_y1_w0 + eps)+eps)
        elif score_func == "abs-diff":
            scores = abs(p_y1_w1 - p_y1_w0)
        elif score_func == "diff":
            scores = p_y1_w1 - p_y1_w0
        class_correlated_feas[c] = (scores, active_indexes)
    model.train()
    return class_correlated_feas



class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
    def forward(self, x):
        return x

class LBC(nn.Module):
    def __init__(self, backbone, num_classes, n_clusters=2, pretrained=True):
        super(LBC, self).__init__()
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
  
        self.classifier = nn.Linear(d, num_classes*n_clusters)
        self.num_classes = num_classes
        self.fea_dim = d
        self.fc = nn.Linear(d, num_classes)
        self.K = n_clusters
    
        
    def normal_forward(self, x):
        fea = self.backbone(x)
        logits = self.fc(fea)
        return logits
    
    def forward(self, x, pred=False):
        fea = self.backbone(x)
        logits = self.classifier(fea)
        if self.classifier.training:
            if pred:
                preds = torch.argmax(logits, dim=1)
                preds = preds // self.K
                return logits, preds
            else:
                return logits
        else:
            class_logits = torch.max(logits.reshape(-1, self.num_classes, self.K),dim=-1)[0]
            return class_logits

class SpuriousSampler:
    def __init__(self, dataset, batch_size, num_batches, group_ratios=None):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.groups = dataset.groups
        all_indexes = np.arange(len(self.groups))
        self.n_groups = dataset.n_groups
        self.group_indexes = []
        group_counts = []
        for g in range(self.n_groups):
            self.group_indexes.append(all_indexes[self.groups == g])
            group_counts.append(len(self.group_indexes[g]))
        K = len(group_counts) // dataset.n_classes
        if group_ratios is None:
            group_counts = np.array(group_counts)
            group_counts = group_counts.reshape(dataset.n_classes, K)
            num_zeros = (group_counts == 0).sum(axis=1)
            ratios = np.log(np.array([group_counts[c,group_counts[c]>0].std() for c in range(dataset.n_classes)])+1)
            
            self.group_ratios = np.zeros((dataset.n_classes, K))
            self.num_per_groups = np.zeros((dataset.n_classes, K),dtype=np.int64)
            self.num_per_groups[group_counts > 0] = 5 # ensure that non-zero group will be sampled
            for c in range(dataset.n_classes):
                self.group_ratios[c,:] = ratios[c] / (K - num_zeros[c])
            self.group_ratios[group_counts==0] = 0
            self.group_ratios = self.group_ratios / self.group_ratios.sum()
            self.num_per_groups += np.round(self.group_ratios * self.batch_size).astype(np.int64)
            self.num_per_groups = self.num_per_groups.reshape(-1)
            self.group_ratios = self.group_ratios.reshape(-1)

        else:
            self.group_ratios = group_ratios
        ratios = self.group_ratios.reshape(-1, K)
       

        
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for n in range(self.num_batches):
            batch = []
            num_per_groups = self.num_per_groups
            for g in range(self.n_groups):
                if num_per_groups[g] == 0:
                    continue
                batch.append(np.random.choice(self.group_indexes[g], num_per_groups[g], replace=True))
            batch = np.concatenate(batch)
            yield batch
    
            
class PseudoGroupDataset(Dataset):
    def __init__(self, dataset, feature_indexes, n_clusters, use_prob=False, prob=[0.7,0.7]):
        self.dataset = dataset
        all_active_indexes = []
        for c in feature_indexes:
            all_active_indexes.append(feature_indexes[c][1])
        all_active_indexes = np.unique(np.concatenate(all_active_indexes))
        ori2idx = {l:i for i,l in enumerate(all_active_indexes)}
        cls_indexes = {}
        for c in feature_indexes:
            cls_indexes[c] = np.array([ori2idx[l] for l in feature_indexes[c][1]])

        embeddings = dataset.embeddings[:,all_active_indexes]
        for c in feature_indexes:
            embeddings[np.ix_(dataset.y_array==c, cls_indexes[c])] = embeddings[np.ix_(dataset.y_array==c, cls_indexes[c])] * feature_indexes[c][0].reshape(1,-1)
        kmeans = KMeans(n_clusters=n_clusters,n_init="auto").fit(embeddings)
        
        self.groups = dataset.y_array * n_clusters + kmeans.labels_

        self.n_classes = dataset.n_classes
        self.n_groups = self.n_classes * n_clusters
        self.K = n_clusters

        self.p = prob
        self.use_prob = use_prob
            
    def __len__(self):
        return len(self.dataset)

    def group2prob(self, y, g, p):
        probs = np.zeros(self.n_groups)
        probs[y*self.K:(y+1)*self.K] = (1-p)/(self.K-1)
        probs[g] = p
        return probs

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        y = self.dataset[idx][1]
        if self.use_prob:
            p = self.p[y]
            return x, y, self.group2prob(y, self.groups[idx], p), self.groups[idx], 0 # the last two are place holders
        else:
            return x, y, self.groups[idx], self.groups[idx], 0 # the last two are place holders
            


def prepare_model(dataset, backbone, load_erm=True, pretrained=True):
    if dataset == "waterbirds":
        ckpt_path = WATERBIRDS_ERM_MODEL
        umodel = LBC(backbone, 2, args.K, pretrained)
    if dataset == "celeba":
        if backbone == "resnet50":
            ckpt_path = CELEBA_ERM_MODEL
        elif backbone == "resnet18":
            ckpt_path = CELEBA_ERM_RESNET18_MODEL
        umodel = LBC(backbone, 2, args.K, pretrained)
    if dataset == "nico":
        ckpt_path = None
        umodel = LBC(backbone, 10, args.K, pretrained)
    if dataset == "imagenet-9":
        ckpt_path = IMAGENET9_ERM_MODEL
        umodel = LBC(backbone, 9, args.K, pretrained)
    if load_erm and ckpt_path:
        state_dict = torch.load(ckpt_path)
        umodel.backbone.load_state_dict(state_dict, strict=False)
        umodel.load_state_dict(state_dict, strict=False)

    umodel.cuda()
    return umodel

def main(args):
    timer = Timer()
    # prepare the experiment folder
    expr_name = f"{args.dataset}_{args.vlm}_K-{args.K}_B-{args.batch_size}_lr-{args.lr:.4f}_epoch-{args.epoch}_nb-{args.num_batches}_load-{args.load}_layer-{args.finetune_part}_{args.score}{args.tag}"
    now = datetime.now()
    timestamp = now.strftime("%m%d%Y-%H%M%S")
    expr_name += f"_{timestamp}"
        
    if args.mode == "debug":
        save_path = os.path.join(EXPR_PATH, f"{args.dataset}_debug")
    else:
        save_path = os.path.join(EXPR_PATH, expr_name)
        
    os.makedirs(save_path, exist_ok=True)
    set_log_path(save_path)
    
    args_str = (
            "--------Parameters--------\n"
            + "\n".join(["{}={}".format(k, args.__dict__[k]) for k in args.__dict__])
            + "\n--------------------"
        )
    log(args_str)
    if args.dataset == "waterbirds":
        sel_metric = "pseudo_val_unbiased"

    elif args.dataset == "celeba":
        sel_metric = "pseudo_val_unbiased"

    elif args.dataset == "nico":
        sel_metric = "pseudo_val_unbiased"
    elif args.dataset == "imagenet-9":
        sel_metric = "val_avg"
        if args.load != "erm" and not args.pretrained:
            args.num_epochs = 120
            args.batch_size = 256
            args.init_lr = 0.05
            milestones = [50, 80, 100]
        
    # set gpu
    gpu = ",".join([str(i) for i in get_free_gpu()[0:1]])
    set_gpu(gpu)
    
    # prepare data loaders
    train_loader, idx_train_loader, val_loader, test_loader = get_dataloader(args.dataset, args.batch_size)
    #prepare the model
    load_erm = True if args.load == "erm" else False
    umodel = prepare_model(args.dataset, args.backbone, load_erm, args.pretrained)
    if args.resume:
        log(f"load the pretrained model from {args.resume}")
        umodel.load_state_dict(torch.load(args.resume))
    tolerance_count = 0

    # LBC training
    umodel.train()
    loss_func = nn.CrossEntropyLoss(reduction="none")

    metrics = {key:BestMetricGroup() for key in ["val_avg", "val_unbiased", "val_worst", "val_pseudo_worst", "pseudo_val_unbiased"]}
    tolerance_count = 0
    if args.finetune_part == "last":
        optimizer = torch.optim.SGD(
                umodel.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
    elif args.finetune_part == "all":
        optimizer = torch.optim.SGD(
                [{'params':umodel.parameters()}], lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
    if args.load != "erm" and not args.pretrained and args.dataset == "imagenet-9":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2)
    else:
        scheduler = None
   

    
    # group_ratios = None
    spurious_fea_dict = {}
    for epoch in range(1, args.epoch+1):
        if epoch == 1:
            class_correlated_feas = get_correlated_features(umodel, idx_train_loader, "normal", score_func=args.score)
        else:
            class_correlated_feas = get_correlated_features(umodel, idx_train_loader, score_func=args.score)
        fea_indexes = {}
        for c in class_correlated_feas:
            sorted_indexes = np.argsort(-class_correlated_feas[c][0])
            sel_indexes = sorted_indexes
            fea_indexes[c] = (class_correlated_feas[c][0][sel_indexes], class_correlated_feas[c][1][sel_indexes])
  

        if args.finetune_part == "last":
            umodel.eval()  
            umodel.classifier.train() 
        else:
            umodel.train()

        pseudo_val_dataset = PseudoGroupDataset(val_loader.dataset, fea_indexes, args.K)
        pseudo_val_loader = DataLoader(
                    pseudo_val_dataset,
                    batch_size=args.batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
        lbc_dataset = PseudoGroupDataset(train_loader.dataset, fea_indexes, args.K, False)
        idx_lbc_dataset = IdxDataset(lbc_dataset)
        batch_sampler = SpuriousSampler(lbc_dataset, args.batch_size, args.num_batches, group_ratios=None)
        lbc_dataloader = DataLoader(
                    idx_lbc_dataset,
                    batch_sampler=batch_sampler,
                    pin_memory=True,
                    num_workers=4,
                )
            
        loss_avg = 0.0
        acc_avg = 0.0
        counts = 0
        avg_group_accs = 0
        for idx, data, y, g_p, g, _ in tqdm(lbc_dataloader, leave=False):
            data = data.cuda()
            y = y.cuda()
            g = g.cuda()
            g_p = g_p.cuda()
            logits, preds = umodel(data, pred=True)
            losses = loss_func(logits, g_p)
            loss = losses.mean()

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            counts += 1
            acc = (preds == y).sum().item() / len(y)
            acc_avg += acc
        acc_avg /= counts
        loss_avg /= counts
       
        if scheduler:
            scheduler.step()
        val_avg_acc, val_unbiased_acc, val_worst_acc = test_model(umodel, val_loader)
        _, pseudo_val_worst, pseudo_val_unbiased = test_model_pseudo(umodel, val_loader)
        test_avg_acc, test_unbiased_acc, test_worst_acc = test_model(umodel, test_loader)
        if sel_metric == "pseudo_val_unbiased":
            sel_metric_val = pseudo_val_unbiased
        elif sel_metric == "val_pseudo_worst":
            sel_metric_val = pseudo_val_worst
        elif sel_metric == "val_avg":
            sel_metric_val = val_avg_acc
        elif sel_metric == "val_unbiased":
            sel_metric_val = val_unbiased_acc
        if metrics[sel_metric].best_val < sel_metric_val:
            torch.save(umodel.state_dict(), os.path.join(save_path, "best_model.pt"))
        
        metrics["val_avg"].add(val_avg_acc, (test_avg_acc, test_unbiased_acc, test_worst_acc, val_avg_acc))
        metrics["val_unbiased"].add(val_unbiased_acc, (test_avg_acc, test_unbiased_acc, test_worst_acc, val_avg_acc))
        metrics["val_worst"].add(val_worst_acc, (test_avg_acc, test_unbiased_acc, test_worst_acc, val_avg_acc))
        
        metrics["val_pseudo_worst"].add(pseudo_val_worst, (test_avg_acc, test_unbiased_acc, test_worst_acc, val_avg_acc))
        metrics["pseudo_val_unbiased"].add(pseudo_val_unbiased, (test_avg_acc, test_unbiased_acc, test_worst_acc, val_avg_acc))
        

        test_best_avg_acc = metrics[sel_metric].best_test[0]
        test_worst_best_group_acc = metrics[sel_metric].best_test[2]
        
        log(f"[Epoch {epoch}] loss {loss_avg:.6f} acc_avg {acc_avg:.6f} val_avg {val_avg_acc:.6f} val_unbiased {val_unbiased_acc:.6f} val_wst {val_worst_acc:.6f} val_pwst {pseudo_val_worst:.6f} val_punbiased {pseudo_val_unbiased:.6f} test_avg {test_avg_acc:.6f} test_wst {test_worst_acc:.6f} (best avg:{test_best_avg_acc:.6f} worst:{test_worst_best_group_acc:.6f})")
    
    
    val_avg_str = metrics["val_avg"].str()
    val_unbiased_str = metrics["val_unbiased"].str()
    val_worst_str = metrics["val_worst"].str()
    val_pseudo_worst_str = metrics["val_pseudo_worst"].str()
    pseudo_val_unbiased_str = metrics["pseudo_val_unbiased"].str()
    
    log(f"\n[val_avg] {val_avg_str} ")
    log(f"\n[val_unbiased] {val_unbiased_str}")
    log(f"\n[val_worst] {val_worst_str}")
    log(f"\n[val_pseudo_worst] {val_pseudo_worst_str}")
    log(f"\n[pseudo_val_unbiased] {pseudo_val_unbiased_str}")
    
     
    torch.save(umodel.state_dict(), os.path.join(save_path, "last_model.pt"))
    elapsed_time = timer.t()
    avg_per_epoch = elapsed_time / epoch
    
    with open(f"debiasing_results_{args.tag}.csv", "a") as f:
        for key in metrics:
            test_avg_acc = metrics[key].best_test[0]
            test_worst_acc = metrics[key].best_test[2]
            test_unbiased = metrics[key].best_test[1]
            val_avg_acc = metrics[key].best_test[3]
            f.write(f"{expr_name},{key},{metrics[key].best_val:.6f},{test_avg_acc:.6f},{test_worst_acc:.6f},{test_unbiased:.6f},{val_avg_acc:.6f},{time_str(elapsed_time)},{time_str(avg_per_epoch)}\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="lbc spurious features")

    parser.add_argument(
        "--dataset",
        default="waterbirds",
        type=str,
        help="select dataset",
    )

    parser.add_argument(
        "--finetune_part",
        default="all",
        type=str,
        help="last_layer or all",
    )

    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float,
        help="select the top scoring features",
    )


    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size",
    )

    parser.add_argument(
        "--num_batches",
        default=20,
        type=int,
        help="number of batches per epoch",
    )


    parser.add_argument(
        "--epoch",
        default=20,
        type=int,
        help="number of epochs to train the main model",
    )
    
    parser.add_argument(
        "--K",
        default=2,
        type=int,
        help="number of clusters",
    )

    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="select the number of spurious features",
    )

    parser.add_argument(
        "--p",
        default=1.0,
        type=float,
        help="control the lbc strength",
    )

    parser.add_argument(
        "--lr",
        default=1.e-4,
        type=float,
        help="learning rate for training the main model",
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
        "--tag",
        default="",
        type=str,
        help="additional information",
    )
    parser.add_argument(
        "--score",
        default="default",
        type=str,
        help="choose score function",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="choose score function",
    )
    


    parser.add_argument(
        "--load",
        default="erm",
        type=str,
        help="select whether to load an ERM model",
    )
    args = parser.parse_args()
    main(args)
    
