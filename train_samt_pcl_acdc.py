import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, val_2d
from utils.losses import PixelContrastLoss, compute_unsupervised_loss
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label
import matplotlib
matplotlib.use('Agg')


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SAMT_PCL', help='experiment_name')
parser.add_argument('--model', type=str, default='samt_pcl', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--base_temperature', type=float, default=0.07, help='')
parser.add_argument('--max_samples', type=int, default=1024, help='')
parser.add_argument('--max_views', type=int, default=1, help='')
parser.add_argument('--devcie', default='cuda:0')
parser.add_argument('--memory', default=True)

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs  
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    base_temperature = args.base_temperature
    max_samples = args.max_samples
    max_views = args.max_views
    memory = args.memory
    temperature = args.temperature
    pixel_classes = 4
    memory_size = 100
    dim = 256
    pixel_update_freq = 10
    iterations = []



    if memory:
        segment_queue = torch.randn(pixel_classes, memory_size, dim)
        segment_queue = nn.functional.normalize(segment_queue, p=2, dim=2)
        segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
        pixel_queue = torch.zeros(pixel_classes, memory_size, dim)
        pixel_queue = nn.functional.normalize(pixel_queue, p=2, dim=2)
        pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)

    def _dequeue_and_enqueue(keys, labels):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = torch.nn.functional.interpolate(labels, (keys.shape[2], keys.shape[3]), mode='nearest')

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x > 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())
                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])

                if ptr + K >= memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % memory_size


    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)


    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum = 0.9, weight_decay=0.001)

    contrast_criterion = PixelContrastLoss(temperature=temperature,
                                            base_temperature=base_temperature,
                                            max_samples=max_samples,
                                            max_views=max_views,
                                            device='cuda:0')
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    model_loss = losses.mse_loss


    # logs
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max_epoch:{}=max_iteration // len(trainloader) + 1".format(max_epoch))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs = model(volume_batch)
            seg_soft = F.softmax(outputs[0], dim=1)
            seg_soft2 = F.softmax(outputs[1], dim=1)
            with torch.no_grad():
                dis_to_mask1 = torch.sigmoid(-1500 * outputs[3])
                dis_to_mask2 = torch.sigmoid(-1500 * outputs[4])
                loss_sdm = 0
                loss_sdm += model_loss(seg_soft, dis_to_mask1)
                loss_sdm += model_loss(seg_soft2, dis_to_mask2)

            loss_seg_dice = 0
            loss_seg_dice += dice_loss(seg_soft[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice += dice_loss(seg_soft2[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            
            seg_mean = torch.mean(torch.stack([seg_soft, seg_soft2]), dim=0)
            uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            uncertainty_mask = (uncertainty > threshold)
            mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
            certainty_pseudo = mean_preds.clone()
            certainty_pseudo[uncertainty_mask] = -1
            certainty_pseudo[:labeled_bs] = label_batch[:labeled_bs].unsqueeze(1)
            
            preds = torch.argmax(seg_soft, dim=1, keepdim=True).to(torch.float)
            queue = segment_queue
            contrast_loss = 0
            contrast_loss += contrast_criterion(outputs[2], certainty_pseudo, preds, queue=queue)
            loss_unsup = 0 
            loss_unsup += F.cross_entropy(seg_soft[labeled_bs:], certainty_pseudo[labeled_bs:].squeeze(1).long(), ignore_index=-1)
            loss_unsup += F.cross_entropy(seg_soft2[labeled_bs:], certainty_pseudo[labeled_bs:].squeeze(1).long(), ignore_index=-1)

            if memory:
                _dequeue_and_enqueue(outputs[2].detach(), certainty_pseudo.detach())


            loss_unsup = consistency_weight * loss_unsup
            contrast_loss = consistency_weight * contrast_loss
            loss_sdm = consistency_weight * loss_sdm

            
            loss = args.lamda * loss_seg_dice + loss_unsup + contrast_loss + loss_sdm


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, dice: %03f, unsup:%03f, contrast:%03f, sdm:%03f' % (
            iter_num, loss, loss_seg_dice, loss_unsup, contrast_loss, loss_sdm))
            iterations.append(iter_num)
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    snapshot_path = "./model/ACDC_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    print("info:")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
