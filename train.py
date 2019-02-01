from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--dataset_root', default=COCO_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class ScatterWrapper:
    """ Input is any number of lists. This will preserve them through a dataparallel scatter. """
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, list):
                print('Warning: ScatterWrapper got input of non-list type.')
        self.args = args
        self.batch_size = len(args[0])
    
    def make_mask(self):
        out = torch.Tensor(list(range(self.batch_size))).long()
        if args.cuda: return out.cuda()
        else: return out
    
    def get_args(self, mask):
        device = mask.device
        mask = [int(x) for x in mask]
        out_args = [[] for _ in self.args]

        for out, arg in zip(out_args, self.args):
            for idx in mask:
                x = arg[idx]
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                out.append(x)
        
        return out_args

        

def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(root=args.dataset_root,
                            image_set=cfg.dataset.train,
                            transform=SSDAugmentation(MEANS))
    
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(root=args.dataset_root,
                                image_set=cfg.dataset.valid,
                                transform=BaseTransform(MEANS))

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom(port=8091)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=3)

    if args.cuda:
        cudnn.benchmark = True
        net       = nn.DataParallel(net).cuda()
        criterion = nn.DataParallel(criterion).cuda()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    if args.visdom:
        vis_title = 'Training yolact with config %s' % cfg.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()
    avg_window = 100
    loss_m_avg, loss_l_avg, loss_c_avg = (MovingAverage(avg_window), MovingAverage(avg_window), MovingAverage(avg_window))

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    adjust_learning_rate(optimizer, args.gamma, step_index)

                # Load training data
                # Note, for training on multiple gpus this will use the custom replicate and gather I wrote up there
                images, targets, masks, num_crowds = prepare_data(datum)
                
                # Forward Pass
                out = net(images)
                
                # Compute Loss
                optimizer.zero_grad()
                
                wrapper = ScatterWrapper(targets, masks, num_crowds)
                losses = criterion(out, wrapper, wrapper.make_mask())
                
                loss_l, loss_c, loss_m = [x.mean() for x in losses] # Mean here because Dataparallel
                loss = loss_l + loss_c + loss_m
                
                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                loss_c_avg.add(loss_c.item())
                loss_l_avg.add(loss_l.item())
                loss_m_avg.add(loss_m.item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())
                    l = loss_l_avg.get_avg()
                    c = loss_c_avg.get_avg()
                    m = loss_m_avg.get_avg()
                    t = l + c + m
                    print('[%3d] %7d || B: %.3f | C: %.3f | M: %.3f | T: %.3f || ETA: %s || timer: %.3f'
                            % (epoch, iteration, l,c,m,t, eta_str, elapsed), flush=True)
                    

                if args.visdom:
                    update_vis_plot(iteration, loss_l_avg.get_avg(), loss_c_avg.get_avg(), iter_plot, epoch_plot, 'append')
                
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))
            
            # This is done per epoch
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    compute_validation_map(yolact_net, val_dataset)

            if args.visdom:
                    update_vis_plot(epoch, loss_l_avg.get_avg(), loss_c_avg.get_avg(), epoch_plot, None, 'append', epoch_size)
    except KeyboardInterrupt:
        print('Stopping early. Saving network...')
        
        # Delete previous copy of the interrupted network so we don't spam the weights folder
        SavePath.remove_interrupt(args.save_folder)
        
        yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

def prepare_data(datum):
    images, (targets, masks, num_crowds) = datum
    
    if args.cuda:
        images = Variable(images.cuda(), requires_grad=False)
        targets = [Variable(ann.cuda(), requires_grad=False) for ann in targets]
        masks = [Variable(mask.cuda(), requires_grad=False) for mask in masks]
    else:
        images = Variable(images, requires_grad=False)
        targets = [Variable(ann, requires_grad=False) for ann in targets]
        masks = [Variable(mask, requires_grad=False) for mask in masks]

    return images, targets, masks, num_crowds

def compute_validation_loss(net, data_loader, criterion):
    with torch.no_grad():
        loss_b, loss_m, loss_c = (0, 0, 0)
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks = prepare_data(datum)

            out = net(images)
            b, c, m = [x.item() for x in criterion(out, targets, masks)]
            
            loss_b += b
            loss_c += c
            loss_m += m

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        loss_b /= iterations
        loss_c /= iterations
        loss_m /= iterations
        loss_t  = loss_b + loss_c + loss_m

        return (loss_b, loss_c, loss_m, loss_t)

def compute_validation_map(yolact_net, dataset):
    with torch.no_grad():
        yolact_net.eval()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        eval_script.evaluate(yolact_net, dataset, train_mode=True)
        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()
