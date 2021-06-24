import argparse
import json
import os
import numpy as np
import logging

import utilities
from perturbation_learning import cvae, perturbations, datasets

import torch
from torch import optim
from torchvision.utils import save_image

from dataset.cifar import DATASET_GETTERS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler  


TRAIN_MODE = 'train'
VAL_MODE = 'val'
TEST_MODE = 'test'

def optimizers(config, model): 
    name = config.training.optimizer
    if name == "adam": 
        opt = optim.Adam(model.parameters(), 
                    lr=1, weight_decay=config.training.weight_decay,)
    elif name == "sgd": 
        opt = optim.SGD(model.parameters(), 
                    lr=1, weight_decay=config.training.weight_decay,
                    momentum=config.training.momentum)
    return opt

def save_chkpt(model, optimizer, epoch, test_loss, name, dp): 
    if dp: 
        model.undataparallel() 
    torch.save({
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(), 
        "epoch": epoch,
        "test_loss": test_loss
    }, name)
    if dp: 
        model.dataparallel()

def loop(config, model, optimizer, lr_schedule, beta_schedule, logger, epoch, loader, mode=TRAIN_MODE): 
    meters = utilities.MultiAverageMeter([
        "recon", "kl", "loss"
    ])
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        epoch_idx = epoch + (batch_idx + 1) / len(loader)
        lr = lr_schedule(epoch_idx)
        optimizer.param_groups[0].update(lr=lr)
        # print(f"x_w : {len(inputs[0])}")
        # print(f"x_s : {len(inputs[1])}")
        data = inputs[0].to(config.device)
        hdata = inputs[1].to(config.device)
        # data = torch.FloatTensor(inputs[0]).to(config.device)
        # hdata = torch.FloatTensor(x_s).to(config.device)

        # print("================= img check ================")
        # save_image(inputs[0], "w_img.png")  
        # print(f"saved train img")
        # save_image(inputs[1], "s_img.png")
        # print(f"saved h_train img")
        # print("================= img check ================")


        if mode == TRAIN_MODE : 
            beta = beta_schedule(epoch)
            optimizer.zero_grad()
        else : 
            beta = beta_schedule(config.training.epochs)

        output = model(data, hdata)
        recon_loss, kl_loss = cvae.vae_loss(hdata, *output, beta=beta,
                             distribution=config.model.output_distribution)

        loss = (recon_loss + kl_loss)

        if mode == TRAIN_MODE : 
            loss.backward()
            optimizer.step() 

        meters.update({
            "recon" : recon_loss.item()/len(data), 
            "kl" : kl_loss.item()/len(data), 
            "loss" : (recon_loss.item() + kl_loss.item())/len(data)
        }, n=data.size(0))
        if mode == TRAIN_MODE and batch_idx % config.training.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(
                epoch, batch_idx, len(loader),
                100. * batch_idx / len(loader),
                str(meters)))
        if mode == TEST_MODE and batch_idx == 0 and (epoch+1) % config.eval.sample_interval == 0:
            n = min(data.size(0), 8)
            recon_hbatch = output[0]
            hcomparison = torch.cat([
                                    data[:n],
                                    hdata[:n],
                                    recon_hbatch.view(*hdata.size())[:n]])
            save_image(hcomparison.cpu(),
                     os.path.join(output_dir, 'images', f'hreconstruction_{epoch}.png'), nrow=n)

            hsample = model.sample(data)
            save_image(hsample[:min(64,config.eval.batch_size)],
                       os.path.join(output_dir, 'images', f'hsample_{epoch}.png'))

            repeat_hsample = torch.cat([model.sample(data)[:8].unsqueeze(1) for i in range(8)],dim=1)
            repeat_hsample = repeat_hsample.view(-1,*hdata.size()[1:])
            save_image(repeat_hsample[:min(64,config.eval.batch_size)],
                       os.path.join(output_dir, 'images', f'repeat_hsample_{epoch}.png'))



    logger.info('====> {} set loss: {} beta {:.4f} lr {:.8f}'.format(
          mode.capitalize().ljust(6), str(meters), beta, lr))
    return meters

    
def train(config, args, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir,'output.log')),
            logging.StreamHandler()
        ])

    model = cvae.models[config.model.type](config)
    model.to(config.device)
    if config.model.load: 
        print(f"loading {config.model.load}")
        model.load_state_dict(torch.load(config.model.load)['model_state_dict'])


    # h_train = perturbations.hs[config.perturbation.train_type](config.perturbation) # 아 그냥 그 람다함수를 리턴하는거구나. h_train은 함수다.
    # h_test = perturbations.hs[config.perturbation.test_type](config.perturbation)
    # train_loader, test_loader, val_loader = datasets.loaders[config.dataset.type](config)

    train_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')

    train_sampler = RandomSampler #if args.local_rank == -1 else DistributedSampler

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=config.training.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=config.eval.batch_size,
        num_workers=args.num_workers)

    # print("=================================")
    # print(f"h_train type: {type(h_train)}")
    # print(f"h_train : {h_train}")
    # print(f"train lodaer type : {type(train_loader)}")

    # train_iter = iter(train_loader)
    # train_data = train_iter.next()
    # save_image(train_data[0], "train_img.png")
    # print(f"saved train img")

    # print(f"train data : {train_data}")
    # h_data = h_train(train_data)
    # print(f"after h, h_train_data : {h_data}")
    # save_image(h_data[0], "h_train_img.png")
    # print(f"saved h_train img")

    # print("=================================")
    
    
    optimizer = optimizers(config, model)

    lr_schedule = lambda t: np.interp([t], *config.training.step_size_schedule)[0]
    beta_schedule = lambda t: np.interp([t], *config.training.beta_schedule)[0]
    best_val_loss = 1e7

    start_epoch = 0
    best_test_loss = 10e9
    if config.resume is not None: 
        d = torch.load(config.resume)
        logger.info(f"Resume model checkpoint {d['epoch']}...")
        optimizer.load_state_dict(d["optimizer_state_dict"])
        model.load_state_dict(d["model_state_dict"])
        start_epoch = d["epoch"] + 1

        try: 
            d = torch.load(os.path.join(output_dir, 'checkpoints', 'checkpoint_best.pth'))
            best_test_loss = d["test_loss"]
        except: 
            logger.info("No best checkpoint to resume test loss from")
    
    if config.dataparallel: 
        model.dataparallel()

    args = (config, model, optimizer, lr_schedule, beta_schedule, logger)
    for epoch in range(start_epoch, config.training.epochs): 
        # Training
        model.train()
        loop(*args, epoch, train_loader, mode=TRAIN_MODE)

        # Testing
        model.eval()
        with torch.no_grad():
            # val_meters = loop(*args, epoch, val_loader, h_train, mode=VAL_MODE)
            test_meters = loop(*args, epoch, test_loader, mode=TEST_MODE)

            test_loss = test_meters['loss']
            if config.training.checkpoint_interval != "skip": 
                if (epoch+1) % config.training.checkpoint_interval == 0: 
                    save_chkpt(model, optimizer, epoch, test_loss, 
                               os.path.join(output_dir, 'checkpoints', f'checkpoint_{epoch}.pth'), 
                               config.dataparallel)

                if test_loss < best_test_loss: 
                    save_chkpt(model, optimizer, epoch, test_loss, 
                               os.path.join(output_dir, 'checkpoints', 'checkpoint_best.pth'), 
                               config.dataparallel)
                    best_test_loss = test_loss

                save_chkpt(model, optimizer, epoch, test_loss, 
                           os.path.join(output_dir, 'checkpoints', 'checkpoint_latest.pth'), 
                           config.dataparallel)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Train script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    parser.add_argument('-dp', '--dataparallel', 
                        help='data paralllel flag', action='store_true')
    parser.add_argument('--resume', default=None, help='path to checkpoint')

    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')

    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'cifar100_for_cvae_train'],
                        help='dataset name')

    args = parser.parse_args()
    config_dict = utilities.get_config(args.config)
    config_dict['dataparallel'] = args.dataparallel

    assert os.path.splitext(os.path.basename(args.config))[0] == config_dict['model']['model_dir']

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device('cuda', args.gpu_id)

    output_dir = os.path.join(config_dict['output_dir'], 
                              config_dict['model']['model_dir'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for s in ['images', 'checkpoints']: 
        extra_dir = os.path.join(output_dir,s)
        if not os.path.exists(extra_dir):
            os.makedirs(extra_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config_dict['resume'] = args.resume

    # make the load argument optional
    if 'load' not in config_dict['model']: 
        config_dict['model']['load'] = False

    config_dict['device'] = "cuda" + ":" + str(args.gpu_id)
 
    config = utilities.config_to_namedtuple(config_dict)
    train(config, args, output_dir)
