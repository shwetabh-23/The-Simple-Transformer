import argparse
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import Simple_Transformer as T


def main() -> None:

    ## Taking the paths of config yaml file path and checkpoint yaml file path

    args = argparse.ArgumentParser(description= "Transformer Model")
    args.add_argument('config_path', type = str, help = "Enter the config yaml file path", default = 'config/train.small.yaml')
    args.add_argument('--checkpoint_path', type = str, help = "Enter any previously saved checkpoint path", default = None)
    arguments = args.parse_args()

    ## Checking if any previously saved models exits

    if arguments.checkpoint_path is not None:
        chkpt = torch.load(arguments.checkpoint_path)
        start = chkpt['epoch'] + 1

        print("Resume training from epoch Number : {}".format(start))

        config_dir = os.path.dirname(arguments.checkpoint_path)
        arguments.config_path = os.path.join(config_dir, "config.yaml")

    else:
        start = 0
        chkpt = None

    config = T.load_config(arguments.config_path)

    eng, ger = T.load_vocab_pair(r'D:\Transformer Model\Data\train.en', r'D:\Transformer Model\Data\train.de')
    
    model = T.make_model(input_vocab_size = len(eng), output_vocab_size  = len(ger), **config.model)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # optimizer 
    optimizer = T.make_optimizer(model.parameters(), **config.optimizer)
    
    # lr_scheduler
    scheduler = T.make_scheduler(optimizer, **config.scheduler) if "scheduler" in config else None

    #loss functions 
    train_loss = T.make_loss_func(**config.loss).to(device)
    val_loss = T.make_loss_func(**config.val_loss).to(device)

    def train(epochs, model, loader, optimizer, scheduler ,loss_fn):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        with tqdm(loader, unit = 'batch') as iterator:
            losses = []

            for i, (source, target, labels, source_masks, target_masks) in enumerate(iterator):
                print('Batch Number : ', i)
                print('the initial source and target shapes are {} and {}'.format(source.shape, target.shape))
                logits = model(source, source_masks, target, target_masks)
                loss = loss_fn(logits, labels)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
        avg_loss = np.mean(losses)
        return avg_loss

    def validate(epochs, loader, model, loss_fn):
        model.eval()

        with tqdm(loader, unit = 'batch') as iterator:
            losses = []

            for source, target, labels, source_masks, target_masks in iterator:
                with torch.no_grad():
                    logits = model(source, target, source_masks, target_masks)

                    loss = loss_fn(logits, labels)
                    losses.append(loss)

        avg_loss = np.mean(losses)
        return avg_loss

    start_epoch = 0

    #train loop
    for epoch in range(start_epoch, config.epochs):
        train_dataset = T.create_dataset(file_path_en = r'D:\Transformer Model\Data\train.en', file_path_de = r'D:\Transformer Model\Data\train.de', split = 'train')
        train_loader = T.make_data_loader(train_dataset, eng, ger, config.batch_size, device)
        train_loss = train(config.epochs, model, train_loader, optimizer, scheduler, train_loss)

        #validation loop

        valid_dataset = T.create_dataset(file_path_en = r'D:\Transformer Model\Data\train.en', file_path_de = r'D:\Transformer Model\Data\train.de', split = 'valid')
        valid_loader = T.make_data_loader(valid_dataset, eng, ger, config.batch_size, device)
        valid_loss = validate(config.epochs, valid_loader, val_loss)

if __name__ == "__main__":
    main()

