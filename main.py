import torch
from torch import nn
from torch import distributed
from torch import optim
from torch import utils
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from tensorboardX import SummaryWriter
from model import *
import numpy as np
import argparse
import json
import time
import os

parser = argparse.ArgumentParser(description='Occupancy Networks')
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training', action='store_true')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--evaluation', action='store_true')
parser.add_argument('--inference', action='store_true')
args = parser.parse_args()


class Dict(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


def main():

    distributed.init_process_group(backend='nccl')

    with open(args.config) as file:
        config = Dict(json.load(file))

    config.update(vars(args))
    config.update(dict(
        world_size=distributed.get_world_size(),
        global_rank=distributed.get_rank(),
        device_count=torch.cuda.device_count(),
        local_rank=distributed.get_rank() % torch.cuda.device_count()
    ))

    config.global_batch_size = config.local_batch_size * config.world_size
    config.lr = config.lr * config.global_batch_size / 256

    if config.global_rank == 0:
        print(f'config: {config}')

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    encoder = models.resnet18(pretrained=True).cuda()
    decoder = ConditionalResidualDecoder(
        1, 2, 3, 4
    ).cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam([
        dict(
            params=encoder.parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2
        ),
        dict(
            params=decoder.parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2
        )
    ])

    last_epoch = -1
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        encoder.load_state_dict(checkpoint.encoder_state_dict)
        decoder.load_state_dict(checkpoint.decoder_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        last_epoch = checkpoint.last_epoch
        global_step = checkpoint.global_step

    if config.global_rank == 0:
        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)
        summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        train_dataset = ImageDataset(
            root=config.train_root,
            meta=config.train_meta,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ]),
            memcached=False
        )

        val_dataset = ImageDataset(
            root=config.val_root,
            meta=config.val_meta,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ]),
            memcached=False
        )

        train_sampler = DistributedSampler(
            dataset=train_dataset,
            shuffle=True,
            last_epoch=last_epoch
        )

        train_batch_sampler = BatchSampler(
            sampler=train_sampler,
            batch_size=config.local_batch_size,
            drop_last=False,
            milestones=config.batch_milestones,
            gamma=config.batch_gamma,
            last_epoch=last_epoch
        )

        val_sampler = DistributedSampler(
            dataset=val_dataset,
            shuffle=False
        )

        train_data_loader = utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )

        val_data_loader = utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config.local_batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )

        def sum_gradients(model):
            for param in model.parameters():
                if param.requires_grad:
                    distributed.all_reduce(param.grad)

        training_begin = time.time()

        for epoch in range(last_epoch + 1, config.num_epochs):

            encoder.train()
            decoder.train()

            for local_step, (images, labels) in enumerate(train_data_loader):

                step_begin = time.time()

                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                loss = criterion(logits, labels) / config.world_size

                optimizer.zero_grad()
                loss.backward()
                sum_gradients(model)
                optimizer.step()

                predictions = logits.topk(1)[1].squeeze()
                accuracy = torch.mean((predictions == labels).float()) / config.world_size

                distributed.all_reduce(loss)
                distributed.all_reduce(accuracy)

                step_end = time.time()

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(training=loss),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(training=accuracy),
                        global_step=global_step
                    )
                    print(f'[training] epoch: {epoch} global_step: {global_step} local_step: {local_step} '
                          f'loss: {loss:.4f} accuracy: {accuracy:.4f} [{step_end - step_begin:.4f}s]')

                global_step += 1

            torch.save(dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                last_epoch=last_epoch,
                global_step=global_step
            ), f'{config.checkpoint_directory}/epoch_{epoch}')

            lr_scheduler.step()

            if config.validation:

                model.eval()

                # batch norm statistics are different on each device
                for buffer in model.buffers():
                    distributed.broadcast(buffer, 0)

                with torch.no_grad():

                    average_loss = 0
                    average_accurtacy = 0

                    for local_step, (images, labels) in enumerate(val_data_loader):

                        images = images.cuda()
                        labels = labels.cuda()

                        logits = model(images)
                        loss = criterion(logits, labels) / config.world_size

                        predictions = logits.topk(1)[1].squeeze()
                        accuracy = torch.mean((predictions == labels).float()) / config.world_size

                        distributed.all_reduce(loss)
                        distributed.all_reduce(accuracy)

                        average_loss += loss
                        average_accurtacy += accuracy

                    average_loss /= (local_step + 1)
                    average_accurtacy /= (local_step + 1)

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(validation=average_loss),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(validation=average_accurtacy),
                        global_step=global_step
                    )
                    print(f'[validation] epoch: {epoch} loss: {average_loss:.4f} accuracy: {average_accurtacy:.4f}')

        training_end = time.time()
        if config.global_rank == 0:
            print(f'training finished [{training_end - training_begin:.4f}s]')

    elif config.validation:

        val_dataset = ImageDataset(
            root=config.val_root,
            meta=config.val_meta,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ]),
            memcached=False
        )

        val_sampler = DistributedSampler(
            dataset=val_dataset,
            shuffle=False
        )

        val_data_loader = utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config.local_batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )

        model.eval()

        # batch norm statistics are different on each device
        for buffer in model.buffers():
            distributed.broadcast(buffer, 0)

        with torch.no_grad():

            average_loss = 0
            average_accurtacy = 0

            for local_step, (images, labels) in enumerate(val_data_loader):

                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                loss = criterion(logits, labels) / config.world_size

                predictions = logits.topk(1)[1].squeeze()
                accuracy = torch.mean((predictions == labels).float()) / config.world_size

                distributed.all_reduce(loss)
                distributed.all_reduce(accuracy)

                average_loss += loss
                average_accurtacy += accuracy

            average_loss /= (local_step + 1)
            average_accurtacy /= (local_step + 1)

            if config.global_rank == 0:
                print(f'[validation] epoch: {epoch} loss: {average_loss:.4f} accuracy: {average_accurtacy:.4f}')

    if config.global_rank == 0:
        summary_writer.close()


if __name__ == '__main__':
    main()
