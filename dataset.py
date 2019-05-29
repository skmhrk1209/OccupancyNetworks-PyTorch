import torch
from torch import utils
from PIL import Image
import numpy as np
import glob


class OccupancyDataset(utils.data.Dataset):

    def __init__(self, root, mode, num_samples=None):

        self.root = root
        self.num_samples = num_samples

        self.directories = []
        for directory in glob.glob(f'{root}/*'):
            with open(f'{directory}/{mode}.lst') as file:
                for line in file:
                    self.directories.append(f'{directory}/{line}')

    def __len__(self):

        return len(self.directories)

    def __getitem__(self, index):

        directory = self.directories[index]
        data = np.load(f'{directory}/points.npz')
        positions = data['points'].astype(np.float32)
        occupancies = np.unpackbits(data['occupancies'])[:positions.shape[0]].astype(np.float32)
        image = Image.open(np.random.choice(glob.glob(f'{directory}/img_choy2016/*')))

        if self.num_samples:
            indices = np.random.choice(positions.shape[0], self.num_samples)
            positions = positions[indices]
            occupancies = occupancies[indices]

        positions = np.transpose(positions)
        occupancies = np.transpose(occupancies)

        return positions, occupancies, image
