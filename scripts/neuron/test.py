from torch import nn
from mlip.pes import PotentialNeuralNet
from mlip.reann import REANN, compress_symbols

def gen_module(species=None, nmax=2, lmax=10, loop=2, device='cpu'):

    encode, decode, numbers = compress_symbols(species)
    species = list(set(numbers))
    reann = REANN(species, nmax=nmax, lmax=lmax, loop=loop, device=device)

    moduledict = nn.ModuleDict()
    desc = reann
    for spe in species:
        moduledict[str(spe)] = nn.Sequential(
            nn.Linear(desc.NO, int(desc.NO*1.3)),
            nn.Softplus(),
            nn.Linear(int(desc.NO*1.3), 1)
        )
    moduledict = moduledict.double().to(device=device)

    return PotentialNeuralNet(desc, moduledict, species)
    
model = gen_module(species=[29], lmax = 2, nmax = 15, loop = 2, device='cuda')

# Nomenclature
# SPECG-CriP(symbols, positions, energies, cells, gradients, crystalindex, pbcs)

import torch as tc
from torch.utils.data import Dataset, DataLoader

class BPTypeDataset(Dataset):
    
    """Behler Parrinello Type datasets
    Indexing should be done in the unit of crystal, a set of atom used in one calculation. 
    
    
    Parameters
    ----------
        symbols: List
        positions: List
        energies: List
        cells: List
    
    
    """
    def __init__(self, symbols, positions, energies, cells, gradients, crystalidx, pbcs):
        self.symbols = symbols
        self.positions = positions
        self.energies = energies
        self.cells = cells
        self.gradients = gradients
        self.crystalidx = crystalidx
        self.pbcs = pbcs

    def __len__(self):
        return len(self.energies)
    
    def __getitem__(self, idx):
        return self.symbols[idx], self.positions[idx], self.energies[idx], self.cells[idx], self.gradients[idx], self.crystalidx[idx], self.pbcs[idx]

    
def concate(batch, device='cuda'):
    cat = lambda x: tc.from_numpy(np.concatenate(x))
    
    symbols, positions, energies, cells, gradients, crystalidx, pbcs = [], [], [], [], [], [], []
    for data in batch:
        symbol, position, energy, cell, gradient, crystali, pbc = data
        symbols.append(symbol)
        positions.append(position)
        energies.append(energy)
        cells.append(cell[None])
        gradients.append(gradient)
        crystalidx.append(crystali)
        pbcs.append(pbc[None])      

    return (cat(symbols), cat(positions).to(device=device).requires_grad_(True), 
            tc.tensor(energies).to(device=device), cat(cells).to(device=device).requires_grad_(True), 
            cat(gradients).to(device=device), cat(crystalidx).to(device=device), cat(pbcs))

import numpy as np
from pymatgen.core import Structure
from monty.serialization import loadfn

def get_dataloader(location, symbol, number, name, batch_size=10):

    data = loadfn(location + symbol + '/' + name)
    encode, decode, numbers = compress_symbols([number])

    #data[0]['structure'].cart_coords;
    #data[0]['structure'].lattice.matrix;
    #data[0]['outputs']['forces'];
    #data[0]['structure'].lattice.pbc;
    #data[0]['num_atoms']

    symbols = [[encode[n] for n in d['structure'].atomic_numbers] for d in data]
    positions = [d['structure'].cart_coords for d in data]
    energies = [d['outputs']['energy'] for d in data]
    cells = [d['structure'].lattice.matrix for d in data]
    gradients = [-np.array(d['outputs']['forces']) for d in data]

    crystalidx = [[idx] * data[idx]['num_atoms'] for idx in range(len(data))]
    pbcs = [np.array(d['structure'].lattice.pbc) for d in data]

    imgdataset = BPTypeDataset(symbols, positions, energies, cells, gradients, crystalidx, pbcs)
    
    return imgdataset, DataLoader(imgdataset, batch_size=batch_size, shuffle=True, collate_fn=concate)

location = "/home01/x2419a03/libCalc/mlip/data/"
symbol = 'Cu'
number = 29
imgdataset, dataloader = get_dataloader(location, symbol, number, 'test.json')

class MSEFLoss:
    def __init__(self, muE=1., muF=10.):
        self.muE = muE
        self.muF = muF
    def __call__(self, predE, predF, y, dy):
        self.lossE = tc.sum((y - predE) ** 2)
        self.lossG = tc.sum((predF - dy)**2)
        return self.muE * self.lossE + self.muF * self.lossG


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, device='cuda'):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = tc.mean(tensor).to(device=device)
        self.std = tc.std(tensor).to(device=device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

import torch as tc
from torch.autograd import grad

def test(dataloader, model, loss_fn, device='cuda'):
    
    lossE, lossG, NTA = 0., 0., 0
    for batch, _ in enumerate(dataloader):

        symbols, positions, energies, cells, gradients, crystalidx, pbcs = _

        _, pred, predG = model(symbols, positions, cells, crystalidx, pbcs)
        
        loss = loss_fn(pred, predG, energies, gradients)
        
        A = len(gradients)
        NTA += A
        lossE += loss_fn.lossE.item()
        lossG += loss_fn.lossG.item()

    return lossE / NTA, lossG / NTA
    

def train(dataloader, model, loss_fn, optimizer, normalizer, device='cuda'):
    
    lossE, lossG, NTA = 0., 0., 0
    model.train()
    for batch, _ in enumerate(dataloader):

        symbols, positions, energies, cells, gradients, crystalidx, pbcs = _
        
        # Backpropagation
        optimizer.zero_grad()

        _, pred, predG = model(symbols, positions, cells, crystalidx, pbcs)
        
        #loss = loss_fn(pred, predG, normalizer.norm(tc.tensor(energies)), gradients)
        loss = loss_fn(pred, predG, energies, gradients)
        
        loss.requires_grad_(True)
        
        loss.backward()
        optimizer.step()
        
        A = len(gradients)
        NTA += A
        lossE += loss_fn.lossE.item()
        lossG += loss_fn.lossG.item()

    return lossE / NTA, lossG / NTA

from torch.utils.tensorboard import SummaryWriter

def run(symbol, number, lmax=2, nmax=15, loop=2, batch_size=100, 
        location="/home01/x2419a03/libCalc/mlip/data/", 
        device='cpu'):
    species = [number]

    log_dir = './20220727/' + symbol + '_log'
    
    model = gen_module(species=species, lmax=lmax, nmax=nmax, loop=loop, device=device)

    imgdataset, dataloader = get_dataloader(location, symbol, number, 'training.json', batch_size=batch_size)
    test_imgdataset, test_dataloader = get_dataloader(location, symbol, number, 'test.json', batch_size=batch_size)


    writer = SummaryWriter(log_dir=log_dir)

    normalizer = Normalizer(tc.tensor(imgdataset.energies).double())

    for t in range(5000):

        lossE, lossG = train(dataloader, model, MSEFLoss(), 
                             tc.optim.Adam(model.parameters(), lr=1e-4), normalizer)
        if t % 10 == 0:
            tc.save(model.state_dict(), './20220727/' + symbol + '_weights_%d.pt' % t)

        writer.add_scalar('training RMSE-E (eV/atom)', lossE, t)
        writer.add_scalar('training RMSE-F (eV/A)', lossG, t)
        
        test_lossE, test_lossG = test(test_dataloader, model, MSEFLoss())
        writer.add_scalar('test RMSE-E (eV/atom)', test_lossE, t)
        writer.add_scalar('test RMSE-F (eV/A)', test_lossG, t)
        
        print(t, ": {0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f}".format(lossE, lossG, test_lossE, test_lossG))

    writer.flush()
    writer.close()
    tc.cuda.empty_cache()
    
device = 'cuda'
tables = {'Cu': 29, 'Ge': 32, 'Li': 3, 'Mo': 42, 'Ni':28, 'Si': 14}
# tables = {'Ge': 32}
import sys
symbol = sys.argv[1]
my_tables = {symbol: tables[symbol]}

for sym, num in my_tables.items():
    run(sym, num, device=device)


