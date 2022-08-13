import os
import torch.distributed as dist
import argparse

class ParseWrapper:
    def __init__(self, rank=0, world_size=1, node=0, process=0, device='cpu', backend='gloo', 
                 master_addr='127.0.0.1', master_port='24500'):
        self.rank = rank
        self.world_size = world_size
        self.node = node
        self.process = process
        self.device = device
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port

def parser():
    parser = argparse.ArgumentParser()
    # This is passed in via launch.py
    parser.add_argument("--rank", type=int, default=0)
    # This needs to be explicitly passed in
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--backend", type=str, default='gloo')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--process", type=int, default=0)
        
    parser.add_argument("--master_addr", type=str, default='127.0.0.1')
    parser.add_argument("--master_port", type=str, default='24500')

    # The main entry point is called directly without using subprocess
    return parser.parse_args()
    # return ParseWrapper(rank=0, world_size=1, backend='nccl', 
    #                     node=0, process=0, device='cuda:0',
    #                     master_addr='localhost', master_port='24900')

def init_ddp(rank, world_size, backend, master_addr, master_port):
    print(rank, world_size, backend, master_addr, master_port)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return dist.new_group([i for i in range(world_size)])
    

def cleanup():
    dist.destroy_process_group()

args = parser()

rank = args.rank
world_size = args.world_size
device = args.device
backend = args.backend
master_addr = args.master_addr
master_port = args.master_port 

group = init_ddp(rank, world_size, backend, master_addr, master_port)

from torch import nn
from mlip.pes import PotentialNeuralNet
from mlip.reann import REANN, compress_symbols
from torch.nn.parallel import DistributedDataParallel as DDP


def gen_module(species=None, moduledict=None, modulelist=None, nmax=2, lmax=10, loop=2, rcut=6.0, device='cpu',
              rank=0, world_size=1):

    encode, decode, numbers = compress_symbols(species)
    species = list(set(numbers))
    
    reann = REANN(species, modulelist=modulelist, nmax=nmax, lmax=lmax, loop=loop, device=device, rcut=rcut)
    module = PotentialNeuralNet(reann, moduledict, species).to(device=device)
    
    return DDP(module, device_ids=[device])

#try:
#    model = gen_module(species=[1, 8], lmax = 2, nmax = 15, loop = 2, device=device, rank=rank, world_size=world_size)
#except:
#    assert False
#    cleanup()
    

# Nomenclature
# SPEC-PECriG(symbols, positions, energies, cells, pbcs, energyidx, crystalidx, gradients)

import torch as tc
from ase.io import read
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
    def __init__(self, symbols, positions, energies, cells, pbcs, energyidx, crystalidx, gradients, rank=0, world_size=1):
        self.symbols = symbols
        self.positions = positions
        self.energies = energies
        self.cells = cells
        self.pbcs = pbcs
        self.energyidx = energyidx
        self.crystalidx = crystalidx
        self.gradients = gradients
        
        self.rank = rank
        self.world_size = world_size
        self.length = len(self.energies) // self.world_size
        if len(self.energies) % self.world_size > self.rank:
            self.length += 1

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        idx = self.world_size * idx + self.rank
        return (self.symbols[idx], self.positions[idx], 
                self.energies[idx], self.cells[idx], self.pbcs[idx], 
                self.energyidx[idx], self.crystalidx[idx], self.gradients[idx])

    
class Concate:
    def __init__(self, device='cpu'):
        self.device = device
        
    def __call__(self, batch):
        device, cat = self.device, lambda x: tc.from_numpy(np.concatenate(x))
        
        symbols, positions, energies, cells, pbcs, energyidx, crystalidx, gradients = [], [], [], [], [], [], [], []
        for data in batch:
            symbol, position, energy, cell, pbc, energyid, crystalid, gradient = data
            symbols.append(symbol)
            positions.append(position)
            energies.append(energy)
            cells.append(cell[None])
            pbcs.append(pbc[None])      
            energyidx.append(energyid)
            crystalidx.append(crystalid)
            gradients.append(gradient)

        return (cat(symbols), cat(positions).to(device=device).requires_grad_(True), 
                tc.tensor(energies).to(device=device), cat(cells).to(device=device).requires_grad_(True), 
                cat(pbcs), tc.tensor(energyidx).to(device=device), cat(crystalidx).to(device=device), cat(gradients).to(device=device))

import numpy as np
#from pymatgen.core import Structure
#from monty.serialization import loadfn

def get_dataloader(location, numbers, batch_size=10, rank=0, world_size=1, group=None, device='cpu', index=':', split=0.95):
    concate = Concate(device=device)

    #data = loadfn(location + symbol + '/' + name)
    data = read(location, index=index)
    encode, decode, numbers = compress_symbols(numbers)
    
    if rank == 0:
        permutations = tc.randperm(len(data)).long().to(device=device)
    else:
        permutations = tc.empty(len(data)).long().to(device=device)
    
    dist.broadcast(permutations, src=0, group=group)
    print('Rank', rank, permutations)

    symbols, positions, energies, cells, pbcs, energyidx, crystalidx, gradients  = [], [], [], [], [], [], [], []
    tymbols, tositions, tnergies, tells, tbcs, tnergyidx, trystalidx, tradients  = [], [], [], [], [], [], [], []
    
    partition = int(len(data) * split)

    for idx, ridx in enumerate(permutations):
        d = data[ridx]
#        if d['outputs']['energy'] > -400:
#            continue
        if idx < partition:
            symbols.append([encode[n] for n in d.symbols.numbers])
            positions.append(d.positions)
            energies.append(d.info['TotEnergy'])
            cells.append(d.cell.array)
            pbcs.append(np.array(d.pbc))
            energyidx.append(idx)
            crystalidx.append([idx] * len(d))
            gradients.append(-d.arrays['force'])
        else:
            tymbols.append([encode[n] for n in d.symbols.numbers])
            tositions.append(d.positions)
            tnergies.append(d.info['TotEnergy'])
            tells.append(d.cell.array)
            tbcs.append(np.array(d.pbc))
            tnergyidx.append(idx)
            trystalidx.append([idx] * len(d))
            tradients.append(-d.arrays['force'])

    imgdataset = BPTypeDataset(symbols, positions, energies, cells, 
                               pbcs, energyidx, crystalidx, gradients, 
                               rank=rank, world_size=world_size)
    
    tmgdataset = BPTypeDataset(tymbols, tositions, tnergies, tells, 
                               tbcs, tnergyidx, trystalidx, tradients, 
                               rank=rank, world_size=world_size)
    
    return (imgdataset, DataLoader(imgdataset, batch_size=batch_size, shuffle=True, collate_fn=concate),
            tmgdataset, DataLoader(tmgdataset, batch_size=batch_size, shuffle=True, collate_fn=concate))

location = '/home01/x2419a03/libCalc/mlip/data/H2O/training-set/dataset_1593.xyz'
numbers = [1, 8]
#imgdataset, dataloader, tmgdataset, tataloader = get_dataloader(location, numbers, 
#                                                                rank=rank, world_size=world_size, device=device, group=group)

class MSEFLoss:
    def __init__(self, muE=1., muF=1., device='cpu'):
        self.muE = muE
        self.muF = muF
        self.device = device
    def __call__(self, predE, predF, y, dy, energyidx, crystalidx):

        NC = len(predE)
        NTA = len(predF)
        counter = self.counter(energyidx, crystalidx)
        
        self.lossE = tc.sum(((y - predE) / counter)**2)
        self.lossG = tc.sum((dy - predF)**2) 
        return self.muE * self.lossE / NC + self.muF * self.lossG / NTA
    
    def counter(self, energyidx, crystalidx):
        count = [tc.sum(cidx == crystalidx).item() for cidx in energyidx]
        return tc.tensor(count).double().requires_grad_(True).to(device=self.device)
        


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, N, device='cpu'):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = tc.mean(tensor).to(device=device) / N
        self.std = tc.std(tensor).to(device=device)

    def norm(self, tensor, N):
        #return (tensor - self.mean) / self.std
        return tensor - N * self.mean

    def denorm(self, normed_tensor, N):
        #return normed_tensor * self.std + self.mean
        return normed_tensor + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

import torch as tc
from torch.autograd import grad

def test(dataloader, model, loss_fn, normalizer, device='cpu'):
        
    lossE, lossG, NTA, NC = 0., 0., 0, 0
    model.eval()
    for batch, _ in enumerate(dataloader):

        symbols, positions, energies, cells, pbcs, energyidx, crystalidx, gradients  = _

        _, pred, predG = model(symbols, positions, cells, pbcs, energyidx, crystalidx)
        
        loss = loss_fn(pred, predG, energies, gradients, energyidx, crystalidx)
        
        NC += len(energies)
        NTA += len(gradients)
        lossE += loss_fn.lossE.item()
        lossG += loss_fn.lossG.item()
        
    return np.sqrt(lossE / NC), np.sqrt(lossG / NTA)
    

def train(dataloader, model, loss_fn, optimizer, normalizer, device='cpu'):
    
    lossE, lossG, NTA, NC = 0., 0., 0, 0
    model.train()
    for batch, _ in enumerate(dataloader):

        symbols, positions, energies, cells, pbcs, energyidx, crystalidx, gradients  = _
        
        # Backpropagation
        optimizer.zero_grad()
        _, pred, predG = model(symbols, positions, cells, pbcs, energyidx, crystalidx)
        loss = loss_fn(pred, predG, energies, gradients, energyidx, crystalidx)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()      
        
        NC += len(energies)
        NTA += len(gradients)
        lossE += loss_fn.lossE.item()
        lossG += loss_fn.lossG.item()
        
    return np.sqrt(lossE / NC), np.sqrt(lossG / NTA)

from torch.utils.tensorboard import SummaryWriter

def run(numbers, moduledict=None, modulelist=None, lmax=2, nmax=15, rcut=6., loop=2, 
        batch_size=30, 
        location='/home01/x2419a03/libCalc/mlip/data/H2O/training-set/dataset_1593.xyz', 
        device='cpu', 
       rank=0, world_size=1, group=None):
    species = numbers
    
    log_dir = './20220813/H2O_log'
    
    model = gen_module(species=species, moduledict=moduledict, modulelist=modulelist, 
                       lmax=lmax, nmax=nmax, loop=loop, rcut=rcut, device=device)
#    model = nn.DataParallel(model)
    ####
#    model.load_state_dict(tc.load('/scratch/x2419a03/workspace/20220812_2/H2O_weights_50.pt'))
#    model.desc.rcut = tc.tensor([rcut]).double().to(device=device)
    ####
    
    imgdataset, dataloader, test_imgdataset, test_dataloader = get_dataloader(
        location, numbers, batch_size=batch_size, rank=rank, world_size=world_size, device=device, group=group)
#    test_imgdataset, test_dataloader = get_dataloader(location, numbers, batch_size=batch_size, rank=rank, world_size=world_size)

    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    normalizer = Normalizer(tc.tensor(imgdataset.energies).double(), len(imgdataset.symbols))

    lr, muF, muE = 1e-2, 1., 3.
    for t in range(500):
        loss_fn = MSEFLoss(muE=muE, muF=muF, device=device)
        lossE, lossG = train(dataloader, model, loss_fn, 
                             tc.optim.Adam(model.parameters(), lr=lr), normalizer)
        if t % 10 == 0:
            tc.save(model.state_dict(), './20220813/H2O_weights_%d.pt' % t)
        loss = lossE + lossG
        
        # 30 meV
        if loss < 3e-2:
            lr = 1e-5
        # 75 meV
        elif loss < 7.5e-2:
            lr = 1e-4
        #    muF = 1.
        # 100 meV
        elif loss < 1e-1:
            lr = 1e-3
#            muF = 1.25
        # 500 meV
        elif loss < 5e-1:
            lr = 3e-3
#            muF = 1.5
        
        test_lossE, test_lossG = test(test_dataloader, model, loss_fn, normalizer)
        if rank == 0:
            writer.add_scalar('training RMSE-E (eV/atom)', lossE, t)
            writer.add_scalar('training RMSE-F (eV/A)', lossG, t)

            writer.add_scalar('test RMSE-E (eV/atom)', test_lossE, t)
            writer.add_scalar('test RMSE-F (eV/A)', test_lossG, t)
        
        print('Rank', rank, t, ": {0:<10.2f} {1:<10.2f} {2:<10.2f} {3:<10.2f}".format(
                  lossE * 1000, lossG * 1000, test_lossE * 1000, test_lossG * 1000))

    writer.close()
    # tc.cuda.empty_cache()

from mlip.reann import Gj

args = parser()

rank = args.rank
world_size = args.world_size
device = args.device
backend = args.backend
master_addr = args.master_addr
master_port = args.master_port 

desc_NO = 4
species = [0, 1]
nmax = 20
loop = 3
rcut = 4.
batch_size = 31

moduledict = nn.ModuleDict()
for spe in species:
    moduledict[str(spe)] = nn.Sequential(
        nn.Linear(desc_NO, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    
modulelist = nn.ModuleList()
for j in range(loop):
    descdict = nn.ModuleDict()
    for spe in species:
        descdict[str(spe)] = nn.Sequential(
                nn.Linear(desc_NO, 128),
                 nn.ReLU(),
                nn.Linear(128, nmax)
        )
    modulelist.append(Gj(descdict, species=species, nmax=nmax))

run([1, 8], device=device, nmax=nmax, loop=loop, rcut=rcut, batch_size=batch_size,
    moduledict=moduledict.double().to(device=device),
    modulelist=modulelist.double().to(device=device),
   rank=rank, world_size=world_size, group=group)


