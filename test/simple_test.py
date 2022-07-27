from mlip.reann import REANN, compress_symbols

_, decode, numbers = compress_symbols([64, 23, 1, 5, 5])
species = list(set(numbers))
reann = REANN(species)

import torch as tc

NTA = 280
NC = 10
cutoff = 6.

basis = tc.rand(NTA, 3).double().requires_grad_(True)

numbers = tc.randint(4, (NTA,))
lattices = (6*tc.eye(3)[None]+tc.zeros(NC, 1, 1 )).double().requires_grad_(True)
pbcs = tc.ones(NC, 3, dtype=bool)
crystalidx = tc.randint(NC, (NTA,))

reann(numbers, basis, lattices, crystalidx, pbcs)

import torch as tc
import numpy as np
from numpy import sin, cos
from torch import nn
from mlip.reann import REANN
app
r, rr, rrr = 1, 1, 1
q, α, θ, ϕ = 1, 0.2, 2., 3.1

ra = r * np.array([0, cos(q), sin(q)])
rb = r * np.array([0, -cos(q), sin(q)])
r1 = rr * np.array([cos(α), 0, sin(α)])
r2 = rrr * np.array([sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)])
r3 = rrr * np.array([-sin(θ)*cos(ϕ), -sin(θ)*sin(ϕ), cos(θ)])

positions1 = np.array([np.zeros(3), ra, r1, r2, r3])
positions2 = np.array([np.zeros(3), rb, r1, r2, r3])

pbc = tc.Tensor([False, False, False]).bool()
cell = tc.eye(3).double() * 10

positions = positions1
#coords = np.ones(100)[:, None, None] * positions
coords = positions
encode = {6:0, 1:1}
numbers = tc.tensor([encode[n] for n in [6, 1, 1, 1, 1]]).long()
crystalidx = tc.tensor([0] * 5).long()
cutoffs = [6.]* len(numbers)

desc = REANN(species=[0, 1], lmax=2, nmax=2, loop=1)
# NxAxG
#desc = Naive()
#descriptor1 = desc(tc.from_numpy(co).double())
descriptor1 = desc(numbers, tc.from_numpy(positions1).double().requires_grad_(True), 
		                   cell[None], crystalidx, pbc)
descriptor2 = desc(numbers, tc.from_numpy(positions2).double().requires_grad_(True),
		                  cell[None], crystalidx, pbc)

print(descriptor1 - descriptor2)


positions = tc.rand(NTA, 3).double().requires_grad_(True)
numbers = tc.randint(4, (NTA,))
cells = (6*tc.eye(3)[None]+tc.zeros(NC, 1, 1 )).double().requires_grad_(True)
pbcs = tc.ones(NC, 3, dtype=bool)
crystalidx = tc.randint(NC, (NTA,))
cutoff = 6.


from torch import nn
from torch.autograd import grad
device = 'cpu'

class PotentialNeuralNet(nn.Module):
    """ Behler-Parrinello type interatomic potential energy module
    
    $$ E(x) = \sum_{i}^N E_i(x) $$
    
    Parameters
    ----------
        desc: torch.nn.Module class
          Chemical descriptor sends cartesian tensor Tensor{Double} to descriptive tensor Tensor{Double}
        moduledict: torch.nn.ModuleDict
          For each species, 
        species: List of int
    
    Example
    -------
        >>> species = [0, 0, 0, 1]  # Compressed expression
        >>> desc = REANN(...)
        >>> moduledict = nn.ModuleDict()
        >>> desc = reann
        >>> for spe in species:
        ...     moduledict[str(spe)] = nn.Sequential(
        ...        nn.Linear(desc.NO, int(desc.NO*1.3)),
        ...        nn.SiLU(),
        ...        nn.Linear(int(desc.NO*1.3), 1)
        ...    )
        >>> moduledict = moduledict.double().to(device=device)
        >>> model = PotentialNeuralNet(desc, moduledict, species)
    """
    def __init__(self, desc, moduledict, species, **kwargs):
        super(PotentialNeuralNet, self).__init__()
        
#        self.desc = desc
        super(PotentialNeuralNet, self).add_module('desc', desc)
        self.moduledict = moduledict
        self.species = species       
        
    def forward(self, symbols, positions, cells, crystalidx, pbcs, cutoff=6.) -> tuple:
        """ 
    
        Parameters
        ----------
            numbers: NTA Tensor{Int} 
              periodic number of atomsㅛㅐㅕ
            positions: NTA x 3 Tensor{Double} 
              Atomic positions
            cells: NC x 3 x 3 Tensor{Double} 
              Lattice vector of `NC` number of atoms
            crystalidx: NTA Tensor{Int}
            pbcs: NC x 3 Tensor{Bool}
        
        Return
        ------
        Tuple of three tensors
        (energies, energy, forces)
        
            energies: NTA Tensor{Double}
            energy: NC Tensor{Double}
            forces: NCx3 Tensor{Double}

        """
        # Descriptor calculation
        # NTA x 3 -> NTA x D
        desc = self.desc(symbols, positions, cells, crystalidx, pbcs)
        
        
        positionsidx = tc.arange(len(positions))
        energies, new_positionsidx = [], []
        for spe in self.species:
            # NTA -> NAS
            smask = spe == symbols        
            # NAS x NO -> NAS
            energies.append(self.moduledict[str(spe)](desc[smask]))

            new_positionsidx.append(positionsidx[smask])
            
        energies = tc.cat(energies)
        new_positionsidx = tc.cat(new_positionsidx)
        _, srtidx = tc.sort(new_positionsidx)
        energies = energies[srtidx]
        

        energy, forces = [], []
        for i, cry in enumerate(tc.unique(crystalidx)):
            cmask = cry == crystalidx
            V = tc.sum(energies[cmask])
            
            energy.append(V[None])            
        energy = tc.cat(energy)
        forces = grad(tc.sum(energy), positions, create_graph=True, allow_unused=True)[0]
        
        return energies[:, 0], energy, forces



moduledict = nn.ModuleDict()
desc = reann
for spe in species:
    moduledict[str(spe)] = nn.Sequential(
        nn.Linear(desc.NO, int(desc.NO*1.3)),
        nn.SiLU(),
        nn.Linear(int(desc.NO*1.3), 1)
    )
moduledict = moduledict.double().to(device=device)
    
model = PotentialNeuralNet(desc, moduledict, species)


energes, energy, forces = model(numbers, positions, cells, crystalidx, pbcs)
energes.shape, energy.shape, forces.shape

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

    
def concate(batch, device='cpu'):
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
            energies, cat(cells).to(device=device).requires_grad_(True), 
            cat(gradients), cat(crystalidx), cat(pbcs))


from taps.ml.descriptors.torch import REANN, compress_symbols

encode, decode, numbers = compress_symbols([29])
species = list(set(numbers))
reann = REANN(species)

moduledict = nn.ModuleDict()
desc = reann
for spe in species:
    moduledict[str(spe)] = nn.Sequential(
        nn.Linear(desc.NO, int(desc.NO*1.3)),
        nn.SiLU(),
        nn.Linear(int(desc.NO*1.3), 1)
    )
moduledict = moduledict.double().to(device=device)
    
model = PotentialNeuralNet(desc, moduledict, species)

from ase.io import read
atoms_list = read('20220709/boltzmann.traj', index=':')
import numpy as np
#symbols = [a.symbols.numbers for a in atoms_list]
symbols = [[encode[n] for n in a.symbols.numbers] for a in atoms_list]
positions = [a.positions for a in atoms_list]
energies = [a.calc.results['energy'] for a in atoms_list]
cells = [a.cell.array for a in atoms_list]
gradients = [-a.calc.results['forces'] for a in atoms_list]

crystalidx = [[idx] * len(atoms_list[idx]) for idx in range(len(atoms_list))]
pbcs = [a.pbc for a in atoms_list]
           
imgdataset = BPTypeDataset(symbols, positions, energies, cells, gradients, crystalidx, pbcs)
dataloader = DataLoader(imgdataset, batch_size=100, shuffle=True, collate_fn=concate)


class MSEFLoss:
    def __call__(self, predE, predF, y, dy):
        N = len(y)
        A = len(dy)
        return tc.sum((y - predE) ** 2) / N +  tc.sum((predF - dy)**2) / A
        
    
class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, device='cpu'):
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


from torch.autograd import grad

def train(dataloader, model, loss_fn, optimizer, normalizer, device='cpu'):
    model.train()
    for batch, _ in enumerate(dataloader):

        symbols, positions, energies, cells, gradients, crystalidx, pbcs = _
        
        # Backpropagation
        optimizer.zero_grad()

        _, pred, predG = model(symbols, positions, cells, crystalidx, pbcs)
        
        lossE, lossG = loss_fn(pred, predG, normalizer.norm(tc.tensor(energies)), gradients)
        loss = lossE + lossG
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        print(loss)

    return lossE, lossG

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(log_dir='./20220709/copper_log3')

normalizer = Normalizer(tc.tensor(imgdataset.energies).double())

for t in range(5000):
    lossE, lossG = train(dataloader, model, MSEFLoss(), 
                         tc.optim.Adam(model.parameters(), lr=1e-4), normalizer)

    writer.add_scalar('Loss / MSE energy (eV)', lossE, t)
    writer.add_scalar('Loss / MSE grad (eV/A)', lossG, t)
    if t % 10 == 0:
        tc.save(model.state_dict(), '20220709/weights4_%d.pt' % t)
    print(lossE + lossG)

writer.flush()
writer.close()
