from torch import nn
from torch.autograd import grad

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


