def test_reann_pes():
    import torch as tc
    from torch import nn
    from mlip.pes import PotentialNeuralNet
    from mlip.reann import REANN, compress_symbols
    
    device = 'cpu'
    species = [29]
    encode, decode, numbers = compress_symbols(species)
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


    positions = tc.rand(NTA, 3).double().requires_grad_(True)
    numbers = tc.randint(4, (NTA,))
    cells = (6*tc.eye(3)[None]+tc.zeros(NC, 1, 1 )).double().requires_grad_(True)
    pbcs = tc.ones(NC, 3, dtype=bool)
    crystalidx = tc.randint(NC, (NTA,))
    cutoff = 6.

    energes, energy, forces = model(numbers, positions, cells, crystalidx, pbcs)
    energes.shape, energy.shape, forces.shape
