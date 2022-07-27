
def test_running():
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

def test_symmetry():
    import torch as tc
    import numpy as np
    from numpy import sin, cos
    from torch import nn
    
    from mlip.reann import REANN
    
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
