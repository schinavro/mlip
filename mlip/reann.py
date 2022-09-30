import itertools
import torch as tc
from torch import nn
from torch.nn.parameter import Parameter


def compress_symbols(symbols):
    """Make each unqiue atomic symbols into consecutive numbers
     This makes coding easier by creating dense matrix.

    Example
    -------

    >>> compress_symbols([1, 1, 1, 1, 6])
    ... ({1:0, 6:1}, {0: 1, 1: 6}, [0, 1])

    Paramers
    --------
     symbols: List

    Return
    ------
     decompressor: Dict{Int, Int}
     sorted_symbols: List{Int}

    """
    species = list(set(symbols))
    encoder = dict([(spe, i) for i, spe in enumerate(species)])
    decoder = dict([(i, spe) for i, spe in enumerate(species)])
    srtd_n = [encoder[n] for n in symbols]
    return encoder, decoder, srtd_n


def get_nn(symbols, positions, cell, pbc, cutoff=6., device='cpu'):
    """From pbc, symbols, cell, positions info, returns

    Parameters
    ----------
     symbols: NTA Tensor{list of symbols
     positions: NTAx3 Tensor{Double}
     cell: 3x3 Tensor{Double}
     pbc: 3 Tensor{Bool}
         Periodic boundary condition
     cutoff:  Tensor{Double} or float

    Returns
    -------
    nidxs, iidxs, jidxs, disps, dists

    """
    # A = len(positions)
    icell = tc.linalg.pinv(cell)
    grid = tc.zeros(3).long().to(device=device)
    full = ((2*cutoff*tc.linalg.norm(icell, axis=0)).long() + 1)
    grid[pbc] = full.to(device=device)[pbc]

    iidxs, jidxs, disps, dists, isymb, jsymb = [], [], [], [], [], []
    for n1, n2, n3 in itertools.product(range(0, grid[0] + 1),
                                        range(-grid[1], grid[1] + 1),
                                        range(-grid[2], grid[2] + 1)):
        # Skip symmetric displacement

        if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):

            continue

        # Calculate the cell jumping
        jumpidx = tc.Tensor([n1, n2, n3]).double().to(device=device)
        # 3
        displacement = jumpidx @ cell

        # Brute force searchingb

        # Ax3 - 3
        jpositions = positions + displacement
        kpositions = positions - displacement
        # Ax1x3 - 1xAx3 -> AxAx3
        disp1 = jpositions[:, None] - positions[None]
        disp2 = kpositions[:, None] - positions[None]
        # AxA
        dist1 = tc.linalg.norm(disp1, axis=2)
        dist2 = tc.linalg.norm(disp2, axis=2)
        # AxA
        if n1 == 0 and n2 == 0 and n3 == 0:
            mask1 = (dist1 < cutoff) * (dist1 > 1e-8)
            mask2 = (dist2 < cutoff) * (dist2 > 1e-8)
        else:
            mask1 = dist1 < cutoff
            mask2 = dist2 < cutoff
        # Get all True idx
        iidx1, jidx1 = mask1.nonzero(as_tuple=True)
        iidx2, jidx2 = mask2.nonzero(as_tuple=True)
        # Appending
        iidxs.append(iidx1)
        jidxs.append(jidx1)
        isymb.append(symbols[iidx1])
        jsymb.append(symbols[jidx1])
        disps.append(disp1[mask1])
        dists.append(dist1[mask1])

        if n1 == 0 and n2 == 0 and n3 == 0:
            continue

        # Symmetric side appending
        iidxs.append(iidx2)
        jidxs.append(jidx2)
        isymb.append(symbols[iidx2])
        jsymb.append(symbols[jidx2])
        disps.append(disp2[mask2])
        dists.append(dist2[mask2])

    return (tc.cat(iidxs), tc.cat(jidxs), tc.cat(isymb), tc.cat(jsymb),
            tc.cat(disps), tc.cat(dists))


def get_neighbors_info(symbols, positions, cells, pbcs, energyidx, crystalidx,
                       cutoff=None, device='cpu'):
    """
    Parameters
    ----------
        symbols: NTA tensor
          Element of atoms corresponds to positions
        positions: NTAx3 tensor
          Atomic positions
        cells: NCx3x3 tensor
          Cell of each crystal
        crystalidx: NTA
          Crystal index corresponds to positions
        pbcs: NCx3 tensor
          Periodic boundary condition of each crystal

    Returns
    -------
     iidx: NN tensor{Int}
     jidx: NN tensor{Int}
     isym: NN tensor{Int}
     jsym: NN tensor{Int}
     disp: NNx3 Tensor{Double}
     disp: NN Tensor{Double}
    """

    # cryset = tc.unique(crystalidx).to(device=device)
    totalidx = tc.arange(len(symbols)).to(device=device)

    iidx, jidx, isym, jsym, cidx, disp, dist = [], [], [], [], [], [], []
    for c, cidx in enumerate(energyidx):
        cmask = crystalidx == cidx
        position = positions[cmask]
        symbol = symbols[cmask]
        crystali = totalidx[cmask]
        pbc, cell = pbcs[c], cells[c]
        # NN, NN, NNx3, NN
        idx, jdx, isy, jsy, dsp, dst = get_nn(symbol, position, cell, pbc,
                                              cutoff=cutoff, device=device)
        iidx.append(crystali[idx])
        # iidx.append(idx)
        isym.append(isy)
        jsym.append(jsy)

        jidx.append(crystali[jdx])
        # jidx.append(jdx)
        disp.append(dsp)
        dist.append(dst)

    return (tc.cat(iidx), tc.cat(jidx), tc.cat(isym), tc.cat(jsym),
            tc.cat(disp), tc.cat(dist))


def cutoff_function(dist, rcut):
    """
    Parameters
    ----------
    dist: NTA
    rcut: float

    Return
    ------
    NTA arr
    """
    # return tc.ones(dist.shape, dtype=dist.dtype, device=dist.device)
    return 0.25 * (tc.cos(dist * tc.pi/rcut)+1)**2


def gauss(dist, jatn, alpha, rs):
    """
    dist : NN
    sig : NS x nmax
    rs : NS x nmax
    return : NN x nmax
    """

    """
    NN = len(dist)
    dtype, device = dist.dtype, dist.device
    gauss = tc.empty((NN, nmax), dtype=dtype, device=device)
    for i, spe in enumerate(species):
        # nmax
        Sig, Rs = sigma[i], rs[i]
        mask = (symbols == spe)
        # NN x nmax
        tmp = np.exp(Sig * (dist[mask, None] - Rs)**2)
        # NN x 1
        gauss.masked_scatter_(mask[:, None], tmp)
    return gauss
    """
    # NSxnmax -> NNxnmax
    Rs = rs[jatn]
    alp = alpha[jatn]
    # NN x nmax - NN x 1 -> NN x nmax
    return tc.exp(alp * (dist[:, None] - Rs)**2)


def angular(disp, lmax):
    """
    disp : NN x 3
    lmax : int
    return : NN x NO
       where NN is number of total atoms
             NO is number of orbitals, or mutinomial expansion

    """
    NN, dtype, device = len(disp), disp.dtype, disp.device

    angular = []
    multinomial = tc.ones(NN, 1, dtype=dtype, device=device)
    angular.append(multinomial)

    for loop in range(lmax):
        # NxNOx1 x Nx1x3 -> NxNOx3 -> NxNO3
        multinomial = (multinomial[..., None] * disp[:, None]).reshape(NN, -1)
        angular.append(multinomial)

    # NNx1 + NNx3 + NNx9 ... -> NNxNO
    return tc.cat(angular, axis=1)


def get_density(Wln, Csn, Fxyz, iidx, jidx, device, dtype, NTA, NO, nmax):
    """
    Parameters
    ----------

    Wln: NO x nmax x norb
    Csn: NTA x nmax
    Fxyz: NNxNOxnmax

    Returns
    -------
     NTA x norb
    """
    # NTAxnmax -> NNxnmax
    cj = Csn[jidx]
    # NNx1xnmax x NNxNOxnmax -> NNxNOxnmax
    cjFxyz = cj[:, None] * Fxyz

    # NN x NO x nmax -> NTA x NO x nmax
    bnl = tc.zeros((NTA, NO, nmax), device=device, dtype=dtype).index_add(
                                            0, iidx, cjFxyz)

    # NTAxNOx(nmax)x1 x 1xNOx(nmax)x(norb) -> NTAxNOx(nmax)xnorb
    #  -> NTAxNOxnorb -> NTAxnorb
    return tc.sum(tc.sum(bnl[..., None] * Wln[None], axis=2) ** 2, axis=1)


class Gj(nn.Module):
    def __init__(self, moduledict, species=None, nmax=None):
        super(Gj, self).__init__()

        self.moduledict = moduledict
        self.species = species
        self.nmax = nmax

    def forward(self, ρ, symbols):
        """  Returns the

        Parameters
        ----------

        ρ : NTA x O
        symbols : NTA number index

        Return
        ------
        NTA x nmax
        """
        NTA, nmax = len(ρ), self.nmax
        coeff = tc.zeros((NTA, nmax), dtype=ρ.dtype, device=ρ.device)

        for spe in self.species:
            mask = spe == symbols
            coeff[mask] = self.moduledict[str(spe)](ρ[mask])
        return coeff


class REANN(nn.Module):
    """ Recursive Embedding Atomic Neural Network

    Parameters
    ----------

    symbols: List
      type of elements

    """
    def __init__(self, species=None, rcut=6., lmax=2, nmax=2, loop=1,
                 norb=None, device='cpu', modulelist=None, **kwargs):
        super(REANN, self).__init__()

        self.device = device
        self.species = species
        norb = norb or int((nmax + 1)*nmax*(lmax+1) / 2)
        # Just because you can write in one line
        malloc = self.register_buffer
        malloc('nmax', tc.Tensor([nmax]).long().to(device=device))
        malloc('lmax', tc.Tensor([lmax]).long().to(device=device))
        malloc('loop', tc.Tensor([loop]).long().to(device=device))
        malloc('norb', tc.Tensor([norb]).long().to(device=device))
        malloc('rcut', tc.Tensor([rcut]).to(device=device))

        assert len(species) == max(species) + 1, "Use compressed expression"
        NS = len(species)
        NO = int((3 ** (lmax+1) - 1) / 2)
        Oidx = []
        for i in range(lmax + 1):
            Oidx.extend([i] * int(3**i))

        self.register_buffer('NS', tc.Tensor([NS]).long().to(device=device))
        self.register_buffer('NO', tc.Tensor([NO]).long().to(device=device))
        self.Oidx = Oidx

        self.α = Parameter(-(tc.rand(NS, nmax, device=device))
                           - tc.log(self.rcut))
        self.rs = Parameter(tc.rand(NS, nmax, device=device))
        # NS x nmax
        self.species_params = Parameter(tc.rand(NS, nmax).to(device=device))
        # Loop x lmax x nmax x norb
        self.orbital_params = Parameter(tc.rand(nmax, norb)[None, None].repeat(
                                  loop + 1, lmax+1, 1, 1).to(device=device))

        if modulelist is None:
            gj = self.init_modulelist()
        else:
            gj = modulelist
        self.gj = gj

    def init_modulelist(self):
        modulelist = nn.ModuleList()
        for j in range(self.loop):
            descdict = nn.ModuleDict()
            for spe in self.species:
                descdict[str(spe)] = nn.Sequential(
                        nn.Linear(self.norb, 128),
                        nn.LeakyReLU(),
                        nn.Linear(128, self.nmax)
                )
            modulelist.append(descdict)

        return modulelist

    def forward(self, symbols, positions, cells, pbcs, energyidx, crystalidx):
        """
        Parameters
        ----------
        symbols: NTA tensor
          Element of atoms corresponds to positions
        positions: NTAx3 tensor
          Atomic positions
        cells: NCx3x3 tensor
          Cell of each crystal
        crystalidx: NTA
          Crystal index corresponds to positions
        pbcs: NCx3 tensor
          Periodic boundary condition of each crystal

        Returns
        -------
        NTAx norb Tensor{Double}
            density ρ
        """
        device = self.device

        # Number of crystals, Number of total atoms
        iidx, jidx, isym, jsym, disp, dist = \
            get_neighbors_info(symbols, positions, cells, pbcs, energyidx,
                               crystalidx, cutoff=self.rcut, device=device)

        NTA = len(positions)
        dtype, device = positions.dtype, positions.device
        NO, Oidx = self.NO, self.Oidx
        nmax, lmax, rcut = self.nmax, self.lmax, self.rcut

        # NN -> NN
        fcut = cutoff_function(dist, rcut)
        # NN -> NN
        radial = gauss(dist, jsym, self.α, self.rs)
        # NN x NO
        angle = angular(disp, lmax)
        # NNx1x1 x NNxNOx1 x NNx1xnmax -> NNxNOxnmax
        Fxyz = (fcut[..., None, None] * angle[..., None] * radial[:, None])

        # NN number of total neighbors
        # NSxnmax -> NTA x nmax
        Csn = self.species_params[symbols]
        # Loop x lmax x nmax x norbit -> NO x nmax x norbit
        Wln = self.orbital_params[0, Oidx]

        params = (device, dtype, NTA, NO, nmax)
        # NTA x norb
        ρ = get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
        for i in range(self.loop):
            # NTAxO -> NTAxnmax
            Csn = Csn + self.gj[i](ρ, symbols)
            # Loop x lmax x nmax x norb -> NO x nmax x norb
            Wln = self.orbital_params[i+1, Oidx]
            # NTA x norb
            ρ = get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
        return ρ
