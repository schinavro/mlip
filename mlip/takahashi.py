import torch as tc
from torch import nn
from torch.nn.parameter import Parameter

from mlip.utils import get_neighbors_info, partitions
from math import factorial


def chebyshev_matrix(lmax):
    """ Generate Chebyshev polynomials of the first kind by recurrence definition
        
    https://en.wikipedia.org/wiki/Chebyshev_polynomials
    
    Parameters
    ----------
     lmax: int
     
    Return
    ------
     (lmax, lmax) shape tensor
    """
    Tlm = tc.zeros(lmax, lmax)
    for l in range(lmax):
        if l == 0:
            Tlm[l, 0] = 1
            continue
        elif l == 1:
            Tlm[l, 1] = 1
            continue
        Tlm[l] = 2 * tc.roll(Tlm[l-1], 1) - Tlm[l-2] 
    return Tlm


class Gj(nn.Module):
    def __init__(self, moduledict, species=None, nmax=None, lmax=None):
        super(Gj, self).__init__()

        self.moduledict = moduledict
        self.species = species
        self.nmax = nmax
        self.lmax = lmax

    def forward(self, ρ, symbols):
        """  Returns the

        Parameters
        ----------

        ρ : NTA x nmax x lmax
        symbols : NTA number index

        Return
        ------
        NTA x nmax x NO
        """
        NTA, nmax, lmax = len(ρ), self.nmax, self.lmax
        coeff = tc.zeros((NTA, nmax * lmax), dtype=ρ.dtype, device=ρ.device)

        for spe in self.species:
            mask = spe == symbols
            N = tc.sum(mask).long()
            coeff[mask] = self.moduledict[str(spe)](ρ[mask])
            
        return coeff

class Takahashi(nn.Module):
    """ Chebyshev polynomial Embedding atomic neural net
    
    Attribute
    ----------

    symbols: List
      type of elements

    """
    def __init__(self, species=None, nmax=2, lmax=2, loop=0, 
                 rcut=tc.tensor([6.]), 
                 device='cpu', modulelist=None,
                 **kwargs):
        super(Takahashi, self).__init__()

        self.device = device
        self.species = species
        # Just because you can write in one line
        malloc = self.register_buffer
        malloc('nmax', tc.Tensor([nmax]).long().to(device=device))
        malloc('lmax', tc.Tensor([lmax]).long().to(device=device))
        malloc('loop', tc.Tensor([loop]).long().to(device=device))
        malloc('rcut', tc.Tensor([rcut]).to(device=device))
        
        pqrs, pqr_idx, clm, NO = [], [], [], 0
        for L in range(lmax):
            npqr = 0
            for pqr in partitions(L, 3):
                pqrs.append(pqr[None])
                p, q, r = pqr
                m = p + q + r
                m1 = factorial(m)
                p1, q1, r1 = factorial(p), factorial(q), factorial(r)
                clm.append(m1/(p1*q1*r1))
                npqr += 1

            pqr_idx.extend([L] * npqr)
            NO += npqr
    
        NS = len(species)

        #self.NS = NS
        #self.NO = NO

        self.register_buffer('NS', tc.Tensor([NS]).long().to(device=device))
        self.register_buffer('NO', tc.Tensor([NO]).long().to(device=device))
        self.register_buffer('pqr_idx', tc.tensor(pqr_idx).long().to(device=device))
        self.register_buffer('pqrs', tc.cat(pqrs).long().to(device=device))

        # self.pqr_idx = tc.tensor(pqr_idx).to(device=device)
        # self.pqrs = tc.cat(pqrs).to(device=device)
        # lmax x lmax 
        Tlm = chebyshev_matrix(lmax).to(device=device)
        self.register_buffer('Tlm', Tlm.long().to(device=device))
        
        # NS x nmax x lmax
        self.alpha = Parameter(tc.rand(NS, nmax, lmax))
        self.μ = Parameter(tc.rand(NS, nmax, lmax))
        self.σ = Parameter(-(tc.rand(NS, nmax, lmax))
                            - tc.log(self.rcut))

        #self.alpha = tc.rand(NS, nmax, lmax).double().requires_grad_(False)
        #self.μ = tc.rand(NS, nmax, lmax).double().requires_grad_(False)
        #self.σ = (-(tc.rand(NS, nmax, lmax)) - tc.log(self.rcut)).double().requires_grad_(Truee)

        # NS x nmax
        # self.species_params = Parameter(tc.rand(NS, nmax).to(device=device))
        # Loop x nmax x lmax x lmax
        self.hopping_params = Parameter(tc.rand(lmax, lmax)[None, None].repeat(
                                         loop + 1, nmax, 1, 1).to(device=device))
        # self.hopping_params = tc.rand(lmax, lmax)[None, None].repeat(loop + 1, nmax, 1, 1).to(device=device).requires_grad_(False)

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
                        nn.Linear(self.nmax * self.lmax, 128),
                        nn.LeakyReLU(),
                        nn.Linear(128, self.nmax * self.lmax)
                ).double()
            modulelist.append(Gj(descdict, species=self.species, 
                                 nmax=self.nmax, lmax=self.lmax))

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
        NTAx nmax*lmax Tensor{Double}
            density ρ
        """
        # Number of crystals, Number of total atoms
        
        device, dtype = self.device, positions.dtype
        nmax, lmax, NO = self.nmax, self.lmax, self.NO
        
        NTA = len(symbols)

        iidx, jidx, isym, jsym, disp, dist = \
            get_neighbors_info(symbols, positions, cells, pbcs, energyidx,
                               crystalidx, cutoff=self.rcut, device=device)

        # NN -> NN
        fcut = self.cutoff_function(dist, self.rcut)
        # NN -> NN x nmax x lmax
        radial = self.gauss(dist, jsym, self.alpha, self.μ, self.σ)
        
        # NN x NNx3 -> NN x NO
        angular = self.chebyshev_series(dist, disp, self.pqrs)
        
        # NNx1x1 x NNx(nmax)x(lmax) -> NNx(nmax)xNO
        fnr = fcut[:, None, None] * radial[..., self.pqr_idx]
        
        # NNx(nmax)xNO x NNx1xNO -> NNx(nmax)xNO
        fnr_xyzr = fnr * angular[:, None]
                
        # NS x nmax x lmax -> NTA x nmax x lmax
        cn = (self.alpha[symbols])
        
        # nmax x lmax x lmax
        hop = self.hopping_params[0]
        
        params = (iidx, jidx, self.pqr_idx, self.Tlm, NTA, nmax, lmax, NO, dtype, device)
        
        # NTA x nmax x lmax
        bnl = self.get_density(hop, cn, fnr_xyzr, *params)
        
        for i in range(self.loop):
            # NTAx(nmax)xlmax + NTAx(nmax)xlmax -> NTA x nmax x lmax
            cn = cn + self.gj[i](bnl, symbols)
            print('CN', cn)
            # NS x NS x nmax x lmax x lmax -> NN x nmax x lmax x lmax -> NTA x nmax x lmax x lmax
            hop = self.hopping_params[i+1]
            
            # NTA x nmax x lmax
            bnl = self.get_density(hop, cn, fnr_xyzr, *params)

        # NTA x (nmax x lmax)
#        return tc.sum(bnl, axis=1)
        return bnl
#       return bnl.view(NTA, self.nmax * self.lmax)

    def cutoff_function(self, dist, rcut):
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
    
    def gauss(self, x, jsym, α, μ, σ):
        """
        x : NN
        sig : NS x nmax x lmax
        rs : NS x nmax x lmax
        return : NN x nmax x lmax
        """
        # NSxnmaxlmax -> NNxnmaxxlmax
        μ = μ[jsym]
        σ = σ[jsym]
    #    α = α[jsym]
        # NN x nmax x lmax - NN x 1 x 1 -> NN x nmax x lmax
    #    return α * tc.exp(-0.5 * ((x[:, None, None] - μ)/σ)**2)
        return tc.ones(len(jsym), self.nmax, self.lmax)
        # return tc.exp(σ* ((x[:, None, None] - μ))**2)

    def chebyshev_series(self, dist, disp, pqrs):
        """ Calculate b_nl^(ij) where
    
        $$ \sum_m T_{lm} \sum_{p+q+r=m} \frac{m!}{p!q!r!} f_n(r_ij) \frac{x^p y^q z^r}{r^m} $$
    
        Parameters
        ----------
    
        """
        bnlm = []
    
        for pqr in pqrs:
            # NNx3 x NN -> NN
            xyz_r = tc.prod(disp ** pqr, axis=1) / dist ** tc.sum(pqr)
            bnlm.append(xyz_r[:, None])
    
        # NN x NO
        return tc.cat(bnlm, axis=1)
    
    def get_density(self, hop, cn, fnr_xyzr, iidx, jidx, pqr_idx, Tlm, NTA, nmax, lmax, NO, dtype, device):
        
        # NTA x nmax x lmax -> NN x nmax x NO
        Cn = (cn[jidx])[..., pqr_idx]
        
        # NNx(nmax)xNO -> NTA x nmax x NO
        bnl2 = tc.zeros((NTA, nmax, NO), device=device, dtype=dtype
                        ).index_add(0, iidx, Cn * fnr_xyzr) ** 2
    
        # NTAx(nmax)xNO -> NTA x nmax x lmax
        bnl1 = tc.zeros((NTA, nmax, lmax), device=device, dtype=dtype
                        ).index_add(2, pqr_idx, bnl2)
           
        # 1x1x(lmax)x(lmax) x NTAx(nmax)x(lmax)x1 -> NTA x nmax x lmax
        bnl = tc.sum(Tlm[None, None] * bnl1[..., None], axis=3)
        
        # 1x(nmax)x(lmax)x(lmax) x NTAx(nmax)x(lmax)x1 -> NTA x nmax x lmax
        return tc.sum(hop[None] * bnl[..., None], axis=3)
