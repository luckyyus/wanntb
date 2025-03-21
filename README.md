# wanntb

Toolkits based on WF based tight-binding Hamiltonian

## Achieved
- restore Tight-binding data in a numpy npz binary file,
- Eigenvalues and eigenvectors for a single k-point,
- Gilbert damping parameter $\alpha$,
- non-adiabatic STT parameter $\beta$,
- Berry curvature calculations,
- Orbit moment calculations

## Usage

### Initialize the TB object
Get the TB object of a WF based tight-binding Hamiltonian:
```
tb = wanntb.TBSystem(tb_file='wannier90_tb.dat')
```
Store the TB system into a numpy npz binary file:
```
tb.output_npz(seedname) # the file [seedname]-tb.npz is created
```

Get the TB object from a npz binary file stored by output_npz:
```
tb = wanntb.TBSystem(npz_file='wannier90-tb.npz')
# if [npz_file] is None, then get the object from [tb_file]
```

### Eigenvalues and Eigenstates

#### Calculate eigenvalues and eigenvectors for a single k-point:
```
eig, uu = tb.get_eig_uu_for_one_kpt(kpt)
```

#### Calculate $\alpha$ and $\beta$
```
ef = xxx
mag = xxx
alpha, alpha_qvd, qvs, beta, ratio = tb.get_alpha_beta((64, 64, 64), ef, mag, eta=1e-3, q=1e-6, adpt_mesh=None)
```
To use adaptive k-mesh, set `adpt_mesh=(4,4,4)` 

### Berry curvature related calculations

#### Calculate AHC in terms of Fermi energies
```
ef_begin = 2.02
ef_end = 2.42
num_ef = 400
tb.get_ahc_kmesh_fermi((kmesh,kmesh,1), 2.ef_end, 2.4200, num_ef, eta=5e-4)
```
The current computational approach for AHC is not efficient 
because the existing algorithm first calculates Berry Curvature and then sums over all occupied states, 
resulting in significant computational redundancy.

#### Calculate the berry curvature component of orbit moment along a $k$-path
```
kpath = np.array([[0.00, 0.00, -0.50], [0.00, 0.00, 0.50]])
ef = 5.00
output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=500, direction=3, eta=1e-2)

```
The first column is the 1D k-point length in units angst.$^{-1}$.
The second column is the orbit moment in units $\mu_B$


## To do list

- Carrier density
- CUDA GPU  