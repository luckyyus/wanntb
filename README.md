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
Load the spin file ('wannier90_SS_R.dat') to the TB object:
```
tb.load_spins(ss_file='wannier90_SS_R.dat')
```
Store the TB system into a numpy npz binary file:
```
tb.output_npz(seedname) # the file [seedname]-tb.npz is created
```
If spins has been loaded, .npz file contains spin data.

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

#### Calculate eigenvalues inside an ellipsoidal reciprocal space around one k-point :
```
kmesh = (64, 64, 64) # this is the k-mesh for the whole BZ
center = (0.0, 0.0, 0.0) # the center k-point in reduced coordinates
distance_cart = (0.25, 0.25, 0.03) # the axis for the ellipsoidal space in Cartesian coordinates
eigs, kpts_frac, kpts_cart = tb.get_eig_for_kpts_around(kmesh, center, distance_cart)
```
The unit of k-point/distances in Cartesian coordinates is angst.$^{-1}$.

### Density-of-state (DOS) and projected-DOS (PDOS)

#### Calculate occupancy and DOS 
```
kmesh = (64, 64, 64)
e_min = 1.0
e_max = 11.0
n_ef = 2000
ef_range = (e_min, e_max, n_ef)
occ, dos = tb.get_occ_dos_kmesh_fermi(kmesh, ef_range, eta=1e-4, lproj=False)
# if lproj=True, the PDOS for each WF will be calculated
```
`occ` and `doc` are the occupancy and DOS as a function of energies. 
For `lproj=False`, the shapes of `occ` and `doc` are both `(n_ef+1, 2)`.
The first column is for the energy and the second column is for the corresponding results (the occupancy and DOS).
For `lproj=True`, the shapes of `occ` and `doc` are both `(n_ef+1, 2+num_wann)`
where `num_wann` is the number of WF.
The WF-projected results (occupancies or PDOS for each WF orbitals) are 
in the corresponding columns following the first two.

### Magnetic related properties

#### Calculate $\alpha$ and $\beta$
```
ef = xxx
mag = xxx
alpha, alpha_qvd, qvs, beta, ratio = tb.get_alpha_beta((64, 64, 64), ef, mag, eta=1e-3, q=1e-6, adpt_mesh=None)
```
To use adaptive k-mesh, set `adpt_mesh=(4,4,4)` 

We can also calculate $\alpha$ and $\beta$ with a list of Fermi energy 
```
ef0 = xxx # the center of the Fermi energy list
mu_d = 0.01 # the Fermi energy list is ranged from ef0-mu_d to ef0+mu_d
n_ef = 10 # Total number of Fermi energy is n_ef+1, the endpoint is included
out = tb.get_alpha_beta_fermi((kmesh, kmesh, kmesh), ef0, mu_d, n_ef, mag, eta=eta, adpt_mesh=(4,4,4))
```
The output data is in the shape (n_ef+1, 6).
The columns represent: mu, alpha, alpha_qvd, qvs, beta, ratio

### Berry curvature related calculations

The function for berry curvature related calculations is `berry_calc_fermi` and `berry_calc_kpath`, 
which correspond to the calculated quantity as a function of Fermi energy and along a $k$-path respectively.

#### The basic usage of `berry_calc_fermi`

Calculate berry curvature related quantities as a function of Fermi energy.
```
kmesh = (64, 64, 64)
e_min = 1.0
e_max = 11.0
n_ef = 2000
ef_range = (e_min, e_max, n_ef)
tb.berry_calc_fermi('ahc+shc+morb', kmesh, ef_range, eta=1e-4, xyz=2, subwf=None)
```
Here the first argument `tasks` can be a combination of
- `ahc` -- anomalous Hall conductivity (**AHC**) in units $e^2/h/Å$, 
- `shc` -- spin Hall conductivity (**SHC**) in units $(\hbar/2e) e^2/h/Å$
- `morb` -- orbit moment (**Morb**) in units $\mu_B$ per u.c. 

with the connector `+`.

`xyz=0,1,2` corresponds to 
- $\alpha\beta=yz, zx, xy$ for **SHC** calculations 
- orbit moment direction $\gamma$ for **Morb** calculations.

The output is a table with the first column of Fermi energies.
Every three columns thereafter correspond to one type of the calculated quantity:
- For **AHC** $\sigma^A_{yz}$, $\sigma^A_{zx}$, $\sigma^A_{xy}$ is calculated.
- For **SHC** $\sigma^{S(x)}_\alpha\beta$, $\sigma^{S(y)}_\alpha\beta$, $\sigma^{S(z)}_\alpha\beta$ is calculated 
(so that argument `xyz` is required).
- For **Morb** $M^{\gamma}_{L1}$ (the local part), 
$M^{\gamma}_{L2}$ (the itinerant  part) 
and $M^{\gamma}_{L}=M^{\gamma}_{L1}+M^{\gamma}_{L2}$ (total) is calculated.

#### The basic usage of `berry_calc_kpath`

Calculate berry curvature related quantities along a $k$-path.
```
ef = 2.000 # one Fermi energy
kpath = np.array([[0.00, 0.00, -0.50], [0.00, 0.00, 0.50]])
berry_calc_kpath('ahc+shc+morb', ef, kpath, nkpts_path=100, eta=1e-4, xyz=2, subwf=None)
```
The output is similar to `berry_calc_fermi` 
while the first column in the output of `berry_calc_kpath` is a $k$-path of the 1D k-point length in units Angst.$^{-1}$.


## To do list

- Carrier density
- CUDA GPU  