# wanntb

Toolkits based on WF based tight-binding Hamiltonian

## Achieved

- Eigenvalues and eigenvectors for a single k-point,

- Gilbert damping parameter $\alpha$,

- non-adiabatic STT parameter $\beta$

## Usage

get the object of a WF based tight-binding Hamiltonian:
```
tb = wanntb.TBSystem(tb_file='wannier90_tb.dat')
```

Calculate eigenvalues and eigenvectors for a single k-point:
```
eig, uu = tb.get_eig_uu_for_one_kpt(kpt)
```


Calculate $\alpha$ and $\beta$
```
alpha, alpha_qvd, qvs, beta, ratio = tb.get_alpha_beta((64, 64, 64), ef, mag, eta=1e-3, q=1e-6)
```


## To do list

- Carrier density
- CUDA GPU  