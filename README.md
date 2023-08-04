# Exact Diagonalization of the Angulon

Exact Diagonalization (ED) provides a simple framework to determine the eigenstates and energy eigenvalues of a quantum Hamiltonian. It obtains Green's functions directly in real time, as opposed to imaginary time as in Diagrammatic Quantum Monte Carlo (DQMC). Here we are interested in a molecular impurity immersed in a Bose-Einstein Condensate (BEC) or superfluid helium. This problem can be conveniently studied using the concept of the angulon quasiparticle which consists of a quantum rotor dressed by a field of many-body excitations.

## Implementation Choices

The ED algorithm requires an efficient representation of ket states to handle the large number of basis vectors effectively. In our implementation, we define a ket state $\ket{\psi} = \ket{\psi_{\text{imp}}} \otimes \ket{\psi_{\text{ph}}}$, with $\ket{\psi_{\text{imp}}} = \ket{L,n}$ denoting the quantum numbers for the impurity, and $\ket{\psi_{\text{ph}}}$ representing the phononic configurations. The phonon ket is defined as follows:

$\ket{\psi_{\text{ph}}} = \ket{S_0,n_{S_0}, ..., S_{\text{max}}, n_{S_{\text{max}}}},$

where $n_{S_i}$ denotes the number of phonons in state $S_i$. Note that we have neglected the good quantum number $M$, since we do not assume any external field that would break rotational symmetry. To save memory, the single-particle phonon state label $S_i \in \\{0, 1, ..., |\\{S_i\\}|\\}$ only appears in the internal representation if it is occupied by at least one phonon. The total number of available phononic states is given by

$$\| \\{S_i\\} \| = S_{\text{max}} + 1 = \sum_{l=0}^{l_{\text{max}}} \left(\frac{k_{\text{max}} - k_{\text{min}}}{\Delta k} + 1\right) (2l+1).$$

Here, $k_{\text{min}}$, $k_{\text{max}}$, $\Delta k$, and $l_{\text{max}}$ are the chosen numerical parameters for the momentum discretization and the maximal phonon angular momentum quantum number (see below). We neglect any external field's influence on the system and thus maintain rotational symmetry.

To optimize memory usage, we utilize the hashing trick. This involves converting the large ket state universe into small practical integer values using a hash function. We choose an injective hash function that efficiently distributes keys and yields a compact hash table:

$$ h(\ket{\psi_{\text{ph}}}) = \sum_{n=0}^{n_{\text{max}}-1} (S_{n} + 1) S_{\text{max}}^n.$$

The sum runs over each phonon index $n$ in state $\ket{\psi_{\text{ph}}}$. Since the hash value might still become considerably large, we further save memory by retaining only the remainder of the hash values after division by some predetermined divisor, for instance, $20$.

## Numerical Parameters

In addition to the physical parameters such as the bath density $n_0$ and the shape of the molecule-boson interaction potential, the numerical discretization parameters that have to be set by the user to perform calculations in this work are:

- Momentum discretization: $\Delta k$
- Maximum momentum: $k_{\text{max}}$
- Infinitesimal imaginary offset for spectral function: $\epsilon$
- Maximum phonon angular momentum quantum number: $l_{\text{max}}$
- Maximum total angular momentum quantum number: $L_{\text{max}}$
- Number of phonons: $N$

## Dependencies

The implementation employs the following libraries:

- Cython with `cython.parallel` module for CPU-bound tasks and `cython_gsl` for GSL integration
- SciPy's `scipy.sparse.csr_matrix` to handle sparse Hamiltonian matrices
- SciPy's `scipy.linalg.cython_lapack` to access `zheevr` and `zgeev` functionalities from LAPACK
- SymPy's `sympy.physics.quantum.cg` for Clebsch-Gordan coefficients

## Usage

To run the ED calculations and compute the angulon spectral function or phonon density profiles, set the physical and numerical parameters in the `config/config.py` file. Then run the `main.py` script from the `run` folder. The results will be dumped into the `output` folder. Make sure to satisfy the dependencies mentioned above before running the code.
