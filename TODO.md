
-----

Add the following Julia based reference solvers to tests/reference_solvers/julia: Tsit5,  KenCarp5, Rodas5 and Kvaerno5. Use the DiffEqGPU.jl package to ensure the Julia ODE solvers run on the GPU. We want to be able to call the Julia based reference solvers from Python tests of the form test_julia_tsit5, so create an entry point Julia script that the Python test can execute, passing it the name of the test ODE system (which will have to reimplemented in Julia alongside the solvers in tests/reference_solvers/julia), test parameters like ensemble size and system dimensions, along with all the args that usually get passed to the python solvers, like the batch of parameters and initial conditions, and output times. Separate the definition of the ODE system in Julia from the code used to invoke the solver, since multiple different Julia solvers will be tested on the same ODE system. The script should return its outputs to Python for the usual consistency checks (though not any correctness/accuracy checks that are not analytically exact and therefore require a separate reference solver). Parameterise the tests on the usual ensemble size, dimensionality, and also on whether to run the ODE with EnsembleGPUArray or EnsembleGPUKernel. In a README.md in the tests/reference_solvers/julia folder explain the difference between EnsembleGPUArray and EnsembleGPUKernel. The docs for Julia's DiffEqGPU.jl package are here: https://docs.sciml.ai/DiffEqGPU/stable/ and the code can be found here: https://github.com/SciML/DiffEqGPU.jl. There is no need to create a make_cached_solver because we will not be using this reference solver to assert the correctness of existing test results. Create new tests for all the Julia reference solvers for all the test ODE systems. For the Julia kencarp5 solver this should run with the same IMEX splits that our own version of kencarp5 runs with.

-----

Shouldn't rodas5.py's stages evaluate ode_fn at different times? The current code is only correct for autonomous ODE systems right? Fix this and check whether this issue exists in any of the other ODE solvers, then fix them too.

-----

Implement a new IMEX solver named kencarpgersh5.py that is a dynamic version of kencarp5 but instead of taking a split ode or jacobian function, it takes a single ode or jacobian, and at every step in the loop of the solver splits the resulting jacobian into stiff and non-stiff parts using Gershgorin's circle theorem and subsequent reodering of the jacobian matrix. Please write a docstring as part of the solver files that describes the method in detail, what it aims to do and in what situations it is better than the ordinary kencarp5.py method.

Here is an overview of the idea:

The **Gershgorin Circle Theorem** is a powerful shortcut because it allows us to estimate the "danger zone" of a matrix's eigenvalues without actually solving a characteristic equation. When we use it to partition a matrix $M$, we are essentially sorting the rows of the system into "fast" and "slow" categories.

Here is the step-by-step breakdown of how you would perform this partition.

## 1. The Core Logic: Estimating Local "Speed"

Recall that for each row $i$ of your matrix $M$, the eigenvalues $\lambda$ must lie within a disk $D_i$ centered at the diagonal element $M_{ii}$ with a radius $R_i$ equal to the sum of the absolute values of the off-diagonal elements:


$$
R_i = \sum_{j \neq i} |M_{ij}|
$$

The "worst-case" eigenvalue associated with row $i$ is roughly $|M_{ii}| + R_i$. If this value is large, that specific row represents a variable (or a coupling) that evolves very quickly, potentially destabilizing an explicit solver.

## 2. The Partitioning Algorithm

### Step 1: Define your "Stability Speed Limit" (τ)

Before you look at the matrix, you must decide your threshold $\tau$. This is usually determined by your desired time step $\Delta t$ and the stability region of the explicit solver you *want* to use.

- For example, if you're using Forward Euler, you might set $\tau = 2/\Delta t$.

### Step 2: Calculate Row Magnitudes

For every row $i = 1, \dots, n$:

1. **Center:** $c_i = M_{ii}$
2. **Radius:** $R_i = \sum_{j \neq i} |M_{ij}|$
3. **Upper Bound:** $\rho_i = |c_i| + R_i$

### Step 3: Classify the Rows

Now, you split the indices of your system into two sets:

- **Stiff Set (S):** All indices $i$ where $\rho_i > \tau$.
- **Non-Stiff Set (N):** All indices $i$ where $\rho_i \leq \tau$.

### Step 4: Assemble the Matrices

You can now create your additive split $M = M_{\text{stiff}} + M_{\text{non-stiff}}$ based on these sets. 

Simply put the "fast" rows into the stiff matrix and the "slow" rows into the non-stiff matrix.

- **Mstiff​**: Contains rows $i \in \mathcal{S}$ of the original matrix $M$, and all other rows are zero.
- **Mnon-stiff​**: Contains rows $i \in \mathcal{N}$ of the original matrix $M$, and all other rows are zero.

## 3. Why this works for IMEX

When you plug this into an IMEX scheme, your implicit step only "cares" about the rows in $M_{\text{stiff}}$.
If your $M_{\text{stiff}}$ is sparse or has a specific structure (like being mostly zero except for a few rows), the linear solve $(I - h\gamma M_{\text{stiff}})$ becomes much cheaper. In the example above, the implicit solve only actually involves $y_2$, while $y_1$ can be updated explicitly.

### The Trade-off

The Gershgorin partition is a **conservative heuristic**. It might over-identify rows as stiff because it assumes the worst-case scenario (the disks are just bounds, not the actual eigenvalues). However, it is **incredibly fast** ($O(n^2)$ for dense, $O(nnz)$ for sparse) because it requires no matrix inversions or iterative solvers to perform the split.

This allows us to do Reduced-Order Solving, which is implemented as follows:

## 1. The "Identity" Save

When you perform an IMEX step, you aren't just solving $M_{\text{stiff}} \mathbf{y} = \mathbf{b}$. You are solving the system that comes from the implicit integration rule (like Backward Euler or a DIRK stage):


$$
(I - h\gamma M_{\text{stiff}}) \mathbf{y}_{n+1} = \text{rhs}
$$

If you have used Method A to zero out the "non-stiff" rows in $M_{\text{stiff}}$, look at what happens to those rows in the total matrix $A = (I - h\gamma M_{\text{stiff}})$:

- In a zeroed-out row $i$, the $M_{\text{stiff}}$ part is $0$.
- The $I$ (Identity) part remains $1$ on the diagonal.
- Therefore, the $i$-th row of the system is simply: $[0, \dots, 0, 1, 0, \dots, 0] \mathbf{y}_{n+1} = \text{rhs}_i$.

This means the variable $y_i$ is already "solved"—it is simply equal to the right-hand side. The system remains **square**, but it becomes **partially decoupled**.

## 2. Block Partitioning (The "Trimming" Logic)

To make this computationally efficient, you can reorder your variables so that all the non-stiff variables come first and all the stiff variables come last. Your system then takes on a **Block Lower Triangular** structure:


$$
\begin{bmatrix} I & 0 \\ \text{Coupling} & I - h\gamma M_{ss} \end{bmatrix} \begin{bmatrix} \mathbf{y}_{non-stiff} \\ \mathbf{y}_{stiff} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_{non-stiff} \\ \mathbf{b}_{stiff} \end{bmatrix}
$$

### The Workflow:

1. **Solve the Top Block:** $\mathbf{y}_{non-stiff} = \mathbf{b}_{non-stiff}$. (This is instant; no math required).
2. **Substitute into the Bottom Block:**


$$
(I - h\gamma M_{ss}) \mathbf{y}_{stiff} = \mathbf{b}_{stiff} - (\text{Coupling} \times \mathbf{y}_{non-stiff})
$$
3. **The Reduced Solve:** Now you only have to perform an **LU factorization** on the matrix $(I - h\gamma M_{ss})$.

This matrix is **square**, it's much smaller than the original $M$, and it represents only the "stiff-on-stiff" interactions.

## 3. When does this benefit disappear?

The only time this "trimming" strategy fails is if your **stiff variables are coupled to every other variable.** If every row has at least one "fast" interaction, you can't zero out any rows. But in physical systems (like a power grid or a large chemical network), stiffness is often localized to a small subset of "fast" components. In those cases, the size of the matrix you actually need to factorize might be only 10% or 20% of the total system size.

-----

Some info:

**[test_reversible_trapping.py](vscode-webview://111ju7d7ot3l2i80tg802elhjuhmqnjafhjedh9dnf3j07to9b0h/tests/linear/test_reversible_trapping.py)** is the closest structural match.

Einstein-Boltzmann linearized CMB has four defining features: (1) linear stiff, (2) block-coupled fast/slow subsystems — Thomson scattering tightly couples photons↔baryons on a much faster timescale than gravitational evolution, (3) spatial/mode transport, (4) mass/energy conservation. Reversible trapping hits all four: a mobile species with Laplacian diffusion (slow transport, like metric/acoustic evolution) coupled to an immobile species via fast `k_on`/`k_off` exchange (tight-coupling analog), with stiffness ~1e4 from the scale separation and exact conservation.

Secondary matches:

- **[test_nn_reactions.py](vscode-webview://111ju7d7ot3l2i80tg802elhjuhmqnjafhjedh9dnf3j07to9b0h/tests/linear/test_nn_reactions.py)** — tridiagonal linear structure resembles the photon multipole hierarchy's ℓ↔ℓ±1 coupling, and its ~1e8 stiffness mimics the Thomson rate, but it lacks the fast/slow block separation.
- **[test_bateman.py](vscode-webview://111ju7d7ot3l2i80tg802elhjuhmqnjafhjedh9dnf3j07to9b0h/tests/linear/test_bateman.py)** — linear stiff with controllable ratio, but the pure unidirectional decay misses the back-reaction coupling that defines the E-B system.

None of the tests capture the oscillatory acoustic character; if that matters for your solver validation, reversible trapping + a damped-rotation-style oscillatory test together would cover more of the CMB system's behaviour than any single test alone.

---

Run the test_reversible_trapping ODE system with dimensions 70D and ensemble size 1k, varying the solver, linear vs nonlinear paths, and 32 bit vs 64 bit precision. Each timing should be done sequentially to avoid saturating the GPU. If a given solver+params takes more than 10s, abort that test and simply record it as >10s. Plot the resulting timings in a bar chart with solver+params on the x axis, and time to compute 1k ODEs on the y axis. The x axis should be ordered in terms of decreasing time.