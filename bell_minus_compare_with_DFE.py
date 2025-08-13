
# bell_minus_compare_with_DFE.py
# Side-by-side: Full tomography vs Classical shadows vs DFE for a probabilistic Bell-minus state.
# Save this file and run with Python 3. Requires: numpy, scipy, matplotlib

import numpy as np
import itertools, math
from numpy.linalg import eig
from scipy.linalg import sqrtm, eigh
import matplotlib.pyplot as plt

# --- Basic operators ---
I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

zero = np.array([1,0], dtype=complex)
one  = np.array([0,1], dtype=complex)

# Bell-minus and projector
psi_minus = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)
proj_psi_minus = np.outer(psi_minus, psi_minus.conj())

# mixed true state: p * |psi-> + (1-p) * |00>
psi00 = np.kron(zero, zero)
proj_00 = np.outer(psi00, psi00.conj())
p = 0.5
rho_true = p * proj_psi_minus + (1-p) * proj_00
d = 4

# Single-qubit Cliffords via H and S closure (small generation)
H = (X + Z) / np.sqrt(2)
S = np.array([[1,0],[0,1j]], dtype=complex)

def generate_cliffords():
    mats = [np.eye(2, dtype=complex)]
    # grow by multiplying H, S, X until stable
    ops = [H, S, X]
    group = {tuple(np.round(mats[0].flatten(),8))}
    changed = True
    while changed:
        changed = False
        new = []
        for g in mats:
            for op in ops:
                prod = op @ g
                key = tuple(np.round(prod.flatten(),8))
                if key not in group:
                    group.add(key); new.append(prod); changed=True
        if new: mats = new
    uniq = []
    for key in group:
        U = np.array(key, dtype=complex).reshape(2,2)
        # normalize global phase
        if abs(U[0,0])>1e-12:
            phase = np.angle(U[0,0])
            U = U * np.exp(-1j*phase)
        found = False
        for V in uniq:
            if np.allclose(U, V, atol=1e-6): found=True; break
        if not found: uniq.append(U)
    return uniq

Cliffords1q = generate_cliffords()[:24]
n_cliff = len(Cliffords1q)

# --- utilities ---
def sample_Z_after_R(rho, R1, R2, rng):
    R = np.kron(R1, R2)
    rho_rot = R @ rho @ R.conj().T
    probs = np.real(np.diag(rho_rot))
    probs = np.maximum(probs, 0); probs = probs / (probs.sum()+1e-14)
    k = rng.choice(4, p=probs)
    return k

def shadow_inverse_local(R1, R2, k):
    b1 = (k >> 1) & 1
    b2 = k & 1
    proj1 = np.outer(np.array([1,0]) if b1==0 else np.array([0,1]),
                     (np.array([1,0]) if b1==0 else np.array([0,1])).conj())
    proj2 = np.outer(np.array([1,0]) if b2==0 else np.array([0,1]),
                     (np.array([1,0]) if b2==0 else np.array([0,1])).conj())
    A1 = 3 * (R1.conj().T @ proj1 @ R1) - I2
    A2 = 3 * (R2.conj().T @ proj2 @ R2) - I2
    return np.kron(A1, A2)

# Pauli basis for DFE / tomography
paulis = {'I':I2, 'X':X, 'Y':Y, 'Z':Z}
pauli_strings = []
labels = []
for a in ['I','X','Y','Z']:
    for b in ['I','X','Y','Z']:
        pauli_strings.append(np.kron(paulis[a], paulis[b]))
        labels.append(a+b)

# rotate to computational basis for measuring Pauli P = A⊗B
def rotation_for_pauli(label):
    mapping = {'X': H, 'Y': S.conj().T @ H, 'Z': I2, 'I': I2}
    return np.kron(mapping[label[0]], mapping[label[1]])

# compute expectation Tr(P rho) exactly (simulation ground truth)
def pauli_expectation(rho, P):
    return np.real(np.trace(P @ rho))

# DFE importance sampling distribution for target |ψ^-⟩
# For Pauli basis P, s_P = Tr(P * ρ_target)
s_vals = np.array([pauli_expectation(proj_psi_minus, P) for P in pauli_strings])
# exclude identity-only term? keep all (including II)
# compute sampling probabilities proportional to s_P^2
prob_P = s_vals**2
if prob_P.sum() == 0:
    prob_P = np.ones_like(prob_P) / len(prob_P)
else:
    prob_P = prob_P / prob_P.sum()

# Function: estimate fidelity via DFE with m total shots
def DFE_estimate(rho_true, total_shots, rng):
    # choose K Pauli terms to sample (we'll sample up to total_shots distinct Pauli terms)
    K = min(len(pauli_strings), total_shots)
    shots_per_pauli = max(1, total_shots // K)
    # sample indices according to prob_P
    inds = rng.choice(len(pauli_strings), size=K, replace=True, p=prob_P)
    estimates = []
    weights = []
    for idx in inds:
        label = labels[idx]
        P = pauli_strings[idx]
        # measure P on the state rho_true using shots_per_pauli shots
        U = rotation_for_pauli(label)
        # simulate measurement probabilities in computational basis for U rho U†
        rho_rot = U @ rho_true @ U.conj().T
        probs = np.real(np.diag(rho_rot)); probs = np.maximum(probs,0); probs = probs/probs.sum()
        # for Pauli expectation, eigenvalues are ±1 mapping on computational basis depending on P diagonal
        # compute diagonal of P in computational basis after rotation (same as diag(U† Z U type)
        diagP = np.real(np.diag(U.conj().T @ P @ U))  # but simpler: diag of P in computational basis equals eigenvals per basis state
        # simulate counts
        counts = rng.multinomial(shots_per_pauli, probs)
        # empirical expectation for this Pauli
        emp = np.dot(diagP, counts / shots_per_pauli)
        # weight: s_P / prob_P[idx] times emp / d  (see Flammia-Liu estimator)
        sP = s_vals[idx]
        if prob_P[idx] == 0:
            continue
        estimates.append(emp * sP / prob_P[idx])
        weights.append(1.0)
    if len(estimates)==0:
        return 0.0
    # final estimator: (1/ (d * N)) sum estimates where d = 4
    F_est = (1.0 / (d * len(estimates))) * sum(estimates)
    # clip to [0,1]
    return max(0.0, min(1.0, np.real(F_est)))

# Tomography linear inversion (simple) using 9 settings X/Y/Z x X/Y/Z
def tomography_linear(rho_true, total_shots, rng):
    settings = list(itertools.product(['X','Y','Z'], repeat=2))
    shots_per_setting = max(1, total_shots // len(settings))
    expvals = {}
    Umap = {'X': H, 'Y': S.conj().T @ H, 'Z': I2}
    for s in settings:
        U1 = Umap[s[0]]; U2 = Umap[s[1]]
        U = np.kron(U1, U2)
        rho_rot = U @ rho_true @ U.conj().T
        probs = np.real(np.diag(rho_rot)); probs = np.maximum(probs,0); probs = probs / probs.sum()
        counts = rng.multinomial(shots_per_setting, probs)
        freqs = counts / shots_per_setting
        # expectation of Pauli A⊗B is diag(M)·freqs where M = U† (Z⊗Z) U in computational basis
        M = np.kron(U1.conj().T @ Z @ U1, U2.conj().T @ Z @ U2)
        diag_m = np.real(np.diag(M))
        exp_val = np.dot(diag_m, freqs)
        expvals[s] = exp_val
    # reconstruct rho from measured Pauli expectations
    rho = np.eye(4, dtype=complex) / 4.0
    for (a,b), val in expvals.items():
        M = np.kron({'X':X,'Y':Y,'Z':Z}[a], {'X':X,'Y':Y,'Z':Z}[b])
        rho += 0.25 * val * M
    # symmetrize and project to PSD trace-one
    rho = (rho + rho.conj().T) / 2
    vals, vecs = eigh(rho)
    vals = np.maximum(vals, 0)
    if vals.sum() == 0:
        rho = vecs[:, -1:] @ vecs[:, -1:].conj().T
    else:
        rho = (vecs @ np.diag(vals) @ vecs.conj().T) / vals.sum()
    return rho

# main experiment: compare methods on shot budgets
def run_experiment(shot_budgets, reps=20, seed=12345):
    rng = np.random.default_rng(seed)
    results = {'shots': [], 'tomog_1F_mean': [], 'tomog_1F_std': [],
               'shadows_1F_mean': [], 'shadows_1F_std': [],
               'dfe_1F_mean': [], 'dfe_1F_std': [],
               'tomog_trace': [], 'shadows_trace': [], 'dfe_trace': []}
    for total_shots in shot_budgets:
        tomog_errs = []; sh_errs = []; dfe_errs = []
        tomog_trace = []; sh_trace = []; dfe_trace = []
        for rep in range(reps):
            # tomography
            rho_tom = tomography_linear(rho_true, total_shots, rng)
            # shadows: collect total_shots single-shot shadow estimators and use MoM
            shadow_rhos = []
            for i in range(total_shots):
                # prepare probabilistic state
                rho_prep = proj_psi_minus if rng.random() < p else proj_00
                R1 = Cliffords1q[rng.integers(0, n_cliff)]
                R2 = Cliffords1q[rng.integers(0, n_cliff)]
                k = sample_Z_after_R(rho_prep, R1, R2, rng)
                shadow_rhos.append(shadow_inverse_local(R1, R2, k))
            # simple average and MoM median-of-means
            avg_shadow = sum(shadow_rhos) / len(shadow_rhos)
            B = min(9, max(3, int(math.sqrt(total_shots))))
            batch_size = max(1, total_shots // B)
            batch_means = []
            for b in range(B):
                start = b*batch_size; end = min(len(shadow_rhos), start+batch_size)
                if start >= end: break
                bm = sum(shadow_rhos[start:end]) / (end - start)
                batch_means.append(bm)
            mats = np.stack([bm.reshape(-1) for bm in batch_means], axis=0)
            med = np.median(mats, axis=0).reshape(4,4)
            shadow_rho_est = (med + med.conj().T) / 2
            vals, vecs = eigh(shadow_rho_est)
            vals = np.maximum(vals, 0)
            if vals.sum() == 0:
                shadow_rho = vecs[:, -1:] @ vecs[:, -1:].conj().T
            else:
                shadow_rho = (vecs @ np.diag(vals) @ vecs.conj().T) / vals.sum()
            # DFE estimate
            F_dfe = DFE_estimate(rho_true, total_shots, rng)
            # compute fidelities with psi_minus projector and trace distances to true rho
            F_tom = np.real(np.trace(sqrtm(sqrtm(proj_psi_minus) @ rho_tom @ sqrtm(proj_psi_minus))))**2 if True else 0
            # But simpler for pure target: F = Tr(rho_tom * proj_psi_minus)
            F_tom = np.real(np.trace(rho_tom @ proj_psi_minus))
            F_sh = np.real(np.trace(shadow_rho @ proj_psi_minus))
            tomog_errs.append(1-F_tom); sh_errs.append(1-F_sh); dfe_errs.append(1-F_dfe)
            tomog_trace.append(0.5 * np.sum(np.sqrt(np.real(np.linalg.eigvals((rho_tom - rho_true).conj().T @ (rho_tom - rho_true))))))
            sh_trace.append(0.5 * np.sum(np.sqrt(np.real(np.linalg.eigvals((shadow_rho - rho_true).conj().T @ (shadow_rho - rho_true))))))
            # For DFE we only have fidelity estimate, approximate trace by abs(F_dfe - Tr(rho_true proj))
            dfe_trace.append(abs(F_dfe - np.real(np.trace(rho_true @ proj_psi_minus))))
        results['shots'].append(total_shots)
        results['tomog_1F_mean'].append(np.mean(tomog_errs)); results['tomog_1F_std'].append(np.std(tomog_errs))
        results['shadows_1F_mean'].append(np.mean(sh_errs)); results['shadows_1F_std'].append(np.std(sh_errs))
        results['dfe_1F_mean'].append(np.mean(dfe_errs)); results['dfe_1F_std'].append(np.std(dfe_errs))
        results['tomog_trace'].append(np.mean(tomog_trace)); results['shadows_trace'].append(np.mean(sh_trace)); results['dfe_trace'].append(np.mean(dfe_trace))
    return results

if __name__ == '__main__':
    shot_budgets = [100, 300, 1000, 3000, 10000]
    results = run_experiment(shot_budgets, reps=30)
    print('Shot budgets:', results['shots'])
    for i,s in enumerate(results['shots']):
        print(f'shots={s}: Tomog 1-F = {results["tomog_1F_mean"][i]:.4f} ± {results["tomog_1F_std"][i]:.4f}, '
              f'Shadows 1-F = {results["shadows_1F_mean"][i]:.4f} ± {results["shadows_1F_std"][i]:.4f}, '
              f'DFE 1-F = {results["dfe_1F_mean"][i]:.4f} ± {results["dfe_1F_std"][i]:.4f}')
    # basic plots
    plt.errorbar(results['shots'], results['tomog_1F_mean'], yerr=results['tomog_1F_std'], label='Tomography 1-F')
    plt.errorbar(results['shots'], results['shadows_1F_mean'], yerr=results['shadows_1F_std'], label='Shadows 1-F')
    plt.errorbar(results['shots'], results['dfe_1F_mean'], yerr=results['dfe_1F_std'], label='DFE 1-F')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Total shots'); plt.ylabel('1 - Fidelity with |ψ^-⟩ (log-log)')
    plt.legend(); plt.grid(True); plt.title('Tomography vs Shadows vs DFE')
    plt.show()

