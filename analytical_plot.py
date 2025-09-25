import numpy as np
import matplotlib.pyplot as plt

# ---------- functions (same as before) -----------------------------------
def sigma2(gamma):
    return (1 - np.exp(-2*gamma)) / 2.0

def expA(N, gamma, delta=0.0):
    sig2 = sigma2(gamma)
    return N**2 - sig2**2*(2*N - 1) - (N - 1)**2*np.exp(-2*gamma) - (2*N - 1)*np.exp(-gamma)*np.cos(delta)+1


def expA22(N, gamma, delta=0.0):
    sigma = sigma2(gamma)
    return     np.exp(-4*gamma)*((N**4 - 4*N**3*sigma**2 + 8*N**2*sigma**4 + 4*N**2*sigma**2 - 16*N*sigma**4 + 4*N*sigma**2 + 8*sigma**4 - 4*sigma**2)*np.exp(4*gamma)\
  + (-2*N**4 + 8*N**3*sigma**2 + 4*N**3 - 24*N**2*sigma**2 + 2*N**2 \
     + 24*N*sigma**2 - 8*N - 8*sigma**2 + 4)*np.exp(2*gamma)  + (N**4 - 4*N**3 + 6*N**2 - 4*N + 1) \
  - 4*(N-1)*np.exp(gamma)*( N**2*(np.exp(2*gamma) - 1) + 2*N - 1 + 4*sigma**2*np.exp(2*gamma)*(1 - N))*np.cos(delta) + 4*(N-1)**2*np.exp(2*gamma)*np.cos(2*delta))


def f2_value(N, gamma, beta_tilde=1.0, delta=0.0):
    sig2 = sigma2(gamma)
    return beta_tilde * N * (-2*np.exp(-gamma) / expA(N, gamma, delta) +
                             4*sig2 / expA22(N, gamma, delta) +4*N*np.exp(-2*gamma)*np.sin(delta)**2)

def f_value(N, gamma, beta_tilde=1.0, delta=0.0):
    sig2 = sigma2(gamma)
    return beta_tilde * (N-1) * (-2*np.exp(-gamma) / expA(N, gamma, delta) +
                             4*sig2 / expA(N, gamma, delta)**2 +4*N*np.exp(-2*gamma)*np.sin(delta)**2)

# ---------- parameters ---------------------------------------------------
gammas = [0.5,1,2]
N_vals = np.arange(2, 10, 1)

# ---------- plot ---------------------------------------------------------
plt.figure(figsize=(5, 3))
for g in gammas:
    y_neg = f_value(N_vals, g,delta=0)  # negative of previous function
    plt.plot(N_vals, y_neg, label=f"γ = {g}",linestyle='--',marker='o',markersize=5)

plt.xlabel("N")
plt.ylabel(r"$f(N,\gamma)$")
#plt.yscale('log')
plt.title("Negative of the expression vs N for different γ (Δ = 0, β̃ = 1)")
#plt.yticks(np.arange(-5,10,5),fontsize=17)
plt.xticks(np.arange(2,10,3),fontsize=17)
plt.tick_params(width=3,length=5,)
plt.legend()
plt.savefig("G:/My Drive/PhD studies/PhD Project/optimalSensing_LEUP/researchPaperFigures/analytical.svg", format="svg", bbox_inches='tight')

plt.show()
