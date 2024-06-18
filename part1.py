#%%
import torch
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')
# %%
P = torch.tensor([
    [0.9915, 0.005, 0.0025, 0, 0.001],
    [0, 0.986, 0.005, 0.004, 0.005],
    [0, 0, 0.992, 0.003, 0.005],
    [0, 0, 0, 0.991, 0.009],
    [0, 0, 0, 0, 1],
]).double()

# %%
# ! TASK 1

m = P.shape[0] # number of states

def init_women(n, s0=None):
    """
    Init n women in state s0.
    """
    women = torch.zeros(n,m,dtype=torch.long)
    if s0 is not None:
        women[:,s0] = 1
    return women

def next_state(women):
    states = women.nonzero(as_tuple=True)[1]
    dist = P[states]
    new_state = torch.multinomial(dist, 1)
    women.fill_(0)
    women.scatter_(1,new_state,1)
    return women

def analytical_dist(dt):
    return P[0] @ torch.linalg.matrix_power(P,dt)

women = init_women(1_000, s0=0)
freqs = [women.sum(dim=0)]
state2_at_some_point = torch.zeros(1_000).bool()
while not torch.isclose(women[:,:-1].sum(), torch.zeros(1,dtype=torch.long), atol=1e-8):
    women = next_state(women)
    state2_at_some_point += women.nonzero()[:,1] == 1
    freqs.append(women.sum(dim=0))

#%%
fig,ax = plt.subplots(1,2,figsize=(15,6))
afreqs = [analytical_dist(t) for t in range(len(freqs)+1)]

ax[0].set_title("Analytical distribution")
ax[1].set_title("Simulated distribution")

ax[0].plot(torch.vstack(afreqs), label=[f"State {i}" for i in torch.arange(1,6)])
ax[1].plot(torch.vstack(freqs), label=[f"State {i}" for i in torch.arange(1,6)])

ax[0].set_ylabel('Probability')
ax[1].set_ylabel('Frequency')

for axi in ax:
    axi.set_xlabel("Step")
    axi.legend()

plt.show()

#%%
# ? Assuming state 2 corresponds to "local recurrence"
state2_at_some_point.sum() / 1_000


#%%
# ! TASK 2
adist = analytical_dist(dt=120)

fig,ax = plt.subplots(1,2,figsize=(15,6))
ax[0].set_title("Analytical distribution")
ax[1].set_title("Simulated distribution")

ax[0].bar(torch.arange(1,6), adist, color='red')
ax[1].bar(torch.arange(1,6), freqs[119], color='blue')

ax[0].set_ylabel('Probability')
ax[1].set_ylabel('Frequency')
for axi in ax:
    axi.set_xlabel("State")

plt.show()
# %%
from scipy.stats import chisquare

test_stat, p_value = chisquare(f_obs=freqs[119].double() / freqs[119].sum(), f_exp=adist / adist.sum())

print(f"Test stat = {test_stat:.5f}")
print(f"p-value = {p_value:.5f}")

# %%
# ! TASK 3
pi = P[0,:-1] / P[0,:-1].sum() # normalize to obtain distribuion
Ps = P[:-1,:-1]
ps = P[:-1,-1]
ones = torch.ones(4,1,dtype=torch.double)
I = torch.eye(4)

E_T = (pi@torch.linalg.solve(I - Ps, ones)).item()
P_T = lambda t: pi@torch.linalg.matrix_power(Ps,t)@ps


freqs = torch.vstack(freqs).double()
prob_died_at_t = torch.diff(1_000 - freqs[:,:-1].sum(dim=1)) / 1_000

T_max = len(prob_died_at_t)
ts = torch.arange(T_max)

fig,ax = plt.subplots(1,1,figsize=(15,6))
ax.plot(ts, prob_died_at_t, linewidth=.3, label='Simulated fraction T=t')
ax.plot(ts, [P_T(t) for t in torch.arange(T_max)], label='Analytical P(T=t)')
ax.set_xlabel("$t$")
ax.legend()
plt.show()
#%%
simCDF = prob_died_at_t.cumsum(dim=0)
aCDF = torch.tensor([P_T(t) for t in torch.arange(T_max)]).cumsum(dim=0)

print(f"Test stat = {test_stat:.5f}")
print(f"p-value = {p_value:.5f}")
test_stat, p_value = chisquare(f_obs=simCDF/simCDF.sum(), f_exp=aCDF/aCDF.sum())

fig,ax = plt.subplots(1,1,figsize=(15,6))
ax.plot(ts, simCDF, linewidth=1, label='Simulated CDF')
ax.plot(ts, aCDF, label='Analytical CDF')
ax.set_xlabel("$t$")
ax.legend()
ax.set_title(f"Comparison of simulated and analytical CDFs, $p$-value={p_value:.4f}")
plt.show()
#%%
# ! TASK 4
"""
Estimate the expected lifetime, after surgery, of a woman who survives the
first 12 months following surgery, but whose breast cancer has also reap-
peared within the first 12 months, either locally or distant.
"""
sample = torch.zeros(1_000, 5)
num_accepted = 0
while sample.sum() < 1_000:
    candidate_sample = init_women(n=1_000, s0=0)
    for _ in range(12):
        candidate_sample = next_state(candidate_sample)
        # discard "dead"
        dead_mask = candidate_sample.nonzero()[:,1] == 4
        candidate_sample = candidate_sample[~dead_mask]
    # discard state 1
    healthy_mask = candidate_sample.nonzero()[:,1] == 0
    candidate_sample = candidate_sample[~healthy_mask]

    accepted_samples = min(candidate_sample.shape[0], 1_000-num_accepted)
    sample[num_accepted:num_accepted+min(candidate_sample.shape[0], 1_000)] = candidate_sample[:accepted_samples]
    num_accepted += accepted_samples 

#%%
plt.bar(torch.arange(1,6),sample.sum(dim=0))
plt.title("Distribution of 1,000 surviving, but metastising women")
plt.xlabel("State")
plt.ylabel("Frequency")
plt.show()
# %%
women = sample.clone().long()
freqs = [women.sum(dim=0)]
while not torch.isclose(women[:,:-1].sum(), torch.zeros(1,dtype=torch.long), atol=1e-8):
    women = next_state(women)
    freqs.append(women.sum(dim=0))

freqs = torch.vstack(freqs).double()
# %%
prob_died_at_t = torch.diff(1_000 - freqs[:,:-1].sum(dim=1)) / 1_000

T_max = len(prob_died_at_t)
ts = torch.arange(T_max)

# ? expected living time
print("Excepted living time for women that have survived 12 months, but metastised in the mean time.")
print(f"E[T] = {12 + (ts * prob_died_at_t).sum().item():.2f}")

# %%
# ! TASK 5

def task5_sim():
    women = init_women(n=200, s0=0)
    freqs = [women.sum(dim=0)]
    for _ in range(350):
        women = next_state(women)
        freqs.append(women.sum(dim=0))
    freqs = torch.vstack(freqs)
    prob_died_at_t = torch.diff(1_000 - freqs[:,:-1].sum(dim=1)) / 1_000
    dead_fraction = prob_died_at_t.cumsum(dim=0)[-1]

    ts = torch.arange(len(prob_died_at_t))
    E_T = (ts * prob_died_at_t).sum().item()

    return dead_fraction, E_T

#%%
dead_fractions = torch.zeros(100)
expected_lifetimes = torch.zeros(100)
for sim_no in range(100):
    dead_fraction, E_T = task5_sim()
    dead_fractions[sim_no] = dead_fraction
    expected_lifetimes[sim_no] = E_T

# %%
# ? crude MC estimator
dead_fractions.mean(), dead_fractions.var()
# %%
mu_Y = expected_lifetimes.mean()
Z = lambda c: dead_fractions + c*(expected_lifetimes - mu_Y)
emp_cov = lambda X,Y: (X*Y).mean() - X.mean()*Y.mean() # empirical covariance estimate
c = - emp_cov(dead_fractions,expected_lifetimes)/expected_lifetimes.var()

cv_estimator = Z(c)
#%%
# ? control variate MC estimator
cv_estimator.mean(), cv_estimator.var()

#%%
# ? reduction in mean
1 - cv_estimator.var() / dead_fractions.var()