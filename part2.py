#%%
import torch, scipy
import torch.distributions as td
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

torch.set_default_dtype(torch.double)
# %%
Q = torch.tensor([
    [-0.0085, 0.005, 0.0025, 0, 0.001],
    [0, -0.014, 0.005, 0.004, 0.005],
    [0, 0, -0.008, 0.003, 0.005],
    [0, 0, 0, -0.009, 0.009],
    [0, 0, 0, 0, 0],
], dtype=torch.double)
# %%

m = Q.shape[0] # number of states

def init_women(n, s0=0):
    """
    Init n women in state s0.
    """
    women = torch.zeros(n,m,dtype=torch.long)
    if s0 is not None:
        women[:,s0] = 1
    return women


def next_state(women, times=None, Q=Q):
    """
    Simulates a single step for a batch of subjects (women),
        according to a CTMC specified by transition rate matrix Q.
    """
    states = women.nonzero(as_tuple=True)[1]
    dead_mask = states == 4
    states = states[~dead_mask]

    stay_rates = - Q[states,states]
    if times is None:
        times = torch.zeros(women.shape[0],dtype=torch.double)
    stay_times = td.Exponential(stay_rates).sample()
    times[~dead_mask] += stay_times

    jump_dist = - Q[states] / Q[states,states].unsqueeze(1)
    jump_dist[torch.arange(len(states)), states] = 0

    new_state = torch.multinomial(jump_dist, 1)
    women[~dead_mask] = women[~dead_mask].fill_(0)
    women[~dead_mask] = women[~dead_mask].scatter_(1,new_state,1)

    return women, times


#%%
# ! TASK 7
# n = 100_000
n = 1_000

def simulate_until_death(n, Q=Q):
    """
    Simulate n subjects in a CTMC described by transition rate matrix Q,
        until all subjects have reached state 5 (death state).
    Params:
        n: int - number of subjects (women) to simulate.
        Q: torch.Tensor [shape (m x m)] - transition matrix for CTMC.
    Returns: 
        time_enter_state: torch.Tensor [shape (n x m)] - timestamps where 
            each subject has entered each state. 
            If state is not entered, time is -inf.
    """
    all_w_idx = torch.arange(n)
    women = init_women(n, s0=0)
    time_enter_state = torch.zeros_like(women,dtype=torch.double)
    time_enter_state[:,1:] = -torch.inf
    times = None
    while not torch.isclose(women[:,-1].sum(), torch.tensor(1_000,dtype=torch.long), atol=1e-8):
        women, times = next_state(women,times,Q=Q)
        states = women.nonzero(as_tuple=True)[1]
        time_enter_state[all_w_idx,states] = times

    return time_enter_state

time_enter_state = simulate_until_death(n=1_000)

#%%
fig,ax = plt.subplots(1,1,figsize=(15,6))
ax.hist(time_enter_state[:,-1], bins=50, color='dodgerblue')
ax.set_title("Simulated lifetime distribution")
ax.set_xlabel("$t$")
ax.set_ylabel("Frequency")

plt.tight_layout()
# plt.show()
plt.savefig("img/part2/lifetime_distribution.pdf")

# %%
# ? times where it reappeared distantly
s3_times = time_enter_state[:,2]
# ? 0 <= times <= 30.5 is where it reappeared distantly in the specified time frame
num_reappeared = ((0 <= s3_times) * (s3_times <= 30.5)).sum()

# %%
# ? fraction where it reappeared distantly within 30.5 months
num_reappeared / n
# %%
# ! TASK 8

Qs = Q[:-1,:-1]
p0 = torch.tensor([1,0,0,0], dtype=torch.double)
ones = torch.ones(4,1)

F = lambda t: (1 - p0@torch.linalg.matrix_exp(Qs*t)@ones).item()

sorted_sim_ts = torch.sort(time_enter_state[:,-1])[0]
ratio_dead = (1 - torch.linspace(1,0,1001))[1:]
aCDF = torch.tensor([F(t) for t in sorted_sim_ts])

simPMF = torch.histc(sorted_sim_ts, bins=n) / n
simCDF = simPMF.cumsum(dim=0)

fig,ax = plt.subplots(1,1,figsize=(15,6))

# ax.plot(sorted_sim_ts, ratio_dead_linspaced, label='Simulated CDF (linspaced ratio)')
ax.plot(torch.linspace(0,sorted_sim_ts[-1],n), simCDF, label='Simulated CDF', color='orange', linewidth=3)
ax.plot(sorted_sim_ts, aCDF, label='Analytical CDF', color='blue', linewidth=1.5, linestyle='--')

ax.set_xlabel("$t$")
ax.legend()
ax.set_title(f"CDF comparison using {n} simulated women")
# plt.show()

plt.tight_layout()
plt.savefig("img/part2/cdf_comp.pdf")
#%%
mean_lifetime = torch.mean(sorted_sim_ts)
std_lifetime = torch.std(sorted_sim_ts) 
n = len(sorted_sim_ts)


confidence_level = 0.95

z_score = scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)
mean_conf_interval = (mean_lifetime - z_score * (std_lifetime / torch.sqrt(torch.tensor(n, dtype=torch.float32))),
                      mean_lifetime + z_score * (std_lifetime / torch.sqrt(torch.tensor(n, dtype=torch.float32))))

# Confidence interval for the standard deviation
alpha = 1 - confidence_level
chi2_lower = scipy.stats.chi2.ppf(alpha / 2, n - 1)
chi2_upper = scipy.stats.chi2.ppf(1 - alpha / 2, n - 1)
std_conf_interval = (torch.sqrt((n - 1) * std_lifetime**2 / chi2_upper),
                     torch.sqrt((n - 1) * std_lifetime**2 / chi2_lower))

# Output results
print(f"Mean Lifetime: {mean_lifetime.item():.2f}")
print(f"Mean Confidence Interval: ({mean_conf_interval[0].item():.2f}, {mean_conf_interval[1].item():.2f})")
print(f"Standard Deviation: {std_lifetime.item():.2f}")
print(f"Standard Deviation Confidence Interval: ({std_conf_interval[0].item():.2f}, {std_conf_interval[1].item():.2f})")


from scipy.stats import chisquare


observed_freq = torch.histc(simCDF, bins=20)
expected_freq = torch.histc(aCDF, bins=20)


test_stat,p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)
print(f"Test stat = {test_stat:.5f}")
print(f"p-value = {p_value:.5f}")

# ! p-value not quite matching?

#%%
# ! TASK 9

# naïve computation of diagonal
q11 = -sum([0.0025, 0.00125, 0, 0.001])
q22 = -sum([0, 0, 0.002, 0.005])
q33 = -sum([0, 0, 0.003, 0.005])
q44 = -sum([0, 0, 0, 0.009])

Q2 = torch.tensor([
    [q11, 0.0025, 0.00125, 0, 0.001],
    [0, q22, 0, 0.002, 0.005],
    [0, 0, q33, 0.003, 0.005],
    [0, 0, 0, q44, 0.009],
    [0, 0, 0, 0, 0],
])


time_enter_state2 = simulate_until_death(n=1_000, Q=Q2)

#%%

sorted_sim_ts2 = torch.sort(time_enter_state2[:,-1])[0]
simPMF2 = torch.histc(sorted_sim_ts2, bins=n) / n
simCDF2 = simPMF2.cumsum(dim=0)

# Kaplan-Meier estimators
ratio_alive = 1 - simCDF
ratio_alive2 = 1 - simCDF2

ts = torch.linspace(0, sorted_sim_ts[-1], n)
ts2 = torch.linspace(0, sorted_sim_ts2[-1], n)

#%%
plt.rc('font',        size=20)          # controls default text sizes
plt.rc('axes',   titlesize=28)     # fontsize of the axes title
plt.rc('axes',   labelsize=25)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=20)    # fontsize of the tick labels
plt.rc('ytick',  labelsize=20)    # fontsize of the tick labels
plt.rc('legend',  fontsize=25)    # legend fontsize
plt.rc('figure', titlesize=25)   # fontsize of the figure title

fig,ax = plt.subplots(1,1,figsize=(15,6))

ax.plot(ts, ratio_alive,   label='[w/o treatment] Sim. S(t)', linewidth=2, color='red')
ax.plot(ts2, ratio_alive2, label='[w/ treatment] Sim. S(t)' , linewidth=3, linestyle=':', color='green')

ax.set_xlabel("$t$")
ax.legend()
ax.set_title("$\hat{S}(t)$ comparison using"+ f" {n} simulated women")

plt.tight_layout()
# plt.show()
plt.savefig("img/part2/treatmeant_survival_comp.pdf")

#%%
# ! TASK 10 
# optional
...
