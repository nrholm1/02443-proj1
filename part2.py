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


def next_state(women, times=None):
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
    # jump_dist[torch.isnan(jump_dist)] = 0

    new_state = torch.multinomial(jump_dist, 1)
    women[~dead_mask] = women[~dead_mask].fill_(0)
    women[~dead_mask] = women[~dead_mask].scatter_(1,new_state,1)

    return women, times


#%%
all_w_idx = torch.arange(1_000)
women = init_women(1_000, s0=0)
freqs = [women.sum(dim=0)]
time_enter_state = torch.zeros_like(women,dtype=torch.double)
time_enter_state[:,1:] = -torch.inf
times = None
while not torch.isclose(women[:,:-1].sum(), torch.zeros(1,dtype=torch.long), atol=1e-8):
    women, times = next_state(women,times)
    states = women.nonzero(as_tuple=True)[1]
    # dead_mask = states == 4
    time_enter_state[all_w_idx,states] = times
    freqs.append(women.sum(dim=0))

# %%
# ? times where it reappeared distantly
s3_times = time_enter_state[:,2]
# ? 0 <= times <= 30.5 is where it reappeared distantly in the specified time frame
num_reappeared = ((0 <= s3_times) * (s3_times <= 30.5)).sum()

# %%
# ? fraction where it reappeared distantly within 30.5 months
num_reappeared / 1_000
# %%

# %%
# ! TASK 7

Qs = Q[:-1,:-1]
p0 = torch.tensor([1,0,0,0], dtype=torch.double)
ones = torch.ones(4,1)

F = lambda t: (1 - p0@torch.linalg.matrix_exp(Qs*t)@ones).item()

# T_max = time_enter_state.max()
# ts = torch.linspace(0, T_max, 1_000)

sorted_sim_ts = torch.sort(time_enter_state[:,-1])[0]
ratio_dead = (1 - torch.linspace(1,0,1001))[1:]
aCDF = [F(t) for t in sorted_sim_ts]

plt.plot(sorted_sim_ts, aCDF, label='Analytical CDF')
plt.plot(sorted_sim_ts, ratio_dead, label='Simulated Lifetime distribution')

plt.legend()
plt.show()
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

#%%

# ! how to make the CDFs match?
torch.histc(ratio_dead, bins=20)

#%%

chisquare(f_obs=ratio_dead, f_exp=aCDF)
# %%
