#%%
import torch
import torch.distributions as td
import scipy.stats as stats

from tqdm import tqdm
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

obs_freq = 48  # doctor's apt. every 4 years = 48 months
n = 1_000      # number of women to simulate
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
    while not torch.isclose(women[:,-1].sum(), torch.tensor(n,dtype=torch.long), atol=1e-8):
        women, times = next_state(women,times,Q=Q)
        states = women.nonzero(as_tuple=True)[1]
        time_enter_state[all_w_idx,states] = times

    return time_enter_state


def create_time_series(time_enter_state, obs_freq=obs_freq):
    """
    Given a time_enter_state tensor, compute time-series.
    """
    n = time_enter_state.shape[0]
    tvals, svals = torch.sort(time_enter_state, dim=1)  # time-values, state-values
    tmax = tvals.max()
    mask = ~(tvals == -torch.inf)  # mask to remove non-existent states

    # Compute steps/disc. points in time-series [X(0), X(48), X(96), ...]
    num_ts_points = torch.ceil(tmax / obs_freq).long()
    ts_points = torch.linspace(0, num_ts_points * obs_freq, num_ts_points + 1)

    # Prepare result tensor
    time_series = torch.zeros((n, num_ts_points + 1), dtype=svals.dtype)

    if mask.sum() == 0:
        return time_series

    for i in range(n):
        valid_indices = mask[i]
        tval, sval = tvals[i, valid_indices], svals[i, valid_indices]
        
        if len(tval) == 0:
            continue

        j, k = 0, 0
        while j < len(ts_points) and k < len(tval):
            if ts_points[j] < tval[k]:
                time_series[i, j] = sval[k - 1] if k > 0 else sval[k]
            else:
                time_series[i, j] = sval[k]
                k += 1
            if sval[k - 1] == 4:
                time_series[i, j:] = 4
                break
            j += 1

    return time_series


#%%
time_enter_state = simulate_until_death(n=n)
time_series = create_time_series(time_enter_state)
# %%
# q = n+1
q = 3

plt.rc('font',        size=20)          # controls default text sizes
plt.rc('axes',   titlesize=28)     # fontsize of the axes title
plt.rc('axes',   labelsize=25)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=13)    # fontsize of the tick labels
plt.rc('ytick',  labelsize=17)    # fontsize of the tick labels
plt.rc('legend',  fontsize=25)    # legend fontsize
plt.rc('figure', titlesize=25)   # fontsize of the figure title


fig,ax = plt.subplots(1,1,figsize=(15,6))
ax.step(torch.arange(time_series.shape[1]), time_series[:q].t(), linewidth=3)
ax.set_yticks([i for i in range(5)],[i for i in range(1,6)])
xticks = torch.arange(time_series.shape[1])
ax.set_xticks(xticks, [(xt*obs_freq).long().item() for xt in xticks])
ax.set_xlabel("$t$")
ax.set_ylabel("State")

ax.set_title(f"Time series of the state of (subset of) {q} women")

plt.tight_layout()
# plt.show()
plt.savefig("img/part3/timeseries_example.pdf")

#%%
def compute_q(time_enter):
    sojourn_time = torch.zeros(5)
    n_matrx = torch.zeros((5, 5))
    for i in range(4):
        for o in range(time_enter.shape[0]):
            elements = time_enter[o].tolist()
            if elements[i] == -torch.inf:
                continue
            successor = min(filter(lambda x: x > elements[i], elements))
            sojourn_time[i] +=  successor - elements[i]

            jump_to_index = elements.index(successor)
            n_matrx[i,jump_to_index] += 1

    q_matrix = torch.zeros((5, 5))
    for i in range(4):
        q_matrix[i] = n_matrx[i] / sojourn_time[i]
    for i in range(4):
        q_matrix[i,i] = -q_matrix[i].sum()
    return q_matrix


def find_next_state_change(current_state, timeseries):
    next_state = list(filter(lambda x: x > current_state, timeseries.tolist()))[0]
    return timeseries.tolist().index(next_state)


def time_series_to_enter_state(time_s):
    current_state = 0
    updated_time_series = torch.ones(5) - torch.inf
    updated_time_series[0] = 0.0

    while(current_state != 4):
        next_id   = find_next_state_change(current_state, time_s)
        next_state = int(time_s[next_id].item())
        updated_time_series[next_state] = next_id*48
        current_state = next_state
    return updated_time_series


def init_Q_heuristic(time_series_):
    approx_time_enter = []
    for i in range(time_series_.shape[0]):
        approx_time_enter.append(time_series_to_enter_state(time_series_[i]))
    return compute_q(torch.stack(approx_time_enter))


def init_Q_random(scale=1e-3):
    Q0 = torch.zeros(5,5)
    triu_idx = torch.triu_indices(5,5,offset=1)
    diag_idx = torch.repeat_interleave(torch.arange(5).unsqueeze(0), 2, dim=0)
    Q0[*triu_idx] = td.HalfNormal(scale=scale*torch.ones(torch.arange(5).sum())).sample() + scale
    Q0[*diag_idx] = -Q0.sum(dim=1)
    return Q0


def simulate_for_t(t, n, women=None, time_enter_state=None, Q=Q):
    assert (women is not None and time_enter_state is not None) \
        or (women is None and time_enter_state is None), "Inconsistent {women,time_enter_state} pair provided - provide both or neither."
    all_w_idx = torch.arange(n)
    
    times = None if time_enter_state is None else time_enter_state.max(dim=1)[0]
    if women is None:
        women = init_women(n, s0=0)
        time_enter_state = torch.zeros_like(women,dtype=torch.double)
        time_enter_state[:,1:] = -torch.inf
    
    dead_mask = torch.zeros_like(all_w_idx).bool()
    time_mask = time_enter_state.max(dim=1)[0] < t
    mask = (~dead_mask)*time_mask
    while torch.any(mask):
        if times is not None: women[mask],times[mask] = next_state(women[mask],times[mask], Q=Q)
        else: women,times = next_state(women, times, Q=Q)
        states = women.nonzero(as_tuple=True)[1]
        time_enter_state[all_w_idx,states] = times
        dead_mask = states == 4
        time_mask = time_enter_state.max(dim=1)[0] < t
        mask = (~dead_mask*time_mask)
        
    return women, time_enter_state


#%%
@torch.no_grad()
def simulate_trajectories(time_series, obs_freq, Q=Q, num_retries=100000):
    num_succesful = 0
    Nij = torch.zeros_like(Q)
    Si = torch.zeros(Q.shape[0])
    desc_str = "Computing trajectories"
    pbar = tqdm(time_series, desc=desc_str)
    for i, ts in enumerate(pbar):
        ts = ts[:torch.where(ts == 4)[0][0]+1]
        num_recreated = 0
        out_ts = torch.ones(len(ts), dtype=torch.long) * (-torch.inf)
        retry = 0
        while num_recreated < len(ts) and retry < num_retries:
            retry += 1
            if num_recreated > 0:
                w0 = init_women(1, s0=int(out_ts[num_recreated-1]))
                tes[0, int(out_ts[num_recreated-1])+1:] = -torch.inf
                w, tes = simulate_for_t(obs_freq*(num_recreated+1), n=1, women=w0, time_enter_state=tes.clone(), Q=Q)
            else:
                w, tes = simulate_for_t(obs_freq*(num_recreated+1), n=1, Q=Q)
            sample_ts = create_time_series(tes).squeeze()[num_recreated:]

            max_comp_idx = min(len(ts) - num_recreated, len(sample_ts))
            slice_idx = torch.nonzero(~(sample_ts[:max_comp_idx] == ts[num_recreated:num_recreated+max_comp_idx]))
            if slice_idx.numel() == 0:
                slice_idx = len(sample_ts)
            elif slice_idx[0] == 0:
                continue
            else:
                slice_idx = slice_idx[0]
            if slice_idx > 0:
                tes = tes
            out_ts[num_recreated:num_recreated+slice_idx] = sample_ts[:slice_idx]
            num_recreated += slice_idx

            # handle padded time series case
            if out_ts[num_recreated-1] == 4:
                out_ts[num_recreated-1:] = 4
                num_recreated = len(out_ts)
        if retry != num_retries:
            vals,traj = torch.sort(tes)
            traj = traj[vals != -torch.inf]
            Nij[traj[:-1], traj[1:]] = Nij[traj[:-1], traj[1:]] + 1
            Si[traj[:-1]] += torch.diff(vals[vals != -torch.inf])
            num_succesful += 1
        pbar.set_description_str(f"{desc_str}, [<👍|👎>: <{num_succesful}|{i+1-num_succesful}>]")

    return Nij, Si


def expectation_maximization(Qinit=None, scale=1e-3, max_steps=10, num_retries=1_000):
    Qk = init_Q_random(scale=scale) if Qinit is None else Qinit
    step = 0
    while step < max_steps:
        step += 1
        Nij, Si = simulate_trajectories(time_series, obs_freq, Q=Qk, num_retries=num_retries)
        Qk_p_1 = torch.zeros_like(Qk)
        Qk_p_1[:-1] = Nij[:-1] / Si[:-1].unsqueeze(1)
        diag_idx = torch.repeat_interleave(torch.arange(5).unsqueeze(0),2,dim=0)
        Qk_p_1[*diag_idx] = -Qk_p_1.sum(dim=1)
        max_delta = torch.norm(Qk_p_1 - Qk, p=torch.inf)
        Qk = Qk_p_1.clone()
        print(f"||{'Q_{k+1}'} - Q_k||_inf = {max_delta.item():.4f}")
        if max_delta < 1e-3: # check for convergence
            break
    return Qk


# %%
# Qinit = init_Q_random(scale=3e-3)
Qinit = init_Q_heuristic(time_series)
Qhat = expectation_maximization(max_steps=5, Qinit=Qinit, num_retries=10_000)
# Qhat = expectation_maximization(max_steps=5, num_retries=10_000, scale=1e-2)
#%%
Qhat = expectation_maximization(max_steps=5, Qinit=Qhat, num_retries=10_000)