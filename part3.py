#%%
import torch
import torch.distributions as td

import scipy.stats as stats

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
    while not torch.isclose(women[:,-1].sum(), torch.tensor(1_000,dtype=torch.long), atol=1e-8):
        women, times = next_state(women,times,Q=Q)
        states = women.nonzero(as_tuple=True)[1]
        time_enter_state[all_w_idx,states] = times

    return time_enter_state


def create_time_series(time_enter_state, n=n):
    """
    Given a time_enter_state tensor, compute time-series.
    """
    tvals,svals = torch.sort(time_enter_state,dim=1) # time-values, state-values
    tmin, tmax = 0, tvals.max()
    mask = ~(tvals == -torch.inf) # mask to remove non-existant states
    smask = mask.long() * torch.arange(n).unsqueeze(1) # subject mask

    tvals,svals,smask = tvals[mask], svals[mask], smask[mask]

    # compute steps/disc. points in time-series [X(0),X(48),X(96),...]
    num_ts_points = torch.round(tmax / obs_freq).long()
    ts_points = torch.linspace(tmin, (num_ts_points+1)*obs_freq, num_ts_points)

    time_series = torch.zeros(n,num_ts_points)

    for i in range(n):
        ts_idx = smask == i
        tval,sval = tvals[ts_idx],svals[ts_idx]

        k = 0
        for j,ts_p in enumerate(ts_points):
            if k+1 < tval.shape[0] and not ts_p < tval[k+1]:
                k += 1
            time_series[i,j] = sval[k]
            if sval[k] == 4:
                time_series[i,j:] = 4
                break
    
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
# From time_enter_state compute N and S
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


compute_q(time_enter_state)

#%%
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

def find_next_state_change(current_state, timeseries):
    next_state = list(filter(lambda x: x > current_state, timeseries.tolist()))[0]
    return timeseries.tolist().index(next_state)

def simulate_to_replicate_time_series(time_series, q_matrix):
    new_time_series = []
    for i in range(time_series.shape[0]):
        all_w_idx = torch.arange(1)
        current_woman = init_women(1)
        time_enter_state = torch.zeros_like(current_woman,dtype=torch.double)
        time_enter_state[:,1:] = -torch.inf
        current_times = None

        next_index = find_next_state_change(0, time_series[i])
        next_time = next_index*48
        counter = 0


        while(True):
            # print(current_woman)
            # print(current_times)
            # print(time_enter_state)

            current_woman_before = current_woman.clone()
            current_times_before = None if current_times == None else current_times.clone()
            time_enter_state_before = time_enter_state.clone()


            current_woman, current_times = next_state(current_woman,current_times,Q=q_matrix)
            states = current_woman.nonzero(as_tuple=True)[1]
            time_enter_state[all_w_idx,states] = current_times
            

            next_time_sim = current_times[0].item()
            state_updated_to = current_woman[0].tolist().index(1)

            # print(next_time_sim)
            if next_time_sim < next_time and next_time_sim > next_time - 48 and state_updated_to == int(time_series[i][next_index].item()):
                # print("Success")
                # print(time_enter_state)
                # print(counter)
                sim_time_series = create_time_series(time_enter_state, n=1)[0]
                # print(sim_time_series)
                # print(state_updated_to)
                
                if state_updated_to == 4:
                    # print(time_enter_state)
                    break

                next_index = find_next_state_change(state_updated_to, time_series[i])
                next_time = next_index*48

                
            else:
                current_woman = current_woman_before.clone()
                current_times = None if current_times_before == None else current_times_before.clone()
                time_enter_state = time_enter_state_before.clone()

            # sim_time_series = create_time_series(time_enter_state, n=1)[0]
            

            

            counter += 1
        # print(next_time)
        # print(next_s)
        # print("FOUND ONE")
        new_time_series.append(time_enter_state[0].tolist())
    return torch.tensor(new_time_series)


def initate_q_matrix():
    q_matrix = torch.zeros((5,5))
    for i in range(4):
        for o in range(5):
            if o >= i:
                q_matrix[i,o] = 1
    for i in range(4):
        q_matrix[i,i] = -q_matrix[i].sum()
    return q_matrix


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

def approximate_q(time_series_):
    approx_time_enter = []
    for i in range(time_series_.shape[0]):
        approx_time_enter.append(time_series_to_enter_state(time_series_[i]))
    return compute_q(torch.stack(approx_time_enter))


def optimize(time_series_input):
    q_matrix = approximate_q(time_series_input)
    while(True):
        new_time_enter_state = simulate_to_replicate_time_series(time_series_input, q_matrix)
        print("Completed simulation!")
        new_q = compute_q(new_time_enter_state)
        if (q_matrix - new_q).norm() < 0.001:
            return new_q
        q_matrix = new_q

# time_series_to_enter_state(time_series[0])
optimize(time_series)






# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

#%%
def init_Q():
    Q0 = torch.zeros(5,5)
    triu_idx = torch.triu_indices(5,5,offset=1)
    diag_idx = torch.repeat_interleave(torch.arange(5).unsqueeze(0),2,dim=0)
    Q0[*triu_idx] = td.HalfNormal(scale=torch.ones(torch.arange(5).sum())).sample()
    Q0[*diag_idx] = -Q0.sum(dim=1)
    return Q0


for i,ts in enumerate(time_series):
    ...

i = 0
ts = time_series[i]

def simulate_for_t(t, n, women=None, time_enter_state=None, Q=Q):
    assert (women is not None and time_enter_state is not None) \
        or (women is None and time_enter_state is None), "Inconsistent {women,time_enter_state} pair provided - provide both or neither."
    all_w_idx = torch.arange(n)
    if women is None:
        women = init_women(n, s0=0)
        time_enter_state = torch.zeros_like(women,dtype=torch.double)
        time_enter_state[:,1:] = -torch.inf
    times = None
    while True:
        women, times = next_state(women, time_enter_state, Q=Q)
        states = women.nonzero(as_tuple=True)[1]
        time_enter_state[all_w_idx,states] = times
        dead_mask = (states == 4)
        women[dead_mask] = init_women(1)
        time_enter_state[dead_mask,1:] = -torch.inf

    return women, time_enter_state

simulate_for_t(48, n=10)

