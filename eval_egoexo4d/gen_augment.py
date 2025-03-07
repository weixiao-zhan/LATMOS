import torch
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# embedding_model = 'xclip'
# embedding_type = 'text'
# embedding_dim = 512

# embedding_model = 'omnivore'
# embedding_type = 'video'
# embedding_dim = 1536

embedding_model='egovlpv2'
embedding_type = 'video'
embedding_dim = 2304

# Load training tensors
meta_train = torch.load(f'data_embedding/{embedding_model}_segment_metadata_train.pt',
                        map_location=torch.device('cpu'), weights_only=True)
embedding_train = torch.load(f'data_embedding/{embedding_model}_segment_{embedding_type}_embedding_train.pt',
                             map_location=torch.device('cpu'), weights_only=True)
# Load validation tensors
meta_val = torch.load(f'data_embedding/{embedding_model}_segment_metadata_val.pt',
                        map_location=torch.device('cpu'), weights_only=True)
embedding_val = torch.load(f'data_embedding/{embedding_model}_segment_{embedding_type}_embedding_val.pt',
                             map_location=torch.device('cpu'), weights_only=True)

# map step id to int
step_unique_ids = set()
for meta in [meta_train, meta_val]:
    for t in meta:
        step_unique_ids.update(e['step_unique_id'] for e in t['segments'])
step_id_to_int = {step_id: i for i, step_id in enumerate(sorted(step_unique_ids))}
int_to_step_id = {i: step_id for step_id, i in step_id_to_int.items()}

# sanity checks
def segment_sanity_check(meta, trace):
    false_count = 0
    for take_idx in range(len(meta)):
        if len(meta[take_idx]['segments']) != trace[take_idx].shape[0]:
            print(f"segments not match {meta_train[take_idx]['take_name']:30}",len(meta_train[take_idx]['segments']), trace[take_idx].shape[0])
            false_count += 1
    print(f'sanity check complete, {false_count} mismatch')
segment_sanity_check(meta_train, embedding_train)
segment_sanity_check(meta_val, embedding_val)

def plot_trace_length_distribution(meta):
    # Plot histogram with 1 bin per virtual bar
    fig, ax1 = plt.subplots()
    trace_lengths = [len(e['segments']) for e in meta]
    ax1.hist(trace_lengths, bins=max(trace_lengths), cumulative=False)
    ax1.set_title('Trace Length Distribution')
    ax1.set_xlabel('Trace Length')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(1, max(trace_lengths))
    # Create twin axis for cumulative line
    ax2 = ax1.twinx()
    trace_lengths.sort()
    cumulative_counts = np.cumsum(trace_lengths)/sum(trace_lengths)
    ax2.plot(trace_lengths, cumulative_counts, color='r')
    ax2.set_ylabel('Cumulative Count')
    ax2.set_ylim(0, 1)
    plt.show()
    # Plot histogram with 1 bin per virtual bar for state occurrences
    fig, ax1 = plt.subplots()
    state_occurrences = [len(v) for v in state_to_embedding_map.values()]
    ax1.hist(state_occurrences, bins=max(state_occurrences), cumulative=False)
    ax1.set_title('Occurrence of States Distribution')
    ax1.set_xlabel('State Occurrences')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(-5, max(state_occurrences)+5)
    # Create twin axis for cumulative line
    ax2 = ax1.twinx()
    state_occurrences.sort()
    cumulative_counts = np.cumsum(state_occurrences)/sum(state_occurrences)
    ax2.plot(state_occurrences, cumulative_counts, color='r')
    ax2.set_ylabel('Cumulative Count')
    ax2.set_ylim(0, 1)
    plt.show()

# parse json to dict
state_to_embedding_map = defaultdict(list)
for meta_trace, embedding_trace in zip(meta_train, embedding_train):
    for segment_idx, segment_json in enumerate(meta_trace['segments']):
        state = step_id_to_int[segment_json['step_unique_id']]
        embedding = embedding_trace[segment_idx, :]
        state_to_embedding_map[state].append(embedding)
for meta_trace, embedding_trace in zip(meta_val, embedding_val):
    for segment_idx, segment_json in enumerate(meta_trace['segments']):
        state = step_id_to_int[segment_json['step_unique_id']]
        embedding = embedding_trace[segment_idx, :]
        state_to_embedding_map[state].append(embedding)
state_to_embedding_stats = dict()
for state, embeddings in state_to_embedding_map.items():
    stacked = torch.stack(embeddings)
    mean = stacked.mean(dim=0)
    if len(embeddings) == 1:
        var = torch.zeros_like(mean)
    else:
        var = stacked.std(dim=0)  # or use .std(dim=0) for standard deviation
    state_to_embedding_stats[state] = {'mean': mean, 'std': var}

# augment negative examples
def gen_augment_one_trace(meta_trace, embedding_trace,
                          state_to_embedding_map, random_sample_per_step=10):
    """
    Generate negative examples from a given trace.
    Args:
    meta_trace (dict): Metadata for the trace.
    embedding_trace (torch.Tensor): Embedding of the trace, shape (num_steps, feature_dim).
    state_to_embedding_map (dict): Mapping of states to their embeddings.
    random_sample_per_step (int): Number of negative samples to generate per step.
    Returns:
    tuple: (augmented_traces, valid_traces, count_traces, state_traces)
        augmented_traces: (num_traces, num_steps, feature_dim)
        valid_traces: (num_traces, num_steps,)
        count_traces: (num_traces, num_steps,)
        state_traces: (num_traces, num_steps,)
    """
    num_steps = len(meta_trace['segments'])
    all_states = list(state_to_embedding_map.keys())
    embedding_traces = []
    acceptance_traces = []
    state_traces = []

    # Generate the original positive trace
    original_acceptance = torch.ones(num_steps)
    original_state = torch.tensor([step_id_to_int[segment['step_unique_id']] for segment in meta_trace['segments']])
    embedding_traces.append(embedding_trace)
    acceptance_traces.append(original_acceptance)
    state_traces.append(original_state)

    # for deviate at each step
    for i in range(0, num_steps):
        # for each negative trace
        for _ in range(random_sample_per_step):
            # Create acceptance_trace tensor
            acceptance_trace = torch.ones(num_steps)
            acceptance_trace[i:] = 0
            
            # Replace embeddings from step i onwards
            embedding_trace = embedding_trace.clone()
            state_trace = original_state.clone()
            for j in range(i, num_steps):
                random_state = random.choice(all_states)
                random_embedding = random.choice(state_to_embedding_map[random_state])
                embedding_trace[j] = random_embedding
                state_trace[j] = random_state
            
            embedding_traces.append(embedding_trace)
            acceptance_traces.append(acceptance_trace)
            state_traces.append(state_trace)

    return torch.stack(embedding_traces), torch.stack(acceptance_traces), torch.stack(state_traces)

def gen_negative(meta_data, embedding_data,
                 state_to_embedding_map, random_sample_per_step=10):
    result_dict = defaultdict(list)
    for meta_trace, embedding_trace in tqdm(zip(meta_data, embedding_data)):
        num_steps = len(meta_trace['segments'])
        X_traces = gen_augment_one_trace(
            meta_trace, embedding_trace, state_to_embedding_map, random_sample_per_step
        )
        result_dict[num_steps].append(X_traces)

    # concatenate same length traces together
    data = {}
    for num_steps, traces in result_dict.items():
        data[num_steps] = tuple(torch.cat([t[i] for t in traces], dim=0) for i in range(len(traces[0])))
    
    return data

if __name__ == '__main__':
    # plot_trace_length_distribution(meta_train)
    random_sample_per_step = 0
    data_train = gen_negative(meta_train, embedding_train, state_to_embedding_map, random_sample_per_step)
    data_val = gen_negative(meta_val, embedding_val, state_to_embedding_map, random_sample_per_step)
    torch.save(data_train, f'data_augment/segment_train_{random_sample_per_step}.pt')
    torch.save(data_val, f'data_augment/segment_val_{random_sample_per_step}.pt')
    print("train size (MB):", sum([
        sum([t.nelement() * t.element_size() for t in trace]) 
        for trace in data_train.values()
    ]) / 1024 /1024)