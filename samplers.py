from toposort import toposort_flatten
import scipy as sc

from scipy.stats import bernoulli


def likelihood_weight(parents, cpt, evidence, num_of_samples=1000):
    """

    Samples weights and particles given the description of
    a network through parents and CPT.

    experimental code.

    """
    topo_nodes = toposort_flatten(parents)
    topo_nodes = topo_nodes[1::]

    n_nodes = len(topo_nodes)

    def single_sample(topo_nodes, n_nodes, parents, cpt):
        # single particle
        particle = dict(zip(topo_nodes, sc.zeros(n_nodes)))

        w = 1
        for i, node_id in enumerate(topo_nodes):

            if parents[node_id] == {None}:
                parent_vals = []
            else:
                parent_vals = [particle[a] for a in parents[node_id]]

            if node_id not in evidence:
                particle[node_id] = cpt_sample(node_id, parent_vals, cpt)
            else:
                # evidence node
                particle[node_id] = evidence[node_id]
                if len(parent_vals) == 0:
                    w *= cpt[node_id][0]
                else:
                    node_val = evidence[node_id]
                    w *= cpt[node_id][tuple(parent_vals)][node_val]

        return particle, w

    samples = []
    weights = sc.zeros([num_of_samples])
    for i in range(num_of_samples):
        temp_particle, temp_w = single_sample(topo_nodes, n_nodes, parents,
                                              cpt)
        samples.append(temp_particle)
        weights[i] = temp_w

    return samples, weights


def cpt_sample(node, parent_vals, cpt):
    """
    Generates single sample for node in {0, 1}.
    """
    if len(parent_vals) == 0:
        p = cpt[node][0]
        sample = bernoulli.rvs(p)
    else:
        p = cpt[node][tuple(parent_vals)][0]
        sample = bernoulli.rvs(p)

    return sample


