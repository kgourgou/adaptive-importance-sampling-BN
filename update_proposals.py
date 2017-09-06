from misc import weight_average, string_to_dict, char_fun

from scipy import log, exp


def update_proposal_cpt(proposal, samples, weights, index, graph,
                        evidence_parents, eta_rate):
    """
    Updates current proposal given the new data.

    Arguments
    =========
    samples: the current samples to use.
    index: the current index (used to pick the weight)
    """

    # Initialize weighted estimator
    wei_est = weight_average(samples, weights)

    # estimate CPT table from samples
    for node in evidence_parents:
        if node is None:
            continue
        elif proposal.is_root_node(node):
            # root node
            def f(sample):
                res1 = char_fun(sample, {node: 0})
                return res1

            p, _ = wei_est.eval(f)

            proposal.cpt[node][0] += eta_rate(index) * (
                p - proposal.cpt[node][0])
            proposal.cpt[node][1] += eta_rate(index) * (
                1 - p - proposal.cpt[node][1])
        else:
            # rest of the nodes
            for key in proposal.cpt[node]:

                parent_dict = string_to_dict(key)

                def f(sample):
                    res1 = char_fun(sample, {node: 0})
                    res2 = char_fun(sample, parent_dict)
                    return res1 * res2

                def g(sample):
                    res2 = char_fun(sample, parent_dict)
                    return res2

                p, _ = wei_est.eval(f)
                q, _ = wei_est.eval(g)

                if abs(p - q) < 1e-10:
                    ratio = 1
                else:
                    ratio = p / q

                proposal.cpt[node][key][0] += eta_rate(index) * (
                    ratio - proposal.cpt[node][key][0])
                proposal.cpt[node][key][1] += eta_rate(index) * (
                    1 - ratio - proposal.cpt[node][key][1])

    return proposal


def update_proposal_lambdas(proposal, samples, weights, index, graph,
                            evidence_parents, eta_rate):
    """
    Updates current proposal given the new data.

    Arguments
    =========
    samples: the current samples to use.
    index: the current index (used to pick the weight)
    """

    # Initialize weighted estimator
    wei_est = weight_average(samples, weights)

    # estimate CPT table from samples
    for child in evidence_parents:

        if child is None:
            continue
        elif proposal.is_root_node(child):
            # root child -- update priors using current samples

            def f(sample):
                res1 = char_fun(sample, {child: 1})
                return res1
            p, _ = wei_est.eval(f)

            proposal.prior_dict[child][0] += eta_rate(index) * (
                1 - p - proposal.prior_dict[child][0])
            proposal.prior_dict[child][1] += eta_rate(index) * (
                p - proposal.prior_dict[child][1])

        else:
            # rest of the childs -- lambdas
            parents = [ident for ident in graph[child]]
            for parent in proposal.lambdas[child]:
                state_vec = {p: False for p in parents}
                if parent == "leak_node":
                    def f(sample):
                        return char_fun(sample, state_vec)

                    q, _ = wei_est.eval(f)

                    state_vec[child] = False

                    def f(sample):
                        return char_fun(sample, state_vec)

                    p, _ = wei_est.eval(f)
                    if abs(p) < 1e-16 or abs(q) < 1e-16:
                        # Do not update if probabilities are
                        # too small
                        continue

                    ratio = exp(log(p) - log(q))

                    proposal.lambdas[child]["leak_node"] += eta_rate(index) * (
                        ratio - proposal.lambdas[child]["leak_node"])
                else:
                    state_vec[parent] = True

                    def f(sample):
                        return char_fun(sample, state_vec)

                    q, _ = wei_est.eval(f)
                    state_vec[child] = False

                    def f(sample):
                        return char_fun(sample, state_vec)

                    p, _ = wei_est.eval(f)

                    if abs(p) < 1e-16 or abs(q) < 1e-16:
                        # Do not update if probabilities are
                        # too small
                        continue

                    ratio = exp(log(p) - log(q))

                    proposal.lambdas[child][parent] += eta_rate(index) * (
                        ratio - proposal.lambdas[child][parent])

    return proposal
