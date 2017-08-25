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
    for node in evidence_parents:
        if node is None:
            continue
        elif proposal.is_root_node(node):
            # root node -- update prior
            def f(sample):
                res1 = char_fun(sample, {node: 0})
                return res1

            p, _ = wei_est.eval(f)

            proposal.prior_dict[node][0] += eta_rate(index) * (
                p - proposal.prior_dict[node][0])
            proposal.prior_dict[node][1] += eta_rate(index) * (
                1 - p - proposal.prior_dict[node][1])
        else:
            # rest of the nodes -- lambdas
            parents = [ident for ident in graph[node]]
            for key in proposal.lambdas[node]:
                state_vec = {p: False for p in parents}
                if key == "leak_node":

                    def f(sample):
                        return char_fun(sample, state_vec)

                    q, _ = wei_est.eval(f)

                    state_vec[node] = False

                    def f(sample):
                        return char_fun(sample, state_vec)

                    p, _ = wei_est.eval(f)

                    if abs(q) < 1e-30:
                        continue

                    ratio = exp(log(p)-log(q))

                    proposal.lambdas[node]["leak_node"] += eta_rate(index) * (
                        ratio - proposal.lambdas[node]["leak_node"])
                else:
                    # TODO: something is wrong with this part
                    # and it doesn't estimate correctly
                    state_vec[key] = True

                    def f(sample):
                        return char_fun(sample, state_vec)

                    q, _ = wei_est.eval(f)
                    state_vec[node] = False

                    def f(sample):
                        return char_fun(sample, state_vec)

                    p, _ = wei_est.eval(f)

                    ratio = exp(log(p)-log(q))

                    proposal.lambdas[node][key] += eta_rate(index) * (
                        ratio - proposal.lambdas[node][key])

    return proposal
