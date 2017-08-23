from misc import weight_average, string_to_dict, char_fun


def update_proposal_cpt(proposal, samples, weights, index, graph,
                        evidence_parents, eta_rate):
    """
    Updates current proposal given the new data.

    Arguments
    =========
    samples: the current samples to use.
    index: the current index (used to pick the weight)

    TODO : define this outside.
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
                ratio = p / q

                proposal.cpt[node][key][0] += eta_rate(index) * (
                    ratio - proposal.cpt[node][key][0])
                proposal.cpt[node][key][1] += eta_rate(index) * (
                    1 - ratio - proposal.cpt[node][key][1])

    return proposal
