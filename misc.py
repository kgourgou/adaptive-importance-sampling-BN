"""
misc. functions
"""

from scipy import mean, var


def dict_to_string(some_dict):
    result = ""
    sorted_keys = sorted(some_dict,
                         key=lambda key: key)
    new_dict = {key: some_dict[key] for key in sorted_keys}

    for key in new_dict:
        result += str(key) + str(new_dict[key])

    return result


def string_to_dict(s):
    """
    Partial-inverse of "dict_to_string", will only work
    if the state is described by a single digit.
    """
    sp = [a for a in s]
    n = len(sp)

    if n % 2 != 0:
        raise ValueError("string has an odd number of characters.")

    new_dict = {sp[i]: float(sp[i+1]) for i in range(0, n-1, 2)}
    return new_dict


def char_fun(sample_dict, comparison_dict):
    """
    Characteristic function of the comparison_dict.
    """

    for key in comparison_dict:
        if key in sample_dict:
            if sample_dict[key] != comparison_dict[key]:
                return 0
        else:
            return 0

    return 1


class weight_average(object):

    """
    Class to calculate weighted averages.
    Defaults to normal averages if no
    weights are provided.
    """

    def __init__(self, values, weights=None):
        """
        values: iterable of 1xN dimension, the values on
        which to evaluate the quantity of interest.
        weights: iterable of 1xN dimension or None, the corresponding
        weight for each value.
        """
        self.values = values

        if weights is not None:
            self.weights = weights/sum(weights)

    def eval(self, f=lambda x: 1):
        """
        evaluate function on the values
        and take weighted average.
        """

        if self.weights is None:
            vec = [f(v) for v in self.values]
            result = mean(vec)
            variance = var(vec)
        else:
            v = self.values
            vec = [f(v[i])*w for i, w in enumerate(self.weights)]
            result = sum(vec)
            variance = var(vec)

        return result, variance





if __name__ == '__main__':
    some_dict = {"B": 0, "A":1}
    print(dict_to_string(some_dict))
