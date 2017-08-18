"""
misc. functions
"""


def dict_to_string(some_dict):
    result = ""
    sorted_keys = sorted(some_dict,
                         key=lambda key: key)
    new_dict = {key:some_dict[key] for key in sorted_keys}

    for key in new_dict:
        result += str(key) + str(new_dict[key])

    return result


if __name__ == '__main__':
    some_dict = {"B": 0, "A":1}
    print(dict_to_string(some_dict))
