import scipy
import json
import pickle
from pprint import pprint
cards = [
    "510",
    #, "511",
    # "553",
    # "558",
    # "562",
    # "609"
    # "610",
    # "615",
    # "618",
    # "629",
]


error = {}

for card in cards:
    with open("samples_and_weights/marginals_{}.json".format(card)) as marg:
        adis_data = json.load(marg)

    with open("/data/testset/prior_proposal_marginals/marginals_{}.p"
              .format(card), 'rb') as truth:
        ground_truth = pickle.load(truth)

        error[card] = {key: abs(adis_data[key]
                       -ground_truth[key]['True']) for key in adis_data}

        print(max(error[card], key=error[card].get))
        error_values = [s for s in error[card].values()]
        print("max relative error = {}".format(max(error_values)))

        error_ga = [[error[card][key],
                     ground_truth[key]["True"],
                     adis_data[key]] for key in ground_truth]
        error_ga = sorted(error_ga)
        pprint(error_ga)
