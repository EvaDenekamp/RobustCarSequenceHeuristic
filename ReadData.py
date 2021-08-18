import pandas as pd
import numpy as np
from scipy.stats import uniform

def descriptive_uniform(vehicle_failure_probababilty, number_of_samples):
    C=[]
    xd = [uniform.ppf((i - 0.5) / len(vehicle_failure_probababilty)) for i in
          range(1, 1 + len(vehicle_failure_probababilty))]
    # print(xd)
    for j in range(number_of_samples):
        np.random.shuffle(xd)
        C.append([(1 if xd[k] > vehicle_failure_probababilty[k] else 0) for k in range(len(xd))])
    return np.array(C)


def descriptive_uniform_with_bound(vehicle_failure_probababilty, number_of_samples, bound):
    C=[]
    xd = [round(uniform.ppf((i - 0.5) / len(vehicle_failure_probababilty)), 3) for i in
          range(1, 1 + len(vehicle_failure_probababilty))]
    for j in range(1,number_of_samples+1):
        np.random.shuffle(xd)
        C.append([(1 if xd[k] > vehicle_failure_probababilty[k] else 0) for k in range(len(xd))])
        if j%(bound/2)==0:
            # print("check")
            CC=np.array(C)
            dt = np.dtype((np.void, CC.dtype.itemsize * CC.shape[1]))
            b = np.ascontiguousarray(CC).view(dt)
            unq_scenarios, cnt_unq_scenarios = np.unique(b,
                                                         return_counts=True)  # save only the unique scenarios and the number of occurrence of the same scenario
            unq_scenarios = unq_scenarios.view(CC.dtype).reshape(-1, CC.shape[1])
            if len(unq_scenarios)>=(bound): # We do not want much more than 5-10%*S unique scenarios!
                return np.array(C)
    return np.array(C)

def get_data(file,number_of_samples,all_weights_are_one, number_of_test_samples):
    ### get data from excel
    data = pd.read_excel(file, header=None)
    number_vehicles = int(data.iat[0,0])
    number_options = int(data.iat[0,1])

    ### get index sets and parameters
    # set of sequence slots
    t_vehicles = np.array(range(1, number_vehicles + 1))

    # set of vehicles
    v = np.array(range(1, number_vehicles + 1),dtype=str)

    # set of options
    options = np.array(range(1, number_options + 1))

    # parameter car sequencing rules
    rules = {}
    for option in options:
        rules[option] = [data.iat[1,option-1],data.iat[2,option-1]]

    # parameter priorities of options
    if all_weights_are_one:
        weights = np.ones(len(options))
    else:
        weights=[]
        for option in options:
            weights=weights.append(data.iat[3,option-1])


    # parameter a indicator a==1 if vehicle v has option o, otherwise a==0
    a = np.repeat(data.loc[3:,2:len(data.columns)-2].values.astype(int), repeats=data.loc[3:,1].values.astype(int), axis=0)

    # parameter remove probability of a vehicle
    # remove =  np.repeat(data.loc[3:,len(data.columns)-1].values, repeats=data.loc[3:,1].values, axis=0) # same prob
    p=np.round(data.loc[3:,len(data.columns)-1].values,3)
    rep=data.loc[3:,1].values.astype(int)
    remove=np.array([])
    for q in range(len(p)):
        for r in range(rep[q]):
            remove = np.append(remove, p[q])
            # remove=np.append(remove,np.random.uniform(low=p[q]*0.8, high=min(1,1.2*p[q]), size=1))
    remove=np.round(remove,3) # expected prob of option


    # set of failure scenarios
    scenarios = descriptive_uniform_with_bound(remove, number_of_samples, bound=50) # some scenarios are the same
    dt = np.dtype((np.void, scenarios.dtype.itemsize * scenarios.shape[1]))
    b = np.ascontiguousarray(scenarios).view(dt)
    unq_scenarios, cnt_unq_scenarios = np.unique(b, return_counts=True) # save only the unique scenarios and the number of occurrence of the same scenario
    unq_scenarios = unq_scenarios.view(scenarios.dtype).reshape(-1, scenarios.shape[1])

    # set of test scenarios
    test_scenarios = descriptive_uniform_with_bound(remove, number_of_test_samples, bound=500)
    dt = np.dtype((np.void, test_scenarios.dtype.itemsize * test_scenarios.shape[1]))
    b = np.ascontiguousarray(test_scenarios).view(dt)
    unq_test_scenarios, cnt_unq_test_scenarios = np.unique(b,
                                                           return_counts=True)  # save only the unique scenarios and the number of occurrence of the same scenario
    unq_test_scenarios = unq_test_scenarios.view(test_scenarios.dtype).reshape(-1, test_scenarios.shape[1])

    ### add dummies (needed for correct counting of viiolations)
    max_No=int(rules[int(max(rules, key=rules.get))][1])
    number_front_dummies = int(abs(np.min(np.array([rules[rule][0]-rules[rule][1] for rule in rules]) + 2-1)))
    number_back_dummies = int(max_No)

    # sequence slot with dummies
    t_min = min(1, -number_front_dummies + 1)
    t_max = number_back_dummies + number_vehicles + 1
    t_with_dummies = np.array(range(t_min, t_max))

    # vehicles with dummies
    v_with_dummies = np.append(["dummy_f" + str(i) for i in range(number_front_dummies)],v)
    v_with_dummies = np.append(v_with_dummies, ["dummy_b" + str(i) for i in range(number_back_dummies)])

    # dummies do not have any options
    a = np.append(np.zeros((number_front_dummies, len(options))), a, axis=0)
    a = np.append(a, np.zeros((number_back_dummies, len(options))), axis=0)

    # dummies are never removed
    remove = np.append(np.zeros(number_front_dummies), remove)
    remove = np.append(remove, np.zeros(number_back_dummies))

    # in each scenario dummies can not be removed
    unq_scenarios = np.append(np.ones((len(unq_scenarios), number_front_dummies)), unq_scenarios, axis=1)
    unq_scenarios = np.append(unq_scenarios, np.ones((len(unq_scenarios), number_back_dummies)), axis=1)

    # in each test scenario dummies can not be removed
    unq_test_scenarios = np.append(np.ones((len(unq_test_scenarios), number_front_dummies)), unq_test_scenarios, axis=1)
    unq_test_scenarios = np.append(unq_test_scenarios, np.ones((len(unq_test_scenarios), number_back_dummies)), axis=1)



    return v, v_with_dummies, options, unq_scenarios, cnt_unq_scenarios, t_vehicles, t_with_dummies, a, rules, remove, max_No, weights,unq_test_scenarios, cnt_unq_test_scenarios

