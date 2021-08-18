import pandas as pd
import ALNS
from alns.criteria import SimulatedAnnealing
import InitialSolution
import ReadData
import numpy as np
import numpy.random as rnd
import random
from RCSP import RCSP
import itertools
import time
import matplotlib.pyplot as plt

SEED = 5432

####################################### Operator Functions #######################################
# The operator function random of Hottenrott et al.
def Random(current, random_state):
    new = current.copy() # copy

    # # destroy multiple!
    if len(new.sequence_slots)>=80:
        size=random.randint(1,int(0.025*len(new.sequence_slots)))
    else:
        size=1

    indexes=np.array([])
    vehicles_changed=np.array([])

    for i in range(size):
        # Destroy randomly one of the vehicles
        destroy_index=random.choice(pick_from_slots)
        destroy_element = new.sequence_slots[destroy_index]
        del new.sequence_slots[destroy_index]

        # Replace at another place, but not the same place as destroyed one
        slots = np.delete(pick_from_slots, np.where(pick_from_slots == destroy_index)[0][0])
        add_at_index = random.choice(slots)
        new.sequence_slots.insert(add_at_index, destroy_element)
        indexes=np.append(indexes,destroy_index+2)
        indexes = np.append(indexes, add_at_index)
        vehicles_changed=np.append(vehicles_changed,destroy_element[0])

    # Update violations
    slots_to_update = new.update_which_slots(indexes,max_No+2, vehicles_plus_dummies)
    new=new.update_violations(slots_to_update, vehicles_plus_dummies, a_vo,options,sequence_rules, weights,max_No)
    return new

# The operator function swap of Hottenrott et al.
def Swap(current, random_state):
    new = current.copy() # copy

    pairs=[]

    ### Consider all options: ###
    # for o in range(len(options)):
    #     option_subsequences = new.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, o,
    #                                                                        sequence_rules[options[o]][0],
    #                                                                        sequence_rules[options[o]][1], a_vo)
    #     for t in pick_from_slots:
    #         if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][o] == 0 and option_subsequences[
    #             t] == 1:  # if vehicle is uncritical and is in full-subsequence
    #             for t_ in pick_from_slots:  # check if you can find another uncritical vehicle with a lower failure probability within a slack-subsequence or non-subsequence
    #                 if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t_][0])[0][0]][o] == 0 and \
    #                         failure_probability_of_each_vehicle[
    #                             np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]] > \
    #                         failure_probability_of_each_vehicle[
    #                             np.where(vehicles_plus_dummies == new.sequence_slots[t_][0])[0][0]] and \
    #                         option_subsequences[t_] <= 0:
    #                     pairs.append([t, t_])

    ### Consider only one option (decrease compuation time): ###
    o = random.choice(options)
    option_subsequences = new.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, np.where(options==o)[0][0],
                                                                       sequence_rules[o][0],
                                                                       sequence_rules[o][1], a_vo)
    for t in pick_from_slots:
        if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][np.where(options==o)[0][0]] == 0 and option_subsequences[
            t] == 1:  # if vehicle is uncritical and is in full-subsequence
            for t_ in pick_from_slots:  # check if you can find another uncritical vehicle with a lower failure probability within a slack-subsequence or non-subsequence
                if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t_][0])[0][0]][np.where(options==o)[0][0]] == 0 and \
                        failure_probability_of_each_vehicle[
                            np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]] > \
                        failure_probability_of_each_vehicle[
                            np.where(vehicles_plus_dummies == new.sequence_slots[t_][0])[0][0]] and \
                        option_subsequences[t_] <= 0:
                    pairs.append([t, t_])


    # Swap the vehicles from a random chosen pair if a pair exists
    if len(pairs)>0:
        # Swap multiple!
        if len(pairs) > 10:
            size = random.randint(1, 10)
        else:
            size = 1
        indexes = np.array([])

        pairs=np.array(pairs)
        for i in range(size):
            choose_random_pair=pairs[random.randint(0,len(pairs)-1)]
            swap_index1=choose_random_pair[0]
            swap_element1=new.sequence_slots[swap_index1]
            swap_index2=choose_random_pair[1]
            swap_element2 = new.sequence_slots[swap_index2]
            new.sequence_slots[swap_index2]=swap_element1
            new.sequence_slots[swap_index1] = swap_element2
            indexes=np.append(indexes,swap_index1)
            indexes = np.append(indexes, swap_index2)
            pairs=np.delete(pairs, (np.where((pairs==choose_random_pair).all(axis=1))[0][0]),axis=0)

        # Update violations
        slots_to_update = new.update_which_slots(indexes,max_No+1, vehicles_plus_dummies)
        new=new.update_violations(slots_to_update, vehicles_plus_dummies, a_vo,options,sequence_rules, weights,max_No)
    return new

# The operator function critical of Hottenrott et al.
def Critical(current, random_state):
    new = current.copy() # copy

    critical_vehicles=[]
    destinations=[]
    pairs=np.empty((0, 2), int)

    ### Consider all options: ###
    # for o in range(len(options)):
    #     # start_time_destroy_dummy=time.time()
    #     option_subsequences=new.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, o, sequence_rules[options[o]][0], sequence_rules[options[o]][1], a_vo)
    #     for t in pick_from_slots:
    #         if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][o]==1 and option_subsequences[t]>=1: # if vehicle is critical and is in full- or overfull-subsequence
    #             critical_vehicles.append(t)
    #     # print("Time part 1a:", time.time() - start_time_destroy_dummy)
    #     # start_time_destroy_dummy = time.time()
    #     if len(critical_vehicles)>0: # if there are critical vehicles in option o, find optional destinations
    #         t=random.choice(critical_vehicles)
    #         for t_ in pick_from_slots:
    #             if option_subsequences[t_] <= 0: # an optional destination is in a slack- or non-subsequence and  ..
    #                 # when we place the critical vehicle at slot t, the sub-sequence is still slack
    #                 if new.still_slack_subsequence_if_you_replace(t, t_, vehicles_plus_dummies, o,  sequence_rules[options[o]][0], sequence_rules[options[o]][1], a_vo):
    #                     destinations.append(t_)
    #         if len(destinations)>0:
    #             combinations=np.array([i for i in itertools.product([t],destinations)])
    #             pairs=np.append(pairs,combinations,axis=0)

    ### Consider only one option (decrease compuation time): ###
    o=random.choice(options)
    option_subsequences = new.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, np.where(options==o)[0][0],
                                                                       sequence_rules[o][0],
                                                                       sequence_rules[o][1], a_vo)
    for t in pick_from_slots:
        if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][np.where(options==o)[0][0]] == 1 and option_subsequences[
            t] >= 1:  # if vehicle is critical and is in full- or overfull-subsequence
            critical_vehicles.append(t)
    if len(critical_vehicles) > 0:  # if there are critical vehicles in option o, find optional destinations
        t = random.choice(critical_vehicles)
        for t_ in pick_from_slots:
            if option_subsequences[t_] <= 0:  # an optional destination is in a slack- or non-subsequence and  ..
                # when we place the critical vehicle at slot t, the sub-sequence is still slack
                if new.still_slack_subsequence_if_you_replace(t, t_, vehicles_plus_dummies, np.where(options==o)[0][0],
                                                              sequence_rules[o][0],
                                                              sequence_rules[o][1], a_vo):
                    destinations.append(t_)
        if len(destinations) > 0:
            combinations = np.array([i for i in itertools.product([t], destinations)])
            pairs = np.append(pairs, combinations, axis=0)



    # Replace vehicles from a random chosen pair of (delete_index,replace_at_index)
    if len(pairs)>0:
        indexes = np.array([])
        choose_random_pair=random.choice(pairs)
        destroy_index=choose_random_pair[0]
        destroy_element=new.sequence_slots[destroy_index]
        add_at_index = choose_random_pair[1]
        del new.sequence_slots[destroy_index]
        new.sequence_slots.insert(add_at_index, destroy_element)
        indexes = np.append(indexes, destroy_index)
        indexes = np.append(indexes, add_at_index)

        # Update violations
        slots_to_update = new.update_which_slots(indexes,max_No+1, vehicles_plus_dummies)
        new=new.update_violations(slots_to_update, vehicles_plus_dummies, a_vo,options,sequence_rules, weights,max_No)
    return new

# The operator function uncritical of Hottenrott et al.
def Uncritical(current, random_state):
    new = current.copy() # copy

    uncritical_vehicles=[]
    destinations=[]
    pairs=np.empty((0, 2), int)

    ### Consider all options: ###
    # for o in range(len(options)):
    #     option_subsequences=new.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, o, sequence_rules[options[o]][0], sequence_rules[options[o]][1], a_vo)
    #     for t in pick_from_slots:
    #         if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][o]==0 and option_subsequences[t]<=0: # if vehicle is uncritical and is in slack- or non-subsequence
    #             uncritical_vehicles.append(t)
    #     if len(uncritical_vehicles)>0: # if there are uncritical vehicles in option o, find optional destinations
    #         for t in uncritical_vehicles:
    #             for t_ in pick_from_slots:
    #                 if option_subsequences[t_] >=1: # an optional destination is in a full or overfull-subsequence
    #                     destinations.append(t_+1) # notes that you are placing it after index t_! then you influence the subsequence at t_
    #             if len(destinations)>0:
    #                 combinations=np.array([i for i in itertools.product([t],destinations)])
    #                 pairs=np.append(pairs,combinations,axis=0)

    ### Consider only one option (decrease compuation time): ###
    o = random.choice(options)
    option_subsequences = new.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, np.where(options==o)[0][0],
                                                                       sequence_rules[o][0],
                                                                       sequence_rules[o][1], a_vo)
    for t in pick_from_slots:
        if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][np.where(options==o)[0][0]] == 0 and option_subsequences[
            t] <= 0:  # if vehicle is uncritical and is in slack- or non-subsequence
            uncritical_vehicles.append(t)
    if len(uncritical_vehicles) > 0:  # if there are uncritical vehicles in option o, find optional destinations
        for t in uncritical_vehicles:
            for t_ in pick_from_slots:
                if option_subsequences[t_] >= 1:  # an optional destination is in a full or overfull-subsequence
                    destinations.append(
                        t_ + 1)  # notes that you are placing it after index t_! then you influence the subsequence at t_
            if len(destinations) > 0:
                combinations = np.array([i for i in itertools.product([t], destinations)])
                pairs = np.append(pairs, combinations, axis=0)

    # replace vehicles from a random chosen pair of (delete_index,replace_at_index)
    if len(pairs)>0:
        indexes = np.array([])
        choose_random_pair = random.choice(pairs)
        destroy_index = choose_random_pair[0]
        destroy_element = new.sequence_slots[destroy_index]
        add_at_index = choose_random_pair[1]
        del new.sequence_slots[destroy_index]
        new.sequence_slots.insert(add_at_index, destroy_element)

        indexes = np.append(indexes, destroy_index)
        indexes = np.append(indexes, add_at_index)

        # Update violations
        slots_to_update = new.update_which_slots(indexes,max_No+1, vehicles_plus_dummies)
        new=new.update_violations(slots_to_update, vehicles_plus_dummies, a_vo,options,sequence_rules, weights,max_No)
    return new

# The operator function begin/end of Hottenrott et al.
def BeginEnd(current, random_state):
    new = current.copy()

    options_that_fails_for_observation3=[] # array which determines where observation 3 of Hottenrott et al.

    ### Consider all options: ###
    # for o in range(len(options)):
    #     # the sum of vehicles requiring option o must be larger or equal than H_o
    #     if sum([a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][o] for t in pick_from_slots])>=sequence_rules[options[o]][0]:
    #         observation3 = True
    #         dummy = 0
    #         while dummy<sequence_rules[options[o]][0] and observation3: # check if observation 3 holds
    #             # observation 3 does not hold when there are no H_o vehicles in the begining and end requiring option o
    #             if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[pick_from_slots[0]+dummy][0])[0][0]][o]==0 or a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[pick_from_slots[-1]-dummy][0])[0][0]][o]==0:
    #                 observation3 = False
    #             dummy += 1
    #         if not observation3:
    #             options_that_fails_for_observation3.append(o)

    ### Consider only one option (decrease compuation time): ###
    o = random.choice(options)

    # The sum of vehicles requiring option o must be larger or equal than H_o
    if sum([a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][np.where(options==o)[0][0]] for t in pick_from_slots]) >= \
            sequence_rules[o][0]:
        observation3 = True
        dummy = 0
        while dummy < sequence_rules[o][0] and observation3:  # check if observation 3 holds
            # observation 3 does not hold when there are no H_o vehicles in the beginning and end requiring option o
            if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[pick_from_slots[0] + dummy][0])[0][0]][
                np.where(options==o)[0][0]] == 0 or a_vo[
                np.where(vehicles_plus_dummies == new.sequence_slots[pick_from_slots[-1] - dummy][0])[0][0]][
                np.where(options==o)[0][0]] == 0:
                observation3 = False
            dummy += 1
        if not observation3:
            options_that_fails_for_observation3.append(np.where(options==o)[0][0])

    # Place H_o vehicles from a random option that fails
    if len(options_that_fails_for_observation3)>0:
        random_option = random.choice(options_that_fails_for_observation3)

        # Determine at which slot indexes a vehicle requires random option
        which_indexes_require_random_option=[]
        for t in pick_from_slots:
            if a_vo[np.where(vehicles_plus_dummies == new.sequence_slots[t][0])[0][0]][random_option] == 1:
                which_indexes_require_random_option.append(t)

        # Where is the vehicle placed?
        place_at_begin=random.choice([True, False])

        # Chose a random vehicle from which_indexes_require_random_option and replace randomly at beginning or end
        random_index_require_option=random.choice(which_indexes_require_random_option)
        place_at=pick_from_slots[0] if place_at_begin else pick_from_slots[-1]
        delete_element = new.sequence_slots[random_index_require_option]
        del new.sequence_slots[random_index_require_option]
        new.sequence_slots.insert(place_at, delete_element)

        # Update violations
        slots_to_update = new.update_which_slots([random_index_require_option, place_at], max_No,
                                                 vehicles_plus_dummies)
        new = new.update_violations(slots_to_update, vehicles_plus_dummies, a_vo, options, sequence_rules, weights,max_No)
    return new

####################################### Main #######################################
# Simulation parameters
number_of_test_samples=10000
number_of_samples_failure_scenario=1000
all_weights_are_one=True


# load data
file_name='example_sequence.xlsx'
file=pd.read_excel(file_name, engine='openpyxl')
vehicles_without_dummies, vehicles_plus_dummies, options, failure_scenarios_unique, number_occurrence_scenario, sequence_slots_vehicles, sequence_slots_with_dummies, a_vo, sequence_rules, failure_probability_of_each_vehicle, max_No, weights, unq_test_scenarios, cnt_unq_test_scenarios = ReadData.get_data(
    file_name, number_of_samples_failure_scenario, all_weights_are_one, number_of_test_samples)
print(file_name)


# initial solution is the optimal non robust car sequencing (you can also choose between random order or the order as in the file)
start_time_initial = time.time()
initialsolution_sequence = InitialSolution.non_robust_car_sequence(failure_scenarios_unique,
                                                                      number_occurrence_scenario,
                                                                      vehicles_without_dummies,
                                                                      vehicles_plus_dummies,
                                                                      options,
                                                                      sequence_slots_vehicles,
                                                                      sequence_slots_with_dummies,
                                                                      a_vo,
                                                                      sequence_rules, weights)
initialsolution = RCSP(initialsolution_sequence, [failure_scenarios_unique, number_occurrence_scenario])
objective_initial=initialsolution.objective()
print("Computation time of the initial solution:", time.time() - start_time_initial)
print("Objective initial with sample scenarios for heuristic:", objective_initial)
print("Objective initial with sample scenarios for test:", initialsolution.test_objective(sequence_slots_with_dummies,
                                                                         vehicles_plus_dummies, max_No, options,
                                                                         sequence_rules, a_vo, weights,
                                                                         unq_test_scenarios,
                                                                         cnt_unq_test_scenarios))

pick_from_slots = ((sequence_slots_with_dummies[:, None] == sequence_slots_vehicles).argmax(axis=0)) # define that only the slots which is not a dummy slot can be changed
# If initial objective is not zero, perform ALNS
if objective_initial > 0:
    # ALNS parameters
    rnd_state = rnd.RandomState(SEED)
    want_plots = True
    want_time_limit = True
    number_of_iterations = 10000
    alns = ALNS.ALNS(rnd_state)
    alns.add_operator(Random)
    alns.add_operator(Swap)
    alns.add_operator(Critical)
    alns.add_operator(Uncritical)
    alns.add_operator(BeginEnd)
    cooling_down_rate = 0.99975
    start_temperature = initialsolution.determine_start_temperature(
        solution_that_is_x_percentage_worse_than_initial=0.05, probability_of_accepting=0.5)
    end_temperature = start_temperature * cooling_down_rate ** number_of_iterations
    criterion = SimulatedAnnealing(start_temperature, end_temperature, cooling_down_rate,
                                   method="exponential")

    start_time_robust = time.time()
    result = alns.iterate(initialsolution, [50, 20, 5, 0], 0.9, criterion, collect_stats=want_plots, limited_time=want_time_limit)
    solution = result.best_state
    objective_robust=solution.test_objective(sequence_slots_with_dummies,vehicles_plus_dummies,max_No,options,sequence_rules,a_vo,weights, unq_test_scenarios, cnt_unq_test_scenarios)
    time_robust=time.time() - start_time_robust
    difference = objective_robust - objective_initial
    difference = difference / objective_initial * 100
    print("Objective robust:", objective_robust)
    print("Difference [%]:", difference)
    print("Computation time robust [s]", time_robust)
    if want_plots:
        _, ax = plt.subplots(figsize=(12, 6))
        result.plot_objectives(ax=ax, file_name=file_name)
        plt.show()
        figure = plt.figure("operator_counts", figsize=(12, 6))
        figure.subplots_adjust(bottom=0.15, hspace=.5)
        result.plot_operator_counts(figure=figure, title="Operator diagnostics", file_name=file_name)
        plt.show()
else:
    print("Initial sequence is zero, therefore, the the robust car sequence equals the initial sequence")





