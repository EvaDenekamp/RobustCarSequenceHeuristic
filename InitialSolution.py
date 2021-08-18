import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Determine the optimal non-robust car sequence planning as described in Appendix A.1 and determine the number of violations
def non_robust_car_sequence(failure_scenarios_unique, number_occurence_scenario,vehicles, vehicles_plus_dummies, options, sequence_slots,sequence_slots_with_dummies,a_vo,sequence_rules,weight):
    # Create model
    m = gp.Model('CSP')

    # Create decision variables
    X = m.addVars(vehicles_plus_dummies, sequence_slots_with_dummies, vtype=GRB.BINARY, name="X_vt")
    Y = m.addVars(len(options), lb=0, vtype=GRB.INTEGER, name="Y_co")
    maxterm = m.addVars(len(options), sequence_slots_with_dummies, lb=0, vtype=GRB.INTEGER, name="maxterms")

    # Create constraints
    m.addConstrs((gp.quicksum(X[v, t] for v in vehicles_plus_dummies) == 1 for t in sequence_slots_with_dummies),
                 name="one_slot_per_vehicle_all")
    m.addConstrs((gp.quicksum(X[v, t] for t in sequence_slots_with_dummies) == 1 for v in vehicles_plus_dummies),
                 name="one_vehicle_per_slot_all")
    m.addConstrs((gp.quicksum(X[v, t] for v in vehicles) == 1 for t in sequence_slots),
                 name="one_slot_per_vehicle")
    m.addConstrs((gp.quicksum(X[v, t] for t in sequence_slots) == 1 for v in vehicles),
                 name="one_vehicle_per_slot")
    m.addConstrs((maxterm[o, t] >=
                  gp.quicksum(gp.quicksum(a_vo[np.where(vehicles_plus_dummies == v)[0][0]][o]
                                                           * X[v, t_] for v in vehicles_plus_dummies)
                                               for t_ in range(t, int(t + sequence_rules[options[o]][1])))
                  - sequence_rules[options[o]][0]
                  for o in range(len(options))
                  for t in range(int(sequence_rules[options[o]][0] - sequence_rules[options[o]][1] + 2),
                                 int(len(vehicles) - sequence_rules[options[o]][0] + 1))), name="maxterm")
    m.addConstrs((Y[o] == gp.quicksum(maxterm[o, t]
                                      for t in range(int(sequence_rules[options[o]][0] - sequence_rules[options[o]][1] + 2),
                                                     int(len(vehicles) - sequence_rules[options[o]][0] + 1)))
                  for o in range(len(options))), name="Y_co")  # determination of Y[c,o]

    # Create objective
    m.setObjective(gp.quicksum(Y[o] for o in range(len(options))), GRB.MINIMIZE)

    # Save model for inspection
    m.update()
    m.write('ExpectedViolations.lp')

    # Run optimization engine
    m.Params.LogToConsole = 0 # do not show the log
    m.Params.timeLimit = 60 # we limit the MINLP with 60s
    m.optimize()

    print("runtime Gurobi MINLP",m.Runtime)

    # Determine the order of the sequence
    solution = []
    for t in sequence_slots_with_dummies:
        for v in vehicles_plus_dummies:
            if (X[v, t].x > 0.9):
                solution.append([v]) # add vehicle to each slot
    print("Order of non-robust car sequence: ", solution)

    # Determine the number of violations for each slot in each remove scenario
    for slot in range(len(sequence_slots_with_dummies)):
        violations_per_slot=0
        if slot <= np.where(sequence_slots_with_dummies == sequence_slots[-1])[0][0]: # if vehicle is smaller than the back dummies
            for c in failure_scenarios_unique:
                # print(c,slot)
                if c[np.where(vehicles_plus_dummies == solution[slot][0])[0][0]] == 1:  # if vehicle at slot t does not fail in scenario c you may count the violations
                    for o in range(len(options)):
                        violations_dummy = -1 * sequence_rules[options[o]][0]
                        dummy = 0
                        t = slot
                        while dummy < sequence_rules[options[o]][1]:
                            if c[np.where(vehicles_plus_dummies == solution[t][0])[0][0]] == 1:  # if vehicle at slot t does not fails in scenario c do:
                                dummy += 1
                                if a_vo[np.where(vehicles_plus_dummies == solution[t][0])[0][0]][o] == 1:  # if vehicle at slot t requirs option o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             do:
                                    violations_dummy += 1
                            t += 1
                        if violations_dummy > 0:
                            violations_per_slot += violations_dummy*weight[o]*number_occurence_scenario[np.where(np.all(failure_scenarios_unique == c, axis=1))[0][0]]
        solution[slot].append(violations_per_slot) # add violations per scenario to each slot
    return solution

# Randomly sequence the vehicles and determine the number of violations
def random_sequence(failure_scenarios_unique, number_occurence_scenario, vehicles_minus_dummies, vehicles_plus_dummies, max_No, options, sequence_slots,sequence_slots_with_dummies,a_vo,sequence_rules,weights):
    # Randomly sequence the vehicles
    solution = []
    for v in vehicles_minus_dummies:
        solution.append([v])
    solution=np.array(solution)
    np.random.shuffle(solution)

    # Add back dummies
    for v in range(max_No):
        solution=np.append(solution,[["dummy_b{}".format(v)]],axis=0)

    # Add front dummies
    if len(vehicles_plus_dummies)>len(solution):
        for v in range(len(vehicles_plus_dummies)-len(solution)):
            solution=np.insert(solution,0,[["dummy_f{}".format(v)]],axis=0)
    solution = solution.tolist()

    print("Order of random sequence: ", solution)

    # Determine the number of violations for each slot in each remove scenario
    for slot in range(len(sequence_slots_with_dummies)):
        violations_per_slot = 0
        if slot <= np.where(sequence_slots_with_dummies == sequence_slots[-1])[0][
            0]:  # if vehicle is smaller than the back dummies
            for c in failure_scenarios_unique:
                if c[np.where(vehicles_plus_dummies == solution[slot][0])[0][
                    0]] == 1:  # if vehicle at slot t does not fail in scenario c you may count the violations
                    for o in range(len(options)):
                        violations_dummy = -1 * sequence_rules[options[o]][0]
                        dummy = 0
                        t = slot
                        while dummy < sequence_rules[options[o]][1]:
                            if c[np.where(vehicles_plus_dummies == solution[t][0])[0][
                                0]] == 1:  # if vehicle at slot t does not fails in scenario c do:
                                dummy += 1
                                if a_vo[np.where(vehicles_plus_dummies == solution[t][0])[0][0]][
                                    o] == 1:  # if vehicle at slot t requirs option o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             do:
                                    violations_dummy += 1
                            t += 1
                        if violations_dummy > 0:
                            violations_per_slot +=  weights[o] * violations_dummy * number_occurence_scenario[
                                np.where(failure_scenarios_unique == c)[0][0]]
        solution[slot].append(violations_per_slot)  # add violations per scenario to each slot
    return solution


# Order sequence as in the file and determine the number of violations
def file_sequence(failure_scenarios_unique, number_occurence_scenario, vehicles_minus_dummies, vehicles_plus_dummies,
                  max_No, options, sequence_slots, sequence_slots_with_dummies, a_vo, sequence_rules, weights):
    # Randomly sequence the vehicles
    solution = []
    for v in vehicles_minus_dummies:
        solution.append([v])
    solution = np.array(solution)

    # Add back dummies
    for v in range(max_No):
        solution = np.append(solution, [["dummy_b{}".format(v)]], axis=0)

    # Add front dummies
    if len(vehicles_plus_dummies) > len(solution):
        for v in range(len(vehicles_plus_dummies) - len(solution)):
            solution = np.insert(solution, 0, [["dummy_f{}".format(v)]], axis=0)
    solution = solution.tolist()

    print("Order of file sequence: ", solution)

    # Determine the number of violations for each slot in each remove scenario
    for slot in range(len(sequence_slots_with_dummies)):
        violations_per_slot = 0
        if slot <= np.where(sequence_slots_with_dummies == sequence_slots[-1])[0][
            0]:  # if vehicle is smaller than the back dummies
            for c in failure_scenarios_unique:
                if c[np.where(vehicles_plus_dummies == solution[slot][0])[0][
                    0]] == 1:  # if vehicle at slot t does not fail in scenario c you may count the violations
                    for o in range(len(options)):
                        violations_dummy = -1 * sequence_rules[options[o]][0]
                        dummy = 0
                        t = slot
                        while dummy < sequence_rules[options[o]][1]:
                            if c[np.where(vehicles_plus_dummies == solution[t][0])[0][
                                0]] == 1:  # if vehicle at slot t does not fails in scenario c do:
                                dummy += 1
                                if a_vo[np.where(vehicles_plus_dummies == solution[t][0])[0][0]][
                                    o] == 1:  # if vehicle at slot t requirs option o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             do:
                                    violations_dummy += 1
                            t += 1
                        if violations_dummy > 0:
                            violations_per_slot += weights[o] * violations_dummy * number_occurence_scenario[
                                np.where(failure_scenarios_unique == c)[0][0]]
        solution[slot].append(violations_per_slot)  # add violations per scenario to each slot
    return solution