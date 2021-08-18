import copy
import numpy as np
import math

class RCSP():
    """
    Solution class for the Robust Car Sequence Problem (RCSP).
    It has two data member: sequence_slots, scenarios.

    The first entry in sequence_slots represents the first slot, the second entry the second slot, etc.
    Each entry contains an array
    sequence_slots[t][0]: vehicle number at slot t
    sequence_slots[t][1]: total violations of all considered scenarios at slot t

    Scnearios consist of two elements; the unique failure scnearios and the occurenct
     scenario[0][c]: unique failure scenario c
     scenario[1][c]: the number of occurrence of unique failure scenario c
    """

    def __init__(self, sequence_slots,scenarios):
        self.sequence_slots = sequence_slots
        self.scenarios=scenarios

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.sequence_slots)

    def objective(self):
        """
            The objective function is the average expected violations of all considered failure scenarios
            Y_t = total number of violations in slot t
            Objective = sum Y_c/(number of considered failure scenarios)
        """

        return sum(self.sequence_slots[slot][1] for slot in range(len(self.sequence_slots)))/sum(self.scenarios[1])

    def test_objective(self, sequence_slots_with_dummies,vehicles_plus_dummies,max_No,options,rules,a_vo,weights, unq_test_scenarios, cnt_unq_test_scenarios):
        """
            The objective function is the average violations of all considered failure scenarios
            Y_t = total number of violations in slot t
            Objective = sum Y_c/(number of considered failure scenarios)

            At the start the number of violations per test scenario is unknown. Therfore, we first calculate the
            number of violations
        """

        new = self.copy()

        # Determine the number of violations for each slot in each remove scenario
        for slot in range(len(sequence_slots_with_dummies)):
            if slot <= len(vehicles_plus_dummies) - max_No:  # if vehicle is smaller than the back dummies
                violations_per_slot = 0
                for c in unq_test_scenarios:
                    if c[np.where(vehicles_plus_dummies == self.sequence_slots[int(slot)][0])[0][
                        0]] == 1:  # if vehicle at slot t does not fail in scenario c you may count the violations
                        for o in range(len(options)):
                            violations_dummy = -1 * rules[options[o]][0]
                            dummy = 0
                            t = int(slot)
                            while dummy < rules[options[o]][1]:
                                if c[np.where(vehicles_plus_dummies == self.sequence_slots[t][0])[0][
                                    0]] == 1:  # if vehicle at slot t does not fails in scenario c do:
                                    dummy += 1
                                    if a_vo[np.where(vehicles_plus_dummies == self.sequence_slots[t][0])[0][0]][
                                        o] == 1:  # if vehicle at slot t requirs option o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             do:
                                        violations_dummy += 1
                                t += 1
                            if violations_dummy > 0:
                                violations_per_slot += weights[o] * violations_dummy * cnt_unq_test_scenarios[
                                    np.where(np.all(unq_test_scenarios == c, axis=1))[0][0]]
                new.sequence_slots[int(slot)][1] = violations_per_slot  # update violations per slot
        return sum(new.sequence_slots[slot][1] for slot in range(len(self.sequence_slots))) / sum(cnt_unq_test_scenarios)


    def objective_without_removals(self,max_No,options,sequence_rules,vehicles_plus_dummies,a_vo,weights):
        """
            The number of violations when no vehicles are removed (ideal case)
            Y_t = total number of violations in slot t
            Number of violations in ideal case= sum Y_c/(number of considered failure scenarios)
        """

        # Determine number of violations for each slot
        violations=0
        for slot in range(len(self.sequence_slots)):
            violations_per_slot = 0
            if slot <= len(self.sequence_slots)-max_No:  # if vehicle is smaller than the back dummies
                for o in range(len(options)):
                    violations_dummy = -1 * sequence_rules[options[o]][0]
                    t = slot
                    dummy = 0
                    while dummy < sequence_rules[options[o]][1]:
                        if a_vo[np.where(vehicles_plus_dummies == self.sequence_slots[t][0])[0][0]][o] == 1.0:  # if vehicle at slot t requirs option o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       do:
                            violations_dummy += 1
                        dummy += 1
                        t+=1
                    if violations_dummy > 0:
                        violations_per_slot += weights[o] * violations_dummy
            violations+=violations_per_slot
        return violations

    def fails_in_a_scenario(self,v):
        """
        Does vehicle v fails in any failure scenario?
        """
        fails_in_a_scenario = False
        for c in self.scenarios[0]:
            if c[v] == 0:
                fails_in_a_scenario = True
        return fails_in_a_scenario

    def update_violations(self, update_slots, vehicles_plus_dummies, a_vo, options, rules,weights, max_No):
        """
        Update violations for specific slots for each scenario c
        """
        new = self.copy()

        for update_slot in update_slots:
            if update_slot <= len(vehicles_plus_dummies) - max_No:  # if vehicle is smaller than the back dummies
                violations_per_slot = 0
                for c in self.scenarios[0]:
                    if c[np.where(vehicles_plus_dummies == self.sequence_slots[int(update_slot)][0])[0][
                        0]] == 1:  # if vehicle at slot t does not fail in scenario c you may count the violations
                        for o in range(len(options)):
                            violations_dummy = -1 * rules[options[o]][0]
                            dummy = 0
                            t = int(update_slot)
                            while dummy < rules[options[o]][1]:
                                if c[np.where(vehicles_plus_dummies == self.sequence_slots[t][0])[0][
                                    0]] == 1:  # if vehicle at slot t does not fails in scenario c do:
                                    dummy += 1
                                    if a_vo[np.where(vehicles_plus_dummies == self.sequence_slots[t][0])[0][0]][
                                        o] == 1:  # if vehicle at slot t requirs option o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             do:
                                        violations_dummy += 1
                                t += 1
                            if violations_dummy > 0:
                                violations_per_slot += weights[o] * violations_dummy * self.scenarios[1][
                                    np.where(np.all(self.scenarios[0] == c, axis=1))[0][0]]
                new.sequence_slots[int(update_slot)][1] = violations_per_slot  # update violations per slot
        return new

    def update_which_slots(self, changed_indexes, max_No, vehicles_with_dummies):
        """
        Which slots need to be updated?
        """

        update_slots = np.array([])
        for i in changed_indexes:
            i = int(i)
            dummy = 1
            j = i
            while dummy <= max_No and j > 0:
                j -= 1
                if not self.fails_in_a_scenario(np.where(vehicles_with_dummies == self.sequence_slots[j][0])[0][0]):
                    dummy += 1
            update_slots = np.concatenate((update_slots, np.array([x for x in range(j, i + 1)])))
        update_slots = np.unique(update_slots)

        return update_slots

    def order(self):
        """
        Return the order of the vehicles
        """
        order_vehicles=[]
        for i in range(len(self.sequence_slots)):
            order_vehicles.append(self.sequence_slots[i][0])
        return order_vehicles

    def determine_start_temperature(self, solution_that_is_x_percentage_worse_than_initial, probability_of_accepting):
        """
        Return start temperature
        """
        return abs(solution_that_is_x_percentage_worse_than_initial/np.log(probability_of_accepting)*self.objective())

    def sequence_define_full_or_slack_for_option(self, vehicles_plus_dummies, option_index, H_o, N_o,a):
        """
        Return [-1,0,1,2] array called full_slack.
        full_slack[t]=-1 if slot t is not in any sequence
        full_slack[t]=0 if slot t is in slack sequence
        full_slack[t]=1 if slot t is in full sequence
        full_slack[t]=2 if slot t is in overfull sequence
        """

        full_slack=[]
        not_in_full_or_slack = True
        dummy=0 # counts the number of vehicles with option o in subsequence
        dummy2=0 # counts the number of vehicles in subsequence
        for t in range(len(self.sequence_slots)):
            dummy2 += 1
            if a[np.where(vehicles_plus_dummies == self.sequence_slots[t][0])[0][0]][option_index] == 1:
                dummy += 1
            if not_in_full_or_slack:
                if dummy == 1:
                    not_in_full_or_slack = False
                    full_slack.append([-1]*(dummy2-1)) # not full and slack
                    dummy2=1
            else:
                if dummy > H_o:
                    if N_o == dummy2-1:
                        full_slack.append([1] * (dummy2-1))  # full
                    elif N_o > dummy2-1:
                        full_slack.append([2] * (dummy2-1))  # overfull
                    else:
                        full_slack.append([0] * (dummy2-1))  # slack
                    dummy = 1
                    dummy2 = 1

        # Add last subsequence
        if math.floor(sum([a[i][option_index] for i in range(len(a))])/H_o)<sum([a[i][option_index] for i in range(len(a))])/H_o:
            full_slack.append([-1] * dummy2) # not full and slack
        else:
            if N_o-H_o==dummy2:
                full_slack.append([1] * dummy2) # full
            elif N_o - H_o > dummy2:
                full_slack.append([1] * dummy2)  # overfull
            else:
                full_slack.append([0] * dummy2) # slack
        return list(np.concatenate(full_slack).flat)

    def still_slack_subsequence_if_you_replace(self, remove_index, replace_index, vehicles_plus_dummies, option_index, H_o, N_o,a):
        """
               Return if the sub-sequence is still a slack sub-sequence when a vehicle is re-planned
        """

        change=self.copy()

        remove_element=change.sequence_slots[remove_index]
        del change.sequence_slots[remove_index]
        change.sequence_slots.insert(replace_index, remove_element)

        still_slack=False
        subsequence=change.sequence_define_full_or_slack_for_option(vehicles_plus_dummies, option_index, H_o, N_o,a)

        if subsequence[replace_index]<=0:
            still_slack=True
        return still_slack



