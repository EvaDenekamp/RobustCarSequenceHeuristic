# This file is the adjusted main file of the packages alns (https://github.com/N-Wouda/ALNS).
# The adjustments: score, weight and probability updating; destroyed and removed operators are combined; extra stopping criteria

import warnings
from collections import OrderedDict
import numpy as np
import numpy.random as rnd
from Result import Result
from alns.State import State  # pylint: disable=unused-import
from alns.Statistics import Statistics
from alns.criteria import AcceptanceCriterion  # pylint: disable=unused-import
from alns.select_operator import select_operator
from alns.tools.warnings import OverwriteWarning
import time

# Weights
_IS_BEST = 0
_IS_BETTER = 1
_IS_ACCEPTED = 2
_IS_REJECTED = 3

# Callbacks
_ON_BEST = 0


class ALNS:

    def __init__(self, rnd_state=rnd.RandomState()):
        """
        Implements the adaptive large neighbourhood search (ALNS) algorithm.
        The implementation optimises for a minimisation problem, as explained
        in the text by Pisinger and Røpke (2010).

        Parameters
        ----------
        rnd_state : rnd.RandomState
            Optional random state to use for random number generation. When
            passed, this state is used for operator selection and general
            computations requiring random numbers. It is also passed to the
            destroy and repair operators, as a second argument.

        References
        ----------
        - Pisinger, D., and Røpke, S. (2010). Large Neighborhood Search. In M.
          Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399-420).
          Springer.
        """
        super().__init__()

        self._operators = OrderedDict()
        self._callbacks = {}

        self._rnd_state = rnd_state

    @property
    def operators(self):
        """
        Returns the operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._operators.items())

    def add_operator(self, operator, name=None):
        """
        Adds an operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. The second argument
            is the random state constructed from the passed-in seed.
        name : str
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._operators, operator, name)

    def iterate(self, initial_solution, reward, operator_decay, criterion,
                collect_stats=True, limited_time=False, limit_time=60, iteration_check_converges=5000):
        """
        Runs the adaptive large neighbourhood search heuristic [1], using the
        previously set operators. The first solution is set
        to the passed-in initial solution, and then subsequent solutions are
        computed by iteratively applying the operators.

        Parameters
        ----------
        initial_solution : State
            The initial solution, as a State object.
        reward: array_like
            A list of four non-negative elements, representing the reward
             when the candidate solution results in a new global best
            (idx 0), is better than the current solution (idx 1), the solution
            is accepted (idx 2), or rejected (idx 3).
        operator_decay : float
            The operator decay parameter, as a float in the unit interval,
            [0, 1] (inclusive).
        criterion : AcceptanceCriterion
            The acceptance criterion to use for candidate states. See also
            the `alns.criteria` module for an overview.
        collect_stats : bool
            Should statistics be collected during iteration? Default True, but
            may be turned off for long runs to reduce memory consumption.
        limited_time: bool
            Extra stopping criteria that limits the compuation time? Default False.
        limit_time: float
            When extra stopping criteria limited_time is true, the time is limited
            to limit_time seconds. Defeault is 1 minute.
        iteration_check_converges: float
            After a module of iteration_check_converges the heuristic checks if
            the best solution is the not changed more than 0.1%

        Returns
        -------
        Result
            A result object, containing the best solution and some additional
            statistics.

        References
        ----------
        [1]: Pisinger, D., & Røpke, S. (2010). Large Neighborhood Search. In M.
        Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399-420).
        Springer.

        [2]: S. Røpke and D. Pisinger (2006). A unified heuristic for a large
        class of vehicle routing problems with backhauls. *European Journal of
        Operational Research*, 171: 750–775, 2006.
        """
        reward = np.asarray(reward, dtype=np.float16)
        current = best = initial_solution

        d_weights = np.ones(len(self.operators), dtype=np.float16)
        d_prob = np.array([])
        p_min=0.05
        for i in range(len(d_weights)):
            d_prob = np.append(d_prob,p_min+d_weights[i]/sum(d_weights)*(1-len(d_weights)*p_min))
        d_score = np.zeros(len(self.operators), dtype=np.float16)
        statistics = Statistics()

        if collect_stats:
            statistics.collect_objective(initial_solution.objective())

        iteration=0
        obj_old_best=best.objective()
        prev_obj = best.objective()
        start_time=time.time()
        while True:
            iteration+=1
            d_idx = select_operator(self.operators, d_prob,
                                    self._rnd_state)

            d_name, d_operator = self.operators[d_idx]
            candidate = d_operator(current, self._rnd_state)

            best, current, score_idx = self._consider_candidate(best,
                                                                 current,
                                                                 candidate,
                                                                 criterion)

            d_score[d_idx] += reward[score_idx] # the score is updated

            if collect_stats:
                statistics.collect_objective(current.objective())
                statistics.collect_destroy_operator(d_name, score_idx)

            if iteration%100==0: # after a segment, the scores are reset and the weights are updated
                d_weights = (1-operator_decay)*d_weights+operator_decay*d_score
                for i in range(len(d_weights)):
                    d_prob[i] =p_min + d_weights[i] / sum(d_weights) * (1 - len(d_weights) * p_min)
                d_score = np.zeros(len(self.operators), dtype=np.float16)
                if prev_obj!=best.objective():
                    prev_obj=best.objective()
            if iteration%iteration_check_converges==0: # stop when the best solution converge
                obj_best=best.objective()
                if (obj_old_best-obj_best)/obj_old_best<0.001:
                    return Result(best, statistics if collect_stats else None)
                obj_old_best=best.objective()
            if best.objective()==0: # stop because the solution can not be reduced lower than zero
                return Result(best, statistics if collect_stats else None)
            if limited_time:
                if limit_time <= time.time() - start_time: # stop because the time limit is reached
                    return Result(best, statistics if collect_stats else None)
        return Result(best, statistics if collect_stats else None)

    def on_best(self, func):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.

        Parameters
        ----------
        func : callable
            A function that should take a solution State as its first parameter,
            and a numpy RandomState as its second (cf. the operator signature).
            It should return a (new) solution State.

        Warns
        -----
        OverwriteWarning
            When a callback has already been set.
        """
        self._set_callback(_ON_BEST, func)

    @staticmethod
    def _add_operator(operators, operator, name=None):
        """
        Internal helper that adds an operator to the passed-in operator
        dictionary. See `add_destroy_operator` and `add_repair_operator` for
        public methods that use this helper.

        Parameters
        ----------
        operators : dict
            Dictionary of (name, operator) key-value pairs.
        operator : Callable[[State, RandomState], State]
            Callable operator function.
        name : str
            Optional operator name.

        Warns
        -----
        OverwriteWarning
            When the operator name already maps to an operator on this ALNS
            instance.
        """
        if name is None:
            name = operator.__name__

        if name in operators:
            warnings.warn("The ALNS instance already knows an operator by the"
                          " name `{0}'. This operator will now be replaced with"
                          " the newly passed-in operator. If this is not what"
                          " you intended, consider explicitly naming your"
                          " operators via the `name' argument.".format(name),
                          OverwriteWarning)

        operators[name] = operator

    def _consider_candidate(self, best, current, candidate, criterion):
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Returns the new solution when it is better or
        accepted, or the current in case it is rejected. Candidate solutions
        are accepted based on the passed-in acceptance criterion.

        Parameters
        ----------
        best : State
            Best solution encountered so far.
        current : State
            Current solution.
        candidate : State
            Candidate solution.
        criterion : AcceptanceCriterion
            The chosen acceptance criterion.

        Returns
        -------
        State
            The (possibly new) best state.
        State
            The (possibly new) current state.
        int
            The weight index to use when updating the operator weights.
        """
        if criterion.accept(self._rnd_state, best, current, candidate):
            if candidate.objective() < current.objective():
                weight = _IS_BETTER
            else:
                weight = _IS_ACCEPTED

            current = candidate
        else:
            weight = _IS_REJECTED

        if candidate.objective() < best.objective():
            # Is a new global best, so we might want to do something to further
            # improve the solution.
            # if _ON_BEST in self._callbacks:
            #     callback = self._callbacks[_ON_BEST]
            #     candidate = callback(candidate, self._rnd_state)

            # Global best solution becomes the new starting point for further
            # iterations.
            # print(candidate.objective(), best.objective())
            return candidate, candidate, _IS_BEST

        # Best has not been updated if we get here, but the current state might
        # have (if the candidate was accepted).
        return best, current, weight

    def _validate_parameters(self, weights, operator_decay, iterations):
        """
        Helper method to validate the passed-in ALNS parameters.
        """
        if len(self.destroy_operators) == 0 or len(self.repair_operators) == 0:
            raise ValueError("Missing at least one destroy or repair operator.")

        if not (0 <= operator_decay <= 1):
            raise ValueError("Operator decay parameter outside unit interval"
                             " is not understood.")

        if any(weight < 0 for weight in weights):
            raise ValueError("Negative weights are not understood.")

        if len(weights) < 4:
            # More than four is not explicitly problematic, as we only use the
            # first four anyways.
            raise ValueError("Unsupported number of weights: expected 4,"
                             " found {0}.".format(len(weights)))

        if iterations < 0:
            raise ValueError("Negative number of iterations.")

    def _set_callback(self, flag, func):
        """
        Sets the passed-in callback func for the passed-in flag. Warns if this
        would overwrite an existing callback.
        """
        if flag in self._callbacks:
            warnings.warn("A callback function has already been set for the"
                          " `{0}' flag. This callback will now be replaced by"
                          " the newly passed-in callback.".format(flag),
                          OverwriteWarning)

        self._callbacks[flag] = func