"""Viterbi over variable candidate sets; all inputs are natural-log scores."""

from math import isfinite


class NoValidPath(ValueError):
    """No complete sequence exists under the supplied scores."""


def decode(emissions, transitions):
    """Return candidate indices and total score.

    transitions[t][i][j] connects candidate i at t to candidate j at t+1.
    Negative infinity means impossible. Ties choose the first candidate.
    """
    if len(transitions) != max(0, len(emissions) - 1):
        raise ValueError("Expected one transition matrix per adjacent observation pair")
    if not emissions:
        return [], 0.0
    for row in emissions:
        if not row:
            raise NoValidPath("Observation has no candidates")
        if any(not isfinite(x) and x != float('-inf') for x in row):
            raise ValueError("Scores must be finite or negative infinity")
    scores = list(emissions[0])
    pointers = []
    for t, matrix in enumerate(transitions):
        if len(matrix) != len(scores) or any(len(r) != len(emissions[t + 1]) for r in matrix):
            raise ValueError("Transition matrix does not match candidate counts")
        if any(not isfinite(x) and x != float('-inf') for r in matrix for x in r):
            raise ValueError("Scores must be finite or negative infinity")
        previous, updated = [], []
        for j, emission in enumerate(emissions[t + 1]):
            options = [score + matrix[i][j] for i, score in enumerate(scores)]
            best = max(range(len(options)), key=options.__getitem__)
            previous.append(best)
            updated.append(options[best] + emission)
        pointers.append(previous)
        scores = updated
    state = max(range(len(scores)), key=scores.__getitem__)
    total = scores[state]
    if not isfinite(total):
        raise NoValidPath("No reachable complete sequence")
    path = [state]
    for previous in reversed(pointers):
        state = previous[state]
        path.append(state)
    return list(reversed(path)), total
