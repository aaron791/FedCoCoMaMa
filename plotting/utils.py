import numpy as np


def downsample_for_errorbars(std: np.ndarray, num_rounds: int, n_points: int) -> np.ndarray:
    """Gibt std-Array zurück, bei dem nur n_points gleichmäßig verteilte Stellen != None sind.

    Verhindert, dass Errorbars bei jedem Datenpunkt gezeichnet werden (zu viel Clutter).
    Erster und letzter Punkt sind immer enthalten.
    """
    result = std.astype(object).copy()
    step = int(num_rounds / n_points)
    for i in range(len(std)):
        if i != 0 and i % step != 0 and i != len(std) - 1:
            result[i] = None
    return result
