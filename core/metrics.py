"""Log-time weights and switching accuracies. Definitions are in the README."""
import torch
import numpy as np

NO_SWITCH_RELATIVE_RANGE = 0.1  # |final - start| / max(|start|, |final|) at or below this: no transition
FINAL_WINDOW = 0.1               # trailing fraction of the time axis averaged for the final current
FINAL_CURRENT_TOLERANCES = (5.0, 10.0)
T90_TOLERANCE = 0.25             # decades; |log10| of the ratio of the 90 % crossing times
ACCURACIES = (
    ('final_current_accuracy_5', 'final_current', FINAL_CURRENT_TOLERANCES[0]),
    ('final_current_accuracy_10', 'final_current', FINAL_CURRENT_TOLERANCES[1]),
    ('t90_accuracy', 't90_dex', T90_TOLERANCE),
)


def log_time_weights(time_real_full):
    """Per-point weights for time_real_full[2:], proportional to the log-time span each point covers (mean 1)."""
    weights = torch.diff(torch.log(time_real_full))[1:].clamp_min(0)
    return weights / weights.mean()


def _crossing(time, current, level, increasing):
    hit = np.flatnonzero(current >= level if increasing else current <= level)
    if not len(hit):
        return None
    k = int(hit[0])
    if k == 0:
        return float(time[0])
    fraction = (level - current[k - 1]) / (current[k] - current[k - 1])
    if time[k - 1] == 0:
        return float(fraction * time[k])
    return float(np.exp(np.log(time[k - 1]) + fraction * (np.log(time[k]) - np.log(time[k - 1]))))


def transition(time, current):
    """Start, final level and 90 % crossing time of one |current| trace. `switches` is false when the trace barely changes."""
    time, current = np.asarray(time, dtype=float), np.abs(np.asarray(current, dtype=float))
    if (time.ndim != 1 or time.shape != current.shape or len(time) < 2 or time[0] < 0 or not (time[1:] > 0).all()
            or (np.diff(time) <= 0).any() or not np.isfinite(current).all()):
        raise ValueError('transition requires increasing nonnegative times and matching finite currents')
    start, final = float(current[0]), float(current[time >= (1 - FINAL_WINDOW) * time[-1]].mean())
    switches = abs(final - start) > NO_SWITCH_RELATIVE_RANGE * max(abs(start), abs(final))
    t90 = _crossing(time, current, start + 0.9 * (final - start), final > start) if switches else None
    return dict(start=start, final=final, switches=bool(switches), t90=t90)


def switching_errors(time, observed, predicted):
    """t90 error in decades and final-current error in percent. None when that trace defines no transition."""
    reference, candidate = transition(time, observed), transition(time, predicted)
    result = dict(eligible=reference['switches'], t90_dex=None,
                  final_current=None if reference['final'] == 0 else float(abs(candidate['final'] - reference['final']) / reference['final'] * 100),
                  observed=reference, predicted=candidate)
    if reference['switches'] and candidate['switches']:
        result['t90_dex'] = float(abs(np.log10(candidate['t90'] / reference['t90'])))
    return result


def mean_relative_error(pred, true):
    """Mean pointwise relative error of one sequence, in percent."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred - true) / (np.abs(true) + 1e-10)) * 100)


def accuracy_percents(records):
    """Accuracies in percent. A miss counts as inaccurate; 0 when a metric has no records."""
    values = {
        'final_current': [r['final_current'] for r in records],
        't90_dex': [r['t90_dex'] for r in records if r['eligible']],
    }
    out = {}
    for key, metric, tolerance in ACCURACIES:
        group = values[metric]
        out[key] = 0.0 if not group else 100.0 * sum(v is not None and v <= tolerance for v in group) / len(group)
    return out
