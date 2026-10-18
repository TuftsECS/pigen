"""Switching accuracies, log errors and the checkpoint selection score. Definitions are in the README."""
import numpy as np

NO_SWITCH_RELATIVE_RANGE = 0.1  # |final - start| / max(|start|, |final|) at or below this: no transition
FINAL_WINDOW = 0.1               # trailing fraction of the time axis averaged for the final current
FINAL_CURRENT_TOLERANCES = (5.0, 10.0)
T90_TOLERANCE = 0.25             # decades; |log10| of the ratio of the 90 % crossing times
CURRENT_FLOOR = 1e-12            # A; the log errors read a smaller |current| at this level
SELECTION_CAP = 4.0              # balanced score: an error counts at most this many tolerances, a miss or false switch this much


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


def time_balanced_weights(time, reference):
    """Trapezoidal point weights summing to one, half in t and half in log1p(t / reference)."""
    time = np.asarray(time, dtype=float)
    delta = np.diff(time)
    halves = []
    for intervals in (delta, np.log1p(delta / (time[:-1] + reference))):
        normalized = intervals / intervals.sum()
        halves.append(np.concatenate((normalized[:1], normalized[:-1] + normalized[1:], normalized[-1:])) / 2)
    return (halves[0] + halves[1]) / 2


def log_errors(time, observed, predicted):
    """Time-balanced and plain mean |log10 I| errors and the linear relative error of one trace whose plateau starts at time[0]."""
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    error = np.abs(np.log10(np.maximum(np.abs(predicted), CURRENT_FLOOR)) - np.log10(np.maximum(np.abs(observed), CURRENT_FLOOR)))
    return dict(time_balanced_logmae=float((time_balanced_weights(time, time[0]) * error).sum()), logmae=float(error.mean()),
                linear_relative_error=float(np.abs(predicted - observed).sum() / max(np.abs(observed).sum(), CURRENT_FLOOR)))


def _capped(value, tolerance):
    return SELECTION_CAP if value is None else min(value / tolerance, SELECTION_CAP)


def balanced_score(records):
    """Selection score (lower is better): capped final-current and t90 errors in tolerances, averaged per record, then within and
    across the switching and non-switching records."""
    def error(r):
        final = _capped(r['final_current'], FINAL_CURRENT_TOLERANCES[1])
        if not r['eligible']:
            return (final + (SELECTION_CAP if r['predicted']['switches'] else 0.0)) / 2
        return (final + _capped(r['t90_dex'], T90_TOLERANCE)) / 2
    classes = [[error(r) for r in records if r['eligible'] == switching] for switching in (True, False)]
    return float(np.mean([np.mean(errors) for errors in classes if errors]))


def joint_pass(record):
    """Final current within 10 % and t90 within 0.25 decade; a non-switching record passes only if no switch is predicted."""
    if record['final_current'] is None or record['final_current'] > FINAL_CURRENT_TOLERANCES[1]:
        return False
    if not record['eligible']:
        return not record['predicted']['switches']
    return record['t90_dex'] is not None and record['t90_dex'] <= T90_TOLERANCE


def summarize_records(groups):
    """Validation metrics of {key: [(switching_errors record, log_errors dict), ...]}: accuracies (percent) and log errors averaged
    equally over keys; joint pass, missed and false switch rates (percent, None without such records) and the balanced score pooled."""
    def share(flags):
        return 100.0 * sum(flags) / len(flags) if flags else None

    per_key = []
    for rows in groups.values():
        records = [r for r, _ in rows]
        per_key.append(dict(
            **{f'final_current_accuracy_{int(t)}': share([r['final_current'] is not None and r['final_current'] <= t for r in records])
               for t in FINAL_CURRENT_TOLERANCES},
            t90_accuracy=share([r['t90_dex'] is not None and r['t90_dex'] <= T90_TOLERANCE for r in records if r['eligible']]),
            **{name: float(np.mean([e[name] for _, e in rows])) for name in ('time_balanced_logmae', 'logmae', 'linear_relative_error')}))
    out = {name: float(np.mean([k[name] for k in per_key if k[name] is not None])) for name in per_key[0]}
    records = [r for rows in groups.values() for r, _ in rows]
    out.update(joint_pass=share([joint_pass(r) for r in records]),
               missed_rate=share([not r['predicted']['switches'] for r in records if r['eligible']]),
               false_rate=share([r['predicted']['switches'] for r in records if not r['eligible']]),
               balanced_score=balanced_score(records))
    return out
