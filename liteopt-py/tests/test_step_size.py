"""Step size uses the same public name across all solver entry points."""
import liteopt
import pytest


def solve(entry, options):
    if entry == "gd":
        return liteopt.gd(
            lambda x: 0.5 * (x[0] - 1.) ** 2,
            lambda x: [x[0] - 1.],
            [0.], options=options,
        )
    kwargs = dict(jacobian=lambda x: [1.], options=options)
    residual = lambda x: [x[0] - 1.]
    if entry.startswith("least_squares:"):
        return liteopt.least_squares(residual, [0.], method=entry.split(":")[1], **kwargs)
    return getattr(liteopt, entry)(residual, [0.], **kwargs)


ENTRIES = ["gd", "gn", "lm", "least_squares:gn", "least_squares:lm"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_step_size_controls_the_first_update(entry):
    small = solve(entry, {"step_size": 0.25, "max_iters": 1})
    large = solve(entry, {"step_size": 0.5, "max_iters": 1})
    # Gradient descent uses -grad; GN/LM use the damped linear direction.
    direction = 1. / 1.001 if entry in ("lm", "least_squares:lm") else 1.
    assert small[0][0] == pytest.approx(0.25 * direction)
    assert large[0][0] == pytest.approx(0.5 * direction)
    assert large[1] < small[1]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("include_new_name", [False, True])
def test_old_step_scale_name_is_rejected(entry, include_new_name):
    options = {"step_scale": 0.5}
    if include_new_name:
        options["step_size"] = 0.5
    with pytest.raises(ValueError, match="unknown options key 'step_scale'"):
        solve(entry, options)
