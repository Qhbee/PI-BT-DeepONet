"""Case registry for data generators."""

from collections.abc import Callable


GeneratorFn = Callable[..., dict]
GENERATORS: dict[str, GeneratorFn] = {}


def register(case: str) -> Callable[[GeneratorFn], GeneratorFn]:
    """Decorator to register a generator function by case name."""

    def decorator(fn: GeneratorFn) -> GeneratorFn:
        GENERATORS[case] = fn
        return fn

    return decorator


def get_generator(case: str) -> GeneratorFn:
    """Fetch registered generator by case name."""
    try:
        return GENERATORS[case]
    except KeyError as exc:
        known = ", ".join(sorted(GENERATORS.keys()))
        raise ValueError(f"Unknown case: {case}. Available: {known}") from exc
