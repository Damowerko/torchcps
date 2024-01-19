import inspect


def get_init_arguments_and_types(cls):
    """

    Args:
        cls: class to get init arguments from

    Returns:
        list of tuples (name, type, default)
    """
    parameters = inspect.signature(cls).parameters
    args = []
    for name, parameter in parameters.items():
        args.append((name, parameter.annotation, parameter.default))
    return args


def add_model_specific_args(cls, group):
    for base in cls.__bases__:
        if hasattr(base, "add_model_specific_args"):
            group = base.add_model_specific_args(group)  # type: ignore
    args = get_init_arguments_and_types(cls)  # type: ignore
    for name, type, default in args:
        if default is inspect.Parameter.empty:
            continue
        if type not in (int, float, str, bool):
            continue
        if type == bool:
            group.add_argument(f"--{name}", dest=name, action="store_true")
        else:
            group.add_argument(f"--{name}", type=type, default=default)
    return group
