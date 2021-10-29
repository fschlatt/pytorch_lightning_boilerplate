import argparse
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Union, overload


class ArgumentConflictSolver:
    def __init__(self) -> None:
        self.conflicting_args = defaultdict(set)

    def catch_conflicting_args(self, add_argument: Callable) -> Callable:
        def wrapper(parser, option_string, *args, **kwargs):
            try:
                return add_argument(parser, option_string, *args, **kwargs)
            except argparse.ArgumentError as err:
                confl_optional = parser._option_string_actions[option_string]
                self.conflicting_args[option_string].add(confl_optional)
                self.conflicting_args[option_string].add(err.args[0])

        return wrapper

    @overload
    def resolve_conflicting_args(
        self,
        parser: argparse.ArgumentParser,
        help_strings: Optional[Dict[str, str]] = None,
    ) -> argparse.ArgumentParser:
        ...

    @overload
    def resolve_conflicting_args(
        self,
        parser: argparse._ArgumentGroup,
        help_strings: Optional[Dict[str, str]] = None,
    ) -> argparse._ArgumentGroup:
        ...

    def resolve_conflicting_args(
        self,
        parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],
        help_strings: Optional[Dict[str, str]] = None,
    ) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        parser = self.remove_conflicting_args(parser)
        if help_strings is None:
            help_strings = {}

        def same(elems: Iterable) -> bool:
            return len(set(elems)) <= 1

        for option_string, actions in self.conflicting_args.items():
            choices = [action.choices for action in actions]
            const = [action.const for action in actions]
            dest = [action.dest for action in actions]
            metavar = [action.metavar for action in actions]
            nargs = [action.nargs for action in actions]
            required = [action.required for action in actions]
            types = [action.type for action in actions]
            if all(
                same(variables)
                for variables in (choices, const, dest, metavar, nargs, required, types)
            ):
                parser.add_argument(
                    option_string,
                    nargs=nargs[0],
                    const=const[0],
                    dest=dest[0],
                    type=types[0],
                    choices=choices[0],
                    required=required[0],
                    help=help_strings.get(option_string, ""),
                )
        return parser

    @overload
    def remove_conflicting_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        ...

    @overload
    def remove_conflicting_args(
        self, parser: argparse._ArgumentGroup
    ) -> argparse._ArgumentGroup:
        ...

    def remove_conflicting_args(
        self, parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]
    ) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # remove all conflicting options
        for option_string, actions in self.conflicting_args.items():
            for action in actions:

                # remove the conflicting option
                action.option_strings.remove(option_string)
                parser._option_string_actions.pop(option_string, None)

                # if the option now has no option string, remove it from the
                # container holding it
                if not action.option_strings:
                    if hasattr(action, "container"):
                        action.container._remove_action(action)
        return parser


def get_argument_names(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]
) -> List[str]:
    arguments = []
    for action in parser._actions:
        arguments.extend(action.option_strings)
    return arguments
