"""Every openrapm command's help formats: a bare percent sign in a help string breaks only that command's --help."""
import argparse

from openrapm.commands.cli import build_parser


def walk(parser, name="openrapm"):
    parser.format_help()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub_name, sub in action.choices.items():
                yield from walk(sub, f"{name} {sub_name}")
    yield name


names = list(walk(build_parser()))
print(f"{len(names)} commands' help formats")
