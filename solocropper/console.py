# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz

import os
import sys


ANSI_RESET = "\033[0m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"


def enable_terminal_colors():
    if not sys.stdout.isatty() and not sys.stderr.isatty():
        return False

    if os.name != "nt":
        return True

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        enable_flag = 0x0004
        handles = (-11, -12)
        enabled = False

        for handle_id in handles:
            handle = kernel32.GetStdHandle(handle_id)
            if handle == 0 or handle == -1:
                continue

            mode = ctypes.c_uint()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue

            if kernel32.SetConsoleMode(handle, mode.value | enable_flag):
                enabled = True

        return enabled
    except Exception:
        return False


TERMINAL_COLORS_ENABLED = enable_terminal_colors()


def color_text(text, color_code):
    if not TERMINAL_COLORS_ENABLED:
        return text
    return f"{color_code}{text}{ANSI_RESET}"


def print_notice(message):
    print(color_text(message, ANSI_YELLOW))


def print_error(message):
    print(color_text(message, ANSI_RED))
