from colorama import Fore, Style
from colorama.ansi import AnsiFore, AnsiStyle

def log(message: str, ansi: AnsiFore | AnsiStyle = Fore.WHITE) -> None:
    """
    Logs a message to the console with a specified color.

    Args:
        message (str): The message to log.
        ansi (AnsiFore | AnsiStyle): The ANSI color code to use.
            Defaults to Fore.WHITE.
    """
    print(f"{ansi}{message}{Style.RESET_ALL}")

def log_red(message: str) -> None:
    """
    Logs a message to the console in red.

    Args:
        message (str): The message to log.
    """
    return f"{Fore.RED}{message}{Style.RESET_ALL}"

def log_green(message: str) -> None:
    """
    Logs a message to the console in green.

    Args:
        message (str): The message to log.
    """
    return f"{Fore.GREEN}{message}{Style.RESET_ALL}"

def log_yellow(message: str) -> None:
    """
    Logs a message to the console in yellow.

    Args:
        message (str): The message to log.
    """
    return f"{Fore.YELLOW}{message}{Style.RESET_ALL}"

def log_blue(message: str) -> None:
    """
    Logs a message to the console in blue.

    Args:
        message (str): The message to log.
    """
    return f"{Fore.BLUE}{message}{Style.RESET_ALL}"

def log_cyan(message: str) -> None:
    """
    Logs a message to the console in cyan.

    Args:
        message (str): The message to log.
    """
    return f"{Fore.CYAN}{message}{Style.RESET_ALL}"

def log_magenta(message: str) -> None:
    """
    Logs a message to the console in magenta.

    Args:
        message (str): The message to log.
    """
    return f"{Fore.MAGENTA}{message}{Style.RESET_ALL}"

def log_bold(message: str) -> None:
    """
    Logs a message to the console in bold.

    Args:
        message (str): The message to log.
    """
    return f"{Style.BRIGHT}{message}{Style.RESET_ALL}"

def log_info(message: str) -> None:
    """
    Logs an informational message to the console in white.

    Args:
        message (str): The informational message to log.
    """
    print(f"[ {log_bold(log_blue('INFO'))} ] {message}")

def log_error(message: str) -> None:
    """
    Logs an error message to the console in red.

    Args:
        message (str): The error message to log.
    """
    print(f"[ {log_bold(log_red('ERROR'))} ] {message}")

def log_success(message: str) -> None:
    """
    Logs a success message to the console in green.

    Args:
        message (str): The success message to log.
    """
    print(f"[ {log_bold(log_green('SUCCESS'))} ] {message}")

def log_warning(message: str) -> None:
    """
    Logs a warning message to the console in yellow.

    Args:
        message (str): The warning message to log.
    """
    print(f"[ {log_bold(log_yellow('WARNING'))} ] {message}")

def log_debug(message: str) -> None:
    """
    Logs a debug message to the console in cyan.

    Args:
        message (str): The debug message to log.
    """
    print(f"[ {log_bold(log_magenta('DEBUG'))} ] {message}")

def log_link(message: str) -> None:
    """
    Logs a link message to the console in blue.

    Args:
        message (str): The link message to log.
    """
    print(f"[ {log_bold(log_cyan('LINK'))} ] {message}")