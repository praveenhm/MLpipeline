from rich import print

SEPERATOR = "." * 80
INTERMEDIATE_SEPERATOR = "-" * 80


def banner(items: list[str], skip_intermediate=False):
    print(SEPERATOR)
    intermediate = INTERMEDIATE_SEPERATOR + "\n"
    if skip_intermediate:
        intermediate = "\n"
    print(intermediate.join(items))
    print(SEPERATOR)
