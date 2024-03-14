def consistent(name: str) -> str:
    """
    consistent is a function to ensure file names are consistent.
    """
    newname = name.lower()
    newname = newname.replace(" ", "_")
    newname = newname.replace("(", "")
    newname = newname.replace(")", "")
    newname = newname.replace("&", "and")
    newname = newname.replace("+", "_")
    return newname
