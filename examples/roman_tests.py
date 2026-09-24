assert roman_to_int("III") == 3
assert roman_to_int("IV") == 4
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("mmxxvi") == 2026  # lowercase must be accepted
for bad in ["", "IIII", "IC", "ABC", "VV"]:
    try:
        roman_to_int(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"roman_to_int({bad!r}) should raise ValueError")
print("roman: all tests passed")
