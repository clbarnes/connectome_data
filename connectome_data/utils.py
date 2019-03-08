

def unpad_name(s: str) -> str:
    s = s.strip()
    out = ''
    this_element = ''
    is_digit = s[0].isdigit()

    for next_char in s:
        next_is_digit = next_char.isdigit()
        if next_is_digit != is_digit:
            this_element = unpad_element(this_element)
            out += this_element
            this_element = ''
            is_digit = next_is_digit
        this_element += next_char

    this_element = unpad_element(this_element)
    out += this_element
    return out


def unpad_element(s: str) -> str:
    if s.isdigit():
        s = str(int(s))
    return s
