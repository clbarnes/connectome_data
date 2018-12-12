

def unpad_number(s: str) -> str:
    out = ''
    this_element = ''
    was_digit = s[0].isdigit()
    for c in s:
        is_digit = c.isdigit()
        if is_digit != was_digit:
            if was_digit:
                this_element = str(int(ele_str))
            out += this_element
            this_element = ''
            was_digit = is_digit
        this_element += c
    if was_digit:
        this_element = str(int(ele_str))
    out += this_element
    return out
