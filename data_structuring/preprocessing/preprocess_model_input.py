# ############################################################################################
# Input preprocessing script taken from section 5.2 of the User Guide
# ############################################################################################

import argparse
import re

SEPARATOR_CHARS = [" ", "\t", "\n"]
MOVE_WORD_NEXT_LINE_LENGTH = 6


def split_every_35th_char(string):
    split_words = []
    for word in re.split(r"(\W)", string):
        if word == "":
            continue
        if word not in SEPARATOR_CHARS:
            if (len(split_words) == 0
                    or any(x in split_words[-1] for x in SEPARATOR_CHARS)):
                split_words.append(word)
            else:
                split_words[-1] = split_words[-1] + word
        else:
            split_words.append(word)
    i, res, current_line = 0, "", ""
    while i < len(split_words):
        if split_words[i] == "\n":
            res = res + current_line + "\n"
            current_line = ""
        elif split_words[i] in [" ", "\t"] and current_line == "":
            pass
        else:
            if (len(split_words[i]) <= MOVE_WORD_NEXT_LINE_LENGTH
                    and (35 < len(current_line + split_words[i])
                         <= 35 + MOVE_WORD_NEXT_LINE_LENGTH)):
                res = res + current_line + "\n"
                current_line = ""
                continue
            else:
                temp_current_line = current_line + split_words[i]
                for j in range(0, len(temp_current_line), 35):
                    if len(temp_current_line[j:j + 35]) >= 35:
                        res = res + temp_current_line[j:j + 35] + "\n"
                        current_line = ""
                    else:
                        current_line = temp_current_line[j:j + 35]
        i += 1
    if current_line != "":
        res = res + current_line
    return res

# Takes a single argument which is the input that needs to be re-formatted
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_str", type=str)
    print(f"{split_every_35th_char(parser.parse_args().input_str).replace('\n', '\\n')}")
