# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

"""Array related utils."""
from typing import List, Union

import pyspark
from pyspark.sql import functions as f

from .alias import alias


@alias()
def array_rlike(
    input: str,  # pylint: disable=redefined-builtin
    pattern: Union[str, List[str]],
) -> pyspark.sql.Column:
    """Uses rlike to parse an array column.

    Use `alias` argument for setting the name of the output column.

    Args:
        input: The input column name.
        pattern: The pattern(s) to search the array for.

    Returns:
        A spark array column with elements rlike the pattern.
    """
    if isinstance(pattern, str):
        pattern = [pattern]

    pattern = _create_regex_from_list(pattern)

    if pattern == "":
        filtered_array = f.array()
    else:
        filtered_array = f.expr(f"filter({input}, x -> x rlike '{pattern}')")

    return filtered_array


def _create_regex_from_list(list_of_regex: List[str]) -> str:
    """Given a list of strings, create a regex expression that matches.

    Matches if any of the words exist in the string. i.e. ['apple', 'oranges', 'pears']
    produces r'(apple)|(oranges)|(pears)'.

    Args:
        list_of_regex: a list of regex values

    Returns:
        A single regex expression
    """
    formatted_codes = [
        r"({})".format(word)  # pylint: disable=consider-using-f-string # noqa: E501
        for word in list_of_regex
    ]
    grouped_codes = "|".join(formatted_codes)
    return grouped_codes
