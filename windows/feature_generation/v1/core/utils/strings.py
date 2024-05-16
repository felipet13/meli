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

"""Contains string related utility functions."""

import unicodedata
from functools import reduce
from itertools import chain
from typing import Dict, List, Optional, Text, Union

import pyspark.sql.functions as f
from pyspark.sql import Column
from pyspark.sql.types import StringType

from .alias import alias


@f.udf(returnType=StringType())
def remove_accents(input_string: str, form: str = "NFKD") -> Text:  # pragma: no cover
    """Removes accents by converting from non-ascii to ascii.

    Valid values for form are 'NFC', 'NFKC', 'NFD', and 'NFKD'.
    The meanings of each from can be found here:
    https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize

    Args:
        input_string: non-ascii string
        form: The normal form 'form' for the Unicode string unistr.

    Returns:
        ascii string
    """
    # normalize string only if value is not None, otherwise it throws a terrible error
    if input_string is None:
        return None

    nfkd_str = unicodedata.normalize(form, input_string)

    # Keep chars that has no other char combined (i.e. accents chars)
    return "".join(  # pylint: disable=redundant-u-string-prefix
        [c for c in nfkd_str if not unicodedata.combining(c)]
    )


@alias()
def map_values_from_dict(
    column: str, mapping_dictionary: dict, coalesce: bool = True
) -> Column:
    """Maps values using a dictionary.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The string name of the column
        mapping_dictionary: Dictionary of mapping values
        coalesce: If key does not exist in mapping dictionary, use original value

    Returns:
        A spark column object
    """
    mapping_expr = f.create_map([f.lit(x) for x in chain(*mapping_dictionary.items())])

    if coalesce:
        return f.coalesce(mapping_expr[f.col(column)], f.col(column))

    return mapping_expr[f.col(column)]


@alias()
def sequential_regexp_replace(column: str, regexp_replace_dict: Dict) -> Column:
    """Sequentially replaces all substrings of the specified string value.

    The value match regexp with rep based on a supplied dictionary.
    Note that if replacements need to be run in a specific order, call this method
    successively.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The column name
        regexp_replace_dict: A dictionary containing characters to be replaced and
        their replacements.

    Returns:
        A spark column object
    """
    keys = list(regexp_replace_dict.keys())
    keys.sort()

    return reduce(
        lambda regexp_column, key: (
            f.regexp_replace(regexp_column, key, regexp_replace_dict[key])
            if regexp_replace_dict[key] is not None
            else f.lit(None)
        ),
        # The aforementioned Lambda function return None column if the
        # value that needs to be replaced
        # is None
        keys,
        column,
    )


@alias()
def mask_string(column: str, regexp_replace_dict: Optional[Dict] = None) -> Column:
    """Generalise string patterns using a default regexp_replace_dict.

    Use `alias` argument for setting the name of the output column.

    Defaults to:
    * A-Z gets replaced with A
    * a-z gets replaced with a
    * 0-9 gets replaced with 0.
    * All symbols remain unchanged.

    Can be used for exploration of general patterns in a string column.

    Args:
        column: The column name
        regexp_replace_dict: A dictionary containing characters to be replaced and
        their replacements.

    Returns:
        A spark column object
    """
    default_dict = {"[A-Z]": "A", "[0-9]": "0", "[a-z]": "a"}

    regexp_replace_dict = regexp_replace_dict or default_dict

    return sequential_regexp_replace(column, regexp_replace_dict)


def keep_alphanumeric(column: str):
    """Removes all non-alphanumeric characters excluding spaces and underscores."""
    return f.regexp_replace(column, r"[^a-zA-Z0-9_ ]+", "")


@alias()
def regex_map_values_from_dict(
    column: str,
    mapping: Dict[str, Union[str, List[str]]],
    other: Optional[str] = None,
    coalesce: bool = False,
) -> Column:
    """Maps values using a dictionary with regular expression rules.

    The key represents the final value and its value represent a list of rules defined
    in regular expressions.

    Use `alias` argument for setting the name of the output column.

    Args:
        column: The string name of the column.
        mapping: Dictionary of mapping values.
        other: default values when the column doesn't match any of the expressions.
        coalesce: Coalesce to original value if ``True``.

    Returns:
        A spark column object.
    """
    coalesce_replacements = []
    for replacement, pattern in mapping.items():
        # Converting list of regex patterns into a string separated by an or condition.
        if isinstance(pattern, list):
            pattern = "|".join(pattern)
        replacement = f.when(f.col(column).rlike(pattern), replacement)
        coalesce_replacements.append(replacement)

    other_replacement = f.col(column) if coalesce else f.lit(other)

    coalesce_replacements.append(other_replacement)

    return f.coalesce(*coalesce_replacements)
