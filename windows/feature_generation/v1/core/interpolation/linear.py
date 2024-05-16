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

"""Linear interpolation between 2 columns."""
from typing import Optional, Union

import pyspark
import pyspark.sql.functions as f

from ..utils.alias import alias


@alias()
def interpolation_linear(
    start_col: Union[str, pyspark.sql.Column],
    end_col: Union[str, pyspark.sql.Column],
    step_size: Optional[Union[int, str]] = None,
) -> pyspark.sql.Column:
    """Creates a row between the start and end columns, inclusive of boundaries.

    Use `alias` argument for setting the name of the output column.

    Args:
        start_col: The start range column.
        end_col: The end range column.
        step_size: The step size to interpolate between. Defaults to None which
            defaults to 1 in the underlying function. For integer interpolation,
            pass in an integer n. For dates, pass in a string ``INTERVAL n <UNIT>``
            where UNIT can be: YEAR, MONTH, DAY, HOUR,
            MINUTE, SECOND, MILLISECOND, and MICROSECOND.

    Returns:
        A linearly exploded column.
    """
    if isinstance(start_col, str):
        start_col = f.col(start_col)

    if isinstance(end_col, str):
        end_col = f.col(end_col)

    if isinstance(step_size, str):
        step_size = f.expr(step_size)

    sequence = f.sequence(start=start_col, stop=end_col, step=step_size)

    return f.explode(sequence)
