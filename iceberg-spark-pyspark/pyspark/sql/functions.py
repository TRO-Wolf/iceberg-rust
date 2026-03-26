"""PySpark functions compatibility shim — re-exports iceberg_spark.functions."""

from iceberg_spark.functions import *  # noqa: F401, F403
from iceberg_spark.functions import (
    ascii_func as ascii,
    chr_func as chr,
    hex_func as hex,
    repeat_func as repeat,
    struct_func as struct,
    date_add_func as date_add,
)
from iceberg_spark import (
    broadcast,
    expr,
    input_file_name,
    monotonically_increasing_id,
    spark_partition_id,
    typedLit,
)
