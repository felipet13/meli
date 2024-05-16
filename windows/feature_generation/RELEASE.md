# Release History

## 0.6.7
- Fixed a bug for when `functools.partial` function is used for aggregation in window features.

## 0.6.6
- Removed unneccecary dependency bounds

## 0.6.5
- Brix post docs update.

## 0.6.4
- Fixed several ruff errors.

## 0.6.3
- Migrated to MonoRepo.
- Use alloy.
- Migrate docs to `ipynb` format.

## 0.6.2
- Remove data fabricator from `config.yml`.

## 0.6.1
- Update docs.

## 0.6.0
- Function argument semantics update for `feature flag`, `feature tags` functions. Refer CHANGELOG.md for detailed list of function signature changes.
- Remove window_tag function from `feature tags`.

## 0.5.2
- String and List datatypes are acceptable in "values" argument in rlike flag method

## 0.5.1
- Remove `nb-black` from test requirements.

## 0.5.0
- Add aggregation functionality for ArrayType Feature over window grid.

## 0.4.4
- Made pytests compatible with Python 3.7 through targeted skipping.

## 0.4.3
- Pinned packages to fix test failures.

## 0.4.2
- Add pytest coverage limits.

## 0.4.1
- Changed the dynamic data generation in myst file to avoid md file changes on each run.

## 0.4.0
- Making window naming more explicit.

## 0.3.1
- Added CHANGELOG.md for keeping track of changes to function signatures.


## 0.3.0
- Changed the dynamic data generation in myst file to avoid md file changes on each run.
- Implemented the alias decorator to all column functions. Where appropriate, use the `alias` kwarg to rename columns.
Please follow the following instructions to fix the breaking changes:
  - **Functions using `alias` parameter**: If you are using implicit arg calls, you will need to specify the `alias`
  param explicitly. If you are using explicit arg calls, no change is required.
  - **Functions using `output_col_name` parameter**: If you are using implicit arg calls, you will need to specify the
  `alias` param explicitly. If you are using explicit arg calls, change the param name to `alias`.
  - **Functions not using `alias` like functionality**: Use `alias` kwarg to specify the alias for the returned column.
- Fixed brix version history.

## 0.2.9
- Replaced augment decorator with individual decorators.

## 0.2.8
- Add `complete_sum`, `complete_max` custom window aggregation function.
- Add blacken-docs to format python code in md files.

## 0.2.7
- Fix `refit` import order.

## 0.2.6
- Rename .myst to .myst.md for better editor support.

## 0.2.5
- Introduce   jupyterbook mechanism.

## 0.2.5
- Remove rename decorator from `feature_generation` utility `array_aggregate`

## 0.2.3
- Update docs.

## 0.2.2
- Migrate to new repo and add new test requirements.

## 0.2.1
- Update sorting of imports.

## 0.2.0
- Deprecated v0!

## 0.1.45
- Added new `union_dataframes` node function.

## 0.1.44
- Added support for functions without inputs in `generate_window_grid`.

## 0.1.43
- Added an alternate for `expand_tags` function - `expand_tags_all`, which expands all tags present in the dataframe.

## 0.1.42
- Added upper bound for `scikit-learn` to prevent breaking changes in `1.1.0`.

## 0.1.41
- Added document for `timeseries` `tall vs wide experiment`.

## 0.1.40
- Added `coalesce_cols` to `flags`.

## 0.1.39
- Update `sorted_collect_list` and add `interpolate_constant` function.

## 0.1.38
- Triggering a new build to Brix

## 0.1.37
- Enable `remove_output_debug_columns` for the `join_dataframes` node

## 0.1.36
- More clean up of requirements.

## 0.1.35
- Cleaned up test_requirements file.

## 0.1.34
- Suppress `redundant-u-string-prefix` pylint error message.

## 0.1.33
- Added `py.typed` file.

## 0.1.32
- Added `aggregate_over_slice_grid` and `aggregate_over_slice` functions.

## 0.1.31
- Loosened `sklearn` requirement from `scikit-learn~=1.0` to `scikit-learn>=0.21`

## 0.1.30
- Added `join_dataframes` node

## 0.1.29
- Updated `time_since` function to correct the default argument

## 0.1.28
- Add `join_dataframes` functionality.

## 0.1.27
- Remove all .Rmd files.

## 0.1.26
- Convert .Rmd to .myst.

## 0.1.25
- Add documentation for `generate_distinct_element_window_grid`,
  `generate_window_ratio`, `generate_window_delta`.

## 0.1.24
- Add function `generate_distinct_element_window_grid` in features.
- Add function to calculate delta and ratio over a window in `Feature Generation`.

## 0.1.23
- Added intro document to timeseries.

## 0.1.22
- Add deprecation warning to `v0`.

## 0.1.21
- Update requirements and test requirements.

## 0.1.20
- Add wrapper function `alias` in utils

## 0.1.19
- Update to `scikit-learn~=1.0`.

## 0.1.18
- Update to sklearn 0.24.

## 0.1.17
- Update `feature_generation` utility `keep_alphanumberic` to include underscore.

## 0.1.16
- Relax very tightly-bound `scikit-learn` dependency.

## 0.1.14
- Update authors.

## 0.1.13
- Fix bug in `create_columns_from_config` with parameters `sequential` and `params_keep_cols`
- Fix bug in `time_since` utilities
- Remove temp tag columns from `create_tags_from_config`

## 0.1.12
- Added case study documentation.

## 0.1.11
- Fix bug in `create_columns_from_config`

## 0.1.10
- Updated docstring for functions with `param_keep_cols`.

## 0.1.9
- Modifications to functions decorated with `make_list_regexable`.

## 0.1.8
- Update `generate_window_spec` to have `orderBy` optional.

## 0.1.7
- Update docs.

## 0.1.6
- Bump pyspark version to < 4.0.

## 0.1.5
- Updated `make_regexable` decorator in `expand_tags` and `create_column` to raise an exception.

## 0.1.4
- Fix docs.

## 0.1.3
- Add `06_utils.md`.

## 0.1.2
- Replaced `node` decorator with `augment`.

## 0.1.1
- Fixed type in doc.

## 0.1.0
- Added feature generation to brix.
