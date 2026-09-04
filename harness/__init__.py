"""The harness: everything that knows ground truth.

Nothing here is imported by `revenew/`. This package declares true response
rates, drives the runtime blind through its normal webhook/detection path, and
then -- and only then -- reads the runtime's own decisions back out to grade
them against the truth it declared. See db/harness_schema.sql for why that
grading happens over an explicit, one-directional ATTACH rather than a shared
connection.
"""
