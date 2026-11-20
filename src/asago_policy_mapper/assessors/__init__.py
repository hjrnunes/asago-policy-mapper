"""Assessors: decision models that answer a question for a batch of cases.

The package stays import-light: no assessor module imports an HTTP client
at module level — the HTTP probability assessors (the generic one in
``http_probability`` and the Jev mapping in ``jev``) import ``httpx``
lazily inside their constructors — so the default install (without the
``decision`` extra) is unchanged.
"""
