# OOI System Interface

Some modules expose thin single-output wrapper functions in addition to the
core multi-output functions documented in the API Reference. These wrappers
exist because the OOI data management system historically required one output
per function call.

External users should use the **core functions** directly — they return all
outputs at once and avoid unnecessary overhead.

Wrapper functions are documented alongside their core counterparts in each
instrument family page, under a collapsed "OOI System Interface" section.
