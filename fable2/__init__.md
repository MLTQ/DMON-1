# __init__.py

## Purpose

Package marker for `fable2`. Modules are imported explicitly
(`fable2.train`, `fable2.audit`, …); nothing is re-exported, so importing the
package never pulls in `transformers` or a checkpoint.
