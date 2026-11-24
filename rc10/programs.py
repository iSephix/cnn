"""Registered RC10 hypothesis families."""
from canvas import infer as canvas
from catalog import infer as catalog
from pointers import infer as pointers
from resample import infer as resample
from stamps import infer as stamps

LEARNERS = dict(canvas=canvas, catalog=catalog, pointers=pointers, resample=resample, stamps=stamps)
