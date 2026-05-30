"""Label construction for offline training.

Triple-barrier labels (López de Prado AFML §3), distinct from the meta-label
algorithm port at ``renquant_backtesting.meta_label.triple_barrier``:

  * ``labels.triple_barrier``      — label generator for training panel
  * ``meta_label.triple_barrier``  — faithful AFML §3.4 algorithm port

The naming overlap is a §5.13.5 candidate to consolidate; for now both live
in the subrepo so umbrella lifts can land without behavioural change.
"""
