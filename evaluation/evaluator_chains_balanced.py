"""
BALANCED chain evaluator.

The balanced dataset has three variants per full GT chain:
  - healthy_partial: execute blocks 0..s, peek at letter_{s+1}, loss masked.
    (training signal only — not evaluated)
  - healthy_full: full N-step GT chain, normal loss.
  - unhealthy: wrong block at step s + STOP.

This evaluator filters out healthy_partial examples, then runs the standard
StopChainEvaluator pipeline:
  - free-gen chain quality on healthy_full
  - forced-prefix stop-chain quality on unhealthy
  - STOP probability probing on healthy_full
  - Health/ summary metrics
"""

from evaluation.evaluator_chains_stop import StopChainEvaluator


class BalancedChainEvaluator(StopChainEvaluator):
    """Filters healthy_partial examples out, then delegates to StopChainEvaluator."""

    def evaluate(self, test_dataset):
        raw = self.raw_test_data
        if raw is not None:
            keep_idx = [i for i, ex in enumerate(raw)
                        if ex.get("variant") != "healthy_partial"]
            if len(keep_idx) < len(raw):
                test_dataset = test_dataset.select(keep_idx)
                self.raw_test_data = [raw[i] for i in keep_idx]
        return super().evaluate(test_dataset)
