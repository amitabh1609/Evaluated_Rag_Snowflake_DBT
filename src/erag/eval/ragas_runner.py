"""Run RAGAS evaluation over benchmark.yaml.

Metrics computed: faithfulness, answer_relevancy, context_precision, context_recall.
Outputs:
  - JSON file: data/benchmark/results_<timestamp>.json
  - Markdown table: printed to stdout (and captured by CI)

Per-tier breakdown is included in addition to the global average.

Usage:
    python -m erag.eval.ragas_runner          # full benchmark
    python -m erag.eval.ragas_runner --smoke  # 10-question CI subset
"""

# TODO Phase 4: implement
#   - Load benchmark.yaml
#   - For each question: call answerer.answer(), collect (question, answer, contexts, ground_truth)
#   - Build ragas.EvaluationDataset and run ragas.evaluate()
#   - Write per-tier breakdown
#   - In --smoke mode: use data/benchmark/*.yaml (10-question subset committed to repo)
