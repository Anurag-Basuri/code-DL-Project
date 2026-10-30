# Hinglish LID — Experiment Results Summary

Generated: 2026-10-27 17:15:43

## Ablation Results

| Loss Function | Eval Set | Accuracy | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|---|
| Cross-Entropy | LinCE Val | 0.9688 | 0.9428 | 0.9689 |
| Cross-Entropy | LinCE Held-out Test | 0.9610 | 0.9347 | 0.9610 |
| Cross-Entropy | Custom Conversational | 0.9823 | 0.9521 | 0.9825 |
| Weighted CE | LinCE Val | 0.9664 | 0.9400 | 0.9669 |
| Weighted CE | LinCE Held-out Test | 0.9596 | 0.9339 | 0.9599 |
| Weighted CE | Custom Conversational | 0.9845 | 0.9535 | 0.9847 |
| Focal Loss (γ=2) | LinCE Val | 0.9686 | 0.9427 | 0.9687 |
| Focal Loss (γ=2) | LinCE Held-out Test | 0.9630 | 0.9373 | 0.9630 |
| Focal Loss (γ=2) | Custom Conversational | 0.9801 | 0.9451 | 0.9805 |

## Per-Class F1 Scores

### Cross-Entropy

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EN | 0.9827 | 0.9824 | 0.9825 | 8388.0 |
| HI | 0.9564 | 0.9579 | 0.9571 | 3181.0 |
| NE | 0.8419 | 0.8625 | 0.8520 | 858.0 |
| OTHER | 0.9849 | 0.9744 | 0.9796 | 2148.0 |

### Weighted CE

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EN | 0.9878 | 0.9732 | 0.9804 | 8388.0 |
| HI | 0.9517 | 0.9601 | 0.9559 | 3181.0 |
| NE | 0.7951 | 0.8998 | 0.8442 | 858.0 |
| OTHER | 0.9836 | 0.9758 | 0.9797 | 2148.0 |

### Focal Loss (γ=2)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EN | 0.9822 | 0.9820 | 0.9821 | 8388.0 |
| HI | 0.9558 | 0.9585 | 0.9571 | 3181.0 |
| NE | 0.8400 | 0.8625 | 0.8511 | 858.0 |
| OTHER | 0.9873 | 0.9735 | 0.9803 | 2148.0 |


## Generated Outputs

- `outputs/ablation_table.json`
- `outputs/cmi_distribution.png`
- `outputs/cmi_scores.json`
- `outputs/confusion_matrix_ce_custom.png`
- `outputs/confusion_matrix_focal_custom.png`
- `outputs/confusion_matrix_weighted_custom.png`
- `outputs/dataset_statistics.png`
- `outputs/error_analysis_ce.json`
- `outputs/error_analysis_focal.json`
- `outputs/error_analysis_weighted.json`
- `outputs/example_predictions.png`
- `outputs/f1_macro_comparison.png`
- `outputs/f1_per_class_custom_conversational.png`
- `outputs/f1_per_class_lince_heldout_test.png`
- `outputs/f1_per_class_lince_val.png`
- `outputs/metrics_heatmap_ce_custom_conversational.png`
- `outputs/metrics_heatmap_ce_lince_heldout_test.png`
- `outputs/metrics_heatmap_ce_lince_val.png`
- `outputs/metrics_heatmap_focal_custom_conversational.png`
- `outputs/metrics_heatmap_focal_lince_heldout_test.png`
- `outputs/metrics_heatmap_focal_lince_val.png`
- `outputs/metrics_heatmap_weighted_custom_conversational.png`
- `outputs/metrics_heatmap_weighted_lince_heldout_test.png`
- `outputs/metrics_heatmap_weighted_lince_val.png`