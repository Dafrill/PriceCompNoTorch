import time
import os
from utils import OUTPUT_DIR


def save_results_rmd(results, best_test):
    """Save results to R Markdown file - one plot with 1-layer vs 2-layer."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rmd_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.Rmd")

    def r_vec(vals):
        return ", ".join(str(v) for v in vals)

    def r_vec_str(vals):
        return ', '.join(f'"{v}"' for v in vals)

    def r_table(name, rows, col_map):
        """Generuje data.frame. col_map = [(col_name, dict_key), ...]"""
        lines_out = []
        lines_out.append(f"{name} <- data.frame(")
        for i, (col_name, dict_key) in enumerate(col_map):
            col_vals = [row[dict_key] for row in rows]
            comma = "," if i < len(col_map) - 1 else ""
            if isinstance(col_vals[0], str):
                lines_out.append(f"  {col_name} = c({r_vec_str(col_vals)}){comma}")
            else:
                lines_out.append(f"  {col_name} = c({r_vec(col_vals)}){comma}")
        lines_out.append(")")
        return lines_out

    # Columns for comparison tables
    COMP_COLS = [
        ('NetworkType', 'network_type'),
        ('Param', 'param_value'),
        ('TrainCosine', 'TrainCosine'),
        ('TrainCosineStd', 'TrainCosineStd'),
        ('TestCosine', 'TestCosine'),
        ('TestCosineStd', 'TestCosineStd')
    ]

    lines = []
    lines.append("---")
    lines.append("title: \"Test Results - Kohonen WTA Network\"")
    lines.append("author: \"Magda\"")
    lines.append(f'date: "{time.strftime("%Y-%m-%d")}"')
    lines.append("output: html_document")
    lines.append("---")
    lines.append("")
    lines.append("```{r setup, include=FALSE}")
    lines.append("knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)")
    lines.append("library(ggplot2)")
    lines.append("library(knitr)")
    lines.append("library(tidyr)")
    lines.append("library(dplyr)")
    lines.append("```")
    lines.append("")
    lines.append("# Kohonen Neural Network Test Results")
    lines.append("")
    lines.append(f"**Generation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Add comparison sections for each series
    series_configs = [
        ('S1_', 'n_neurons_l1', 'Series 1: Number of L1 Neurons (L2=5)', 'L1'),
        ('S2_', 'n_neurons_l2', 'Series 2: Number of L2 Neurons (L1=40)', 'L2'),
        ('S3_', 'eta_init', 'Series 3: Learning Rate (L1=40, L2=5)', 'Learning Rate'),
        ('S4_', 'eta_decay', 'Series 4: Learning Rate Decay (L1=40, L2=5)', 'Decay'),
        ('S5_', 'epochs', 'Series 5: Number of Epochs (L1=40, L2=5)', 'Epochs'),
        ('S6_', 'activation', 'Series 6: Different Activation Functions (L1=40, L2=5)', 'Activation'),
        ('S7_', 'train_ratio', 'Series 7: Train Ratio (L1=40, L2=5)', 'Train Ratio')
    ]

    for series_prefix, param_name, title, x_var in series_configs:
        # Filter results for this series
        series_results = [r for r in results if r['config'].startswith(series_prefix)]
        
        if not series_results:
            continue

        # Add param_value for plotting
        for r in series_results:
            val = r.get(param_name, 0)
            if isinstance(val, str):
                r['param_value'] = f'"{val}"'
            else:
                r['param_value'] = val
        
        lines.append(f"## {title}")
        lines.append("")
        
        # Table comparison
        lines.append("### Cosine Similarity Comparison")
        lines.append("")
        lines.append("```{r}")
        comp_name = f"comp_{series_prefix.lower()}"
        lines += r_table(comp_name, series_results, COMP_COLS)
        lines.append(f"kable({comp_name}, digits = 4, caption = 'Comparison: 1-Layer vs 2-Layer Networks')")
        lines.append("```")
        lines.append("")
        
        # One plot: x = Param, fill = NetworkType (different colors for 1-layer vs 2-layer)
        # Train/Test side-by-side within each NetworkType
        lines.append(f"ggplot({comp_name}_long, aes(x = factor(Param), y = Value, fill = NetworkType, alpha = Metric)) +")
        lines.append(f"  geom_col(position = position_dodge(width = 0.7), width = 0.6) +")
        lines.append(f"  geom_text(aes(label = sprintf('%.4f', Value)),")
        lines.append(f"            position = position_dodge(width = 0.7), vjust = -0.5, size = 3) +")
        lines.append(f"  coord_cartesian(ylim = c(cos_min, cos_max)) +")
        lines.append(f"  labs(x = '{x_var}', y = 'Cosine Similarity', title = '{title}') +")
        lines.append(f"  scale_fill_manual(values = c('1-layer' = '#1F77B4', '2-layer' = '#2CA02C')) +")
        lines.append(f"  scale_alpha_manual(values = c('Train' = 1.0, 'Test' = 0.7)) +")
        lines.append(f"  theme_minimal() +")
        lines.append(f"  theme(legend.position = 'top', plot.title = element_text(size = 10))")
        lines.append("```")
        lines.append("")

    # Best Configuration
    lines.append("## Best Configuration")
    lines.append("")
    lines.append(f"**Name:** {best_test['config']}")
    lines.append("")
    lines.append(f"- Number of layers: {best_test['n_layers']}")
    lines.append(f"- Number of neurons (L1): {best_test['n_neurons_l1']}")
    if best_test['n_layers'] == 2:
        lines.append(f"- Number of neurons (L2): {best_test['n_neurons_l2']}")
    lines.append(f"- Activation function: {best_test['activation']}")
    lines.append(f"- Learning rate: {best_test['eta_init']}")
    lines.append(f"- Learning rate decay: {best_test['eta_decay']}")
    lines.append(f"- Number of epochs: {best_test['epochs']}")
    lines.append(f"- Train ratio: {best_test['train_ratio']}")
    lines.append(f"- Train Cosine: {best_test.get('TrainCosine', 0):.4f} +/- {best_test.get('TrainCosineStd', 0):.4f}")
    lines.append(f"- Test Cosine: {best_test.get('TestCosine', 0):.4f} +/- {best_test.get('TestCosineStd', 0):.4f}")
    lines.append("")

    # ALL RESULTS
    lines.append("## All Results")
    lines.append("")
    lines.append("```{r}")
    ALL_COLS = [
        ('Config', 'config'),
        ('Type', 'network_type'),
        ('L1', 'n_neurons_l1'),
        ('L2', 'n_neurons_l2'),
        ('Activation', 'activation'),
        ('TrainCosine', 'TrainCosine'),
        ('TrainCosineStd', 'TrainCosineStd'),
        ('TestCosine', 'TestCosine'),
        ('TestCosineStd', 'TestCosineStd')
    ]
    lines += r_table("all_data", results, ALL_COLS)
    lines.append("kable(all_data, digits = 4, caption = 'Full results of all tests')")
    lines.append("```")

    with open(rmd_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Wyniki zapisane do: {rmd_path}")
    return rmd_path
