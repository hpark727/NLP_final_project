# Cross-Adaptive Qwen4B/Llama3B Eval

| Target model | Prefix source | k | ASR | Success / N | 95% CI |
|---|---:|---:|---:|---:|---:|
| llama3b | llama3b_mined | 5 | 17.4% | 15/86 | 10.9%-26.8% |
| llama3b | llama3b_mined | 10 | 17.4% | 15/86 | 10.9%-26.8% |
| llama3b | llama3b_mined | 20 | 25.6% | 22/86 | 17.5%-35.7% |
| llama3b | llama3b_mined | 40 | 34.9% | 30/86 | 25.7%-45.4% |
| llama3b | llama3b_mined | full | 36.0% | 31/86 | 26.7%-46.6% |
| llama3b | qwen4b_mined | 5 | 13.9% | 10/72 | 7.7%-23.7% |
| llama3b | qwen4b_mined | 10 | 8.3% | 6/72 | 3.9%-17.0% |
| llama3b | qwen4b_mined | 20 | 36.1% | 26/72 | 26.0%-47.6% |
| llama3b | qwen4b_mined | 40 | 58.3% | 42/72 | 46.8%-69.0% |
| llama3b | qwen4b_mined | full | 54.2% | 39/72 | 42.7%-65.2% |
| qwen4b | llama3b_mined | 5 | 7.0% | 6/86 | 3.2%-14.4% |
| qwen4b | llama3b_mined | 10 | 2.3% | 2/86 | 0.6%-8.1% |
| qwen4b | llama3b_mined | 20 | 5.8% | 5/86 | 2.5%-12.9% |
| qwen4b | llama3b_mined | 40 | 15.1% | 13/86 | 9.1%-24.2% |
| qwen4b | llama3b_mined | full | 24.4% | 21/86 | 16.6%-34.5% |
| qwen4b | qwen4b_mined | 5 | 50.0% | 36/72 | 38.7%-61.3% |
| qwen4b | qwen4b_mined | 10 | 48.6% | 35/72 | 37.4%-59.9% |
| qwen4b | qwen4b_mined | 20 | 51.4% | 37/72 | 40.1%-62.6% |
| qwen4b | qwen4b_mined | 40 | 54.2% | 39/72 | 42.7%-65.2% |
| qwen4b | qwen4b_mined | full | 59.7% | 43/72 | 48.2%-70.3% |
