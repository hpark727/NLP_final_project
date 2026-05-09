# Cross-Adaptive Qwen4B/Llama3B Eval

| Target model | Prefix source | k | ASR | Success / N | 95% CI |
|---|---:|---:|---:|---:|---:|
| llama3b_static_augmented | llama3b_mined | 5 | 3.5% | 3/86 | 1.2%-9.8% |
| llama3b_static_augmented | llama3b_mined | 10 | 2.3% | 2/86 | 0.6%-8.1% |
| llama3b_static_augmented | llama3b_mined | 20 | 10.5% | 9/86 | 5.6%-18.7% |
| llama3b_static_augmented | llama3b_mined | 40 | 26.7% | 23/86 | 18.5%-36.9% |
| llama3b_static_augmented | llama3b_mined | full | 19.8% | 17/86 | 12.7%-29.4% |
| llama3b_static_augmented | qwen4b_mined | 5 | 48.6% | 35/72 | 37.4%-59.9% |
| llama3b_static_augmented | qwen4b_mined | 10 | 38.9% | 28/72 | 28.5%-50.4% |
| llama3b_static_augmented | qwen4b_mined | 20 | 44.4% | 32/72 | 33.5%-55.9% |
| llama3b_static_augmented | qwen4b_mined | 40 | 41.7% | 30/72 | 31.0%-53.2% |
| llama3b_static_augmented | qwen4b_mined | full | 48.6% | 35/72 | 37.4%-59.9% |
| qwen4b_static_augmented | llama3b_mined | 5 | 30.2% | 26/86 | 21.5%-40.6% |
| qwen4b_static_augmented | llama3b_mined | 10 | 22.1% | 19/86 | 14.6%-31.9% |
| qwen4b_static_augmented | llama3b_mined | 20 | 90.7% | 78/86 | 82.7%-95.2% |
| qwen4b_static_augmented | llama3b_mined | 40 | 73.3% | 63/86 | 63.1%-81.5% |
| qwen4b_static_augmented | llama3b_mined | full | 79.1% | 68/86 | 69.3%-86.3% |
| qwen4b_static_augmented | qwen4b_mined | 5 | 93.1% | 67/72 | 84.8%-97.0% |
| qwen4b_static_augmented | qwen4b_mined | 10 | 94.4% | 68/72 | 86.6%-97.8% |
| qwen4b_static_augmented | qwen4b_mined | 20 | 73.6% | 53/72 | 62.4%-82.4% |
| qwen4b_static_augmented | qwen4b_mined | 40 | 68.1% | 49/72 | 56.6%-77.7% |
| qwen4b_static_augmented | qwen4b_mined | full | 72.2% | 52/72 | 61.0%-81.2% |
