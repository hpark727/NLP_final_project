# Cross-Adaptive Qwen4B/Llama3B Eval

| Target model | Prefix source | k | ASR | Success / N | 95% CI |
|---|---:|---:|---:|---:|---:|
| llama3b_static_augmented | llama3b_mined | 5 | 3.7% | 6/163 | 1.7%-7.8% |
| llama3b_static_augmented | llama3b_mined | 10 | 8.6% | 14/163 | 5.2%-13.9% |
| llama3b_static_augmented | llama3b_mined | 20 | 16.0% | 26/163 | 11.1%-22.3% |
| llama3b_static_augmented | llama3b_mined | 40 | 21.5% | 35/163 | 15.9%-28.4% |
| llama3b_static_augmented | llama3b_mined | full | 30.7% | 50/163 | 24.1%-38.1% |
| llama3b_static_augmented | qwen4b_mined | 5 | 8.4% | 45/533 | 6.4%-11.1% |
| llama3b_static_augmented | qwen4b_mined | 10 | 10.5% | 56/533 | 8.2%-13.4% |
| llama3b_static_augmented | qwen4b_mined | 20 | 12.6% | 67/533 | 10.0%-15.7% |
| llama3b_static_augmented | qwen4b_mined | 40 | 14.8% | 79/533 | 12.1%-18.1% |
| llama3b_static_augmented | qwen4b_mined | full | 21.0% | 112/533 | 17.8%-24.7% |
| qwen4b_static_augmented | llama3b_mined | 5 | 1.2% | 2/163 | 0.3%-4.4% |
| qwen4b_static_augmented | llama3b_mined | 10 | 1.8% | 3/163 | 0.6%-5.3% |
| qwen4b_static_augmented | llama3b_mined | 20 | 46.6% | 76/163 | 39.1%-54.3% |
| qwen4b_static_augmented | llama3b_mined | 40 | 55.8% | 91/163 | 48.2%-63.2% |
| qwen4b_static_augmented | llama3b_mined | full | 57.7% | 94/163 | 50.0%-65.0% |
| qwen4b_static_augmented | qwen4b_mined | 5 | 33.6% | 179/533 | 29.7%-37.7% |
| qwen4b_static_augmented | qwen4b_mined | 10 | 39.2% | 209/533 | 35.2%-43.4% |
| qwen4b_static_augmented | qwen4b_mined | 20 | 44.3% | 236/533 | 40.1%-48.5% |
| qwen4b_static_augmented | qwen4b_mined | 40 | 48.8% | 260/533 | 44.6%-53.0% |
| qwen4b_static_augmented | qwen4b_mined | full | 50.1% | 267/533 | 45.9%-54.3% |
