# Teacher Forcing Qwen30B -> Qwen4B Zigzag Summary

## ASR: static harmful-prefix eval

| iter | k=5 | k=10 | k=20 | k=40 |
|---:|---:|---:|---:|---:|
| 1 | 10.3% | 14.8% | 20.6% | 29.1% |
| 2 | 0.9% | 0.3% | 1.5% | 3.0% |
| 3 | 0.0% | 0.3% | 1.2% | 1.5% |

## Mining yields

| iter | role | examples | attempted | kept | keep rate | refusal reject | verification reject |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | teacher | 330 | 1041 | 581 | 55.8% | 10.1% | 34.1% |
| 1 | student | 330 | 1772 | 289 | 16.3% | 65.7% | 17.9% |
| 2 | teacher | 289 | 623 | 575 | 92.3% | 0.8% | 6.9% |
| 2 | student | 330 | 1951 | 25 | 1.3% | 97.5% | 1.2% |
| 3 | teacher | 25 | 50 | 50 | 100.0% | 0.0% | 0.0% |
| 3 | student | 50 | 216 | 72 | 33.3% | 42.1% | 24.5% |

## SFT training

| iter | rows | steps | tokens | first NLL | last NLL | min NLL | mean sec/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1338 | 167 | 1,629,161 | 2.096 | 0.558 | 0.536 | 2.21 |
| 2 | 2370 | 296 | 2,887,326 | 0.556 | 0.486 | 0.400 | 2.30 |
| 3 | 2406 | 300 | 2,929,095 | 0.437 | 0.464 | 0.296 | 2.33 |

## Final adaptive-prefix spot checks

- `final_on_iter3_teacher_prefixes.json`: keyword ASR 46.0% (23/50). Note: keyword ASR overcounts soft refusals that lack exact refusal strings. 
- `final_on_iter3_student_prefixes.json`: keyword ASR 55.6% (40/72). Note: keyword ASR overcounts soft refusals that lack exact refusal strings. 
