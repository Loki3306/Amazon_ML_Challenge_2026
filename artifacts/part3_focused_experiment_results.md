# Focused Experiment Results: nprobe=1024, K=300

- Dense Pair Recall = 85.46%
- Hybrid Pair Recall = 86.92%
- Dense Query Recall = 97.04%
- Hybrid Query Recall = 97.44%
- Average candidates/S1 = 302.84
- Total candidates = 28,580,084
- Search runtime = 2027.7s

Conclusion: Increasing nprobe to 1024 from 512 only provided a +0.76% improvement in hybrid pair recall but doubled the search runtime. Thus, nprobe=512, K=300 is the superior hyperparameter setting.
