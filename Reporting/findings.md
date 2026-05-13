AiDotNet vs Pytorch Benchmarks
==============================
Used two datasets one with 100 rows and another with 48000 rows

AiDotNet Benchmarks are better on small dataset, but Pytorch is better on large dataset in regrads to time taken to process the data.
About the CPU usage AiDotNet is more efficient on both datasets
About memory usage Pytorch is more efficient on both dataset

Accuracy is not compared as of now.

small dataset (100 rows):
- AiDotNet: ~190 milliseconds
- Pytorch: ~650 milliseconds
large dataset (48000 rows):
- AiDotNet: ~4100 milliseconds
- Pytorch: ~770 milliseconds

For more information about the benchmarks, please refer to the .json files in the folder reporting