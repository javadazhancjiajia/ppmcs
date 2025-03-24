# README 
This repository contains four test datasets in the `.dataset` directory: `lab`, `weather`, `sotck`, and `pems08`.
## Files 
- `ppmcs.py`: This is the main file for running the experiments. The entry function is `start_with_dp()`, which takes the dataset name `db_name` as its parameter.

Example usage: 
```python 
start_with_dp('lab')
```

In each round, we use two 2D arrays, `data1` and `data2`, to record the data published by users. Then, we store `data1` and `data2` in the directories `.log/ppmcs/{db_name}/data1` and `.log/ppmcs/{db_name}/data2`, respectively, using the same filenames.

It is worth noting that we convert the 2D arrays into 1D arrays before storing them using NumPy, following the format:

```python
np_data1 = np.array(data1)
d1_f.write(';'.join(map(str, np_data1.flatten())) + "\n")
```

When conducting further data analysis, it is essential to pay attention to data format issues.

Specifically, our implementation of the square wave mechanism is based on the reference from https://github.com/vvv214/LDP_Protocols.git.
