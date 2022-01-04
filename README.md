# Artificial Neural Networks

## How to run:

### Part 1
1. In the terminal, run `python3 part1_nn_lib.py`


### Part 2
We have implemented part 2 to contain multiple functions.

1. To load the tuned neural network, and to **evaluate** the performance, run `python3 part2_house_value_regression.py "evaluate"` <br>

2. To run hyperparameter search within a **defined** hyperparameters in the code, run `python3 part2_house_value_regression.py "grid" 100 "record.xlsx"` <br>

3. To run hyperparameter search within **random** hyperparameters in a larger hyperparameters space, run `python3 part2_house_value_regression.py "random" 100 "record.xlsx"` <br>

    - **NOTES**:
    - This will save and replace the **part2_model.pickle** 
    - This will save the performance of the models trained on the list of hyperparameters in an excel **record.xlsx**. Feel free to change the name of **record.xlsx** to a name preferred for the excel file
    - Only for no.3(random search of hyperparameteres). Replace the **100** with any numbers of iterations wanted. For example, replace **100** with **200** to run 200 runs of random search of hyperparameters


## Authors

* **Salim Al-Wahaibi** - *saa221@ic.ac.uk*
* **Wei Jie Chua** - *wc1021@ic.ac.uk*
* **Alicia Jiayun Law** - *ajl115@ic.ac.uk*
* **Marcos-Antonios Charalambous** - *mc921@ic.ac.uk*

## License
[MIT License](https://choosealicense.com/licenses/mit/)
