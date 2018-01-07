## Build

Install python 3, pip3, and run:

```
pip3 install -r requirements.txt
``

**FIXME: This requirements.txt file is incomplete, as I added it after the fact.
Please add to it what is missing. Also, there are certainly native packages
needed; for example, python3-tk! Not sure what else...**

## Run

To run, first generate pre-processed data (see the `generate` directory
sibling to this one) and put some in a `train` directory, some in a `validate` directory, and some in a `test` directory. Then run:

```
python3 train.py
```

This will train the model on the data, then save the model to a file `model.h5`,
then run it on test data, showing (using `Tk`) a visualization of the
activations that lead to the predictions.

## Sources

This is based mostly on this:

https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/image_ocr.ipynb
