## Prerequisites

### Setting up the environment:
Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment using Python version 3.8 (optional):

```shell
$ virtualenv -p python3.8 venv
$ source venv/bin/activate
```

Before installing the required packages, if you wish to run the training or prediction with the causal model via prompting, please follow this next step:
```
$ export CUDA_HOME=/path/to/CUDA
```
Otherwise, you can skip this.

Then, install all the required packages:
```shell
$ pip install -r requirements.txt
```
You might need `flash_attn` package as well which can be downloaded with `pip install flash_attn`.
By default, we do not use the prompting mechanism as it takes a long time and has a worse performance.

There are two final steps, for the setup, please download the `models` folder from this Google Drive [link](https://drive.google.com/drive/folders/1l3aLZKx6CEmrw2CkTRApKoM39RnwrjU8?usp=drive_link), which contains two fine-tuned models. Make sure they are located in a directory called "models". The model named `classifier` is the fine-tuned BERTweet model which we use when we apply the ensemble mechanism. The other model is the best-performing fine-tuned model. Besides downloading the model, please take the Tweet dataset and make sure they are located in a folder named "data". __These two parts are crucial, or the prediction will not work. Please email luca.mouchel@epfl.ch in case you have troubles__.

### Training 
By default, training is done with a RoBERTa model pre-trained for sentiment analysis on 124M tweets (`cardiffnlp/twitter-roberta-base-sentiment-latest`). You can directly run the `project/src/scripts/train.py` file and launch the training with its default parameters. Evaluation is done every 1000 steps.

However, training is very customizable, and you can train with different models from the command line, simply run the following command:
```
python project/src/scripts/train.py --language-model <> --epochs <> --batch-size <> --val-batch-size <> --lr <> --gradient-accumulation <>
```  
`language-model` is the HuggingFace model name and can be any pretrained-model which is suitable for classification (e.g., `bert-base-cased`, or `vinai/bertweet-base`).

By default, we also use the ensemble method and not the prompting one, you can change this manually at the top of the `Trainer.py` file.

We advise not to train any model, because it will take a very long time.
### Testing
You can also launch the `project/src/scripts/predict.py` without any arguments to launch the prediction on the test set, the output csv will be saved at the root of the project. Launching this will use both models saved in the `models` directory you downloaded from GDrive and will produce a csv which scores 0.899 on AICrowd.

If you do not give any arguments, it will use the best-performing model we used and do the prediction on the test set. 

If you train the model with another language model, say `vinai/bertweet-base`, then the finetuned model will be saved during training to `models/vinai_bertweet-base` and you can launch the prediction with this model by using: 

```
python project/src/scripts/predict.py --model-dir models/vinai_bertweet-base --per_gpu_eval_batch_size <>
```

## Model Pre-testing

Before choosing a model, it is advisable to test each candidate on a sub-sample of the main dataset. Running the below command will train your model on a sub-sample of 14'080 tweets, and will test on 4'400 tweets:

```shell
$ python project/src/testingScripts/train_test.py --language-model <> --epochs <> --batch-size <> --val-batch-size <> --lr <> --gradient-accumulation <>
```
where ```--language-model``` is the HuggingFace model name, and must be a model suitable for text classification, as above.

**Note:** For our purposes, we used the following parameters:

| Parameter          | Value  |
|:------------------:|:------:|
|```--epochs```      |  4     |
|```--batch_size```  | 32     |

with all other parameters as defaults.

To predict using the above-trained model, run:

```
python project/src/testingScripts/predict_test.py --model-dir testing_models/<> --per_gpu_eval_batch_size <>
```

The ```--model_dir``` parameter will take the directory of the saved pre-trained model as a parameter. This is of the form ```testing_models/{model name as used for training}```. This is the same form as above, i.e. If you train the model say `vinai/bertweet-base`, then the finetuned model will be saved during training to `testing_models/vinai_bertweet-base`.

Finally, to validate the results of the test, run the following command:

```
python project/src/testingScripts/model_testing.py --model-name <>
```

With ```language-model``` being the same path as used in the training step. This will then print the F1-score and accuracy for the fine-tuned model.

