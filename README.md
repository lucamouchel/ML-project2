## Prerequisites

### Setting up the environment:
Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
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

### Training 
By default, training is done with a RoBERTa model pretrained for sentiment analysis on 124M tweets (`cardiffnlp/twitter-roberta-base-sentiment-latest`). You can directly run the `project/src/scripts/train.py` file and launch the training with its default parameters. Evaluation is done every 1000 steps.

However, training is very customizable, and you can train with different models from the command line, simply run the following command:
```
python project/src/scripts/train.py --language-model <> --epochs <> --batch-size <> --val-batch-size <> --lr <> --gradient-accumulation <>
```
`language-model` is the HuggingFace model name and can be any pretrained-model which is suitable for classification (e.g., `bert-base-cased`, or 'vinai/bertweet-base').

By default, we also use the ensemble method and not the prompting one, you can change this manually at the top of the `Trainer.py` file.
### Testing
You can also launch the `project/src/scripts/predict.py` without any arguments to launch the prediction. By default, it will use the fine-tuned folder saved by the training output which was trained without any outputs.
If you train the model with another language model, say `vinai/bertweet-base`, then the finetuned model will be saved during training to `models/vinai_bertweet-base` and you can launch the prediction with this model by using: 

```
python project/src/scripts/predict.py --model-dir models/vinai_bertweet-base --per_gpu_eval_batch_size <>
```
