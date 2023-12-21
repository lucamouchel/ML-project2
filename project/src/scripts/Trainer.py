import logging
import torch
import sys
sys.path.append(".")
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    AutoModel,
    get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
)

from load_data import DatasetLoader
from project.src.utils import metrics as metrics
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

USE_ENSEMBLE = True
USE_CAUSAL_MODEL = False

class Classifier:
    def __init__(self,
                 output_model_dir,
                 pretrained_model_name_or_path,                 
                 cache_dir='data/pretrained/',
                 do_lower_case=True):
        
      
        self.output_model_dir = output_model_dir

        self.logger = logging.getLogger(__name__)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

        # Setup CUDA, GPU & distributed training
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                     do_lower_case=do_lower_case,
                                                     cache_dir=self.cache_dir)
        
        self.data_loader = DatasetLoader(tokenizer=self.tokenizer)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def train(self, 
              train_batch_size,
              gradient_accumulation_steps,
              num_train_epochs,
              learning_rate,
              weight_decay=0.01,
              warmup_steps=100,
              adam_epsilon=1e-6,
              max_grad_norm=10.0):

        """ Train the model """
        
        train_dataset, _ = self.data_loader.load_dataset('train')
        val_dataset, val_labels = self.data_loader.load_dataset('dev')
        
        train_sampler = RandomSampler(train_dataset) 
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path, num_labels=2, ignore_mismatched_sizes=True, cache_dir='models/cache')
        model.to(self.device)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * gradient_accumulation_steps
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
       
        best_acc = 0
        best_f1 = 0

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=False
        )
        save_steps = 10
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
             
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
                
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if save_steps > 0 and global_step % save_steps == 0:
                    # Log metrics
                        # Only evaluate when single GPU otherwise metrics may not avg well
                        preds = self._predict(eval_dataset=val_dataset,
                                                per_gpu_eval_batch_size=train_batch_size,
                                                model=model)
                            
                        accuracy, f1 = metrics.compute(predictions=preds, labels=val_labels)

                        if accuracy > best_acc or f1 > best_f1: 
                            print("Saving model checkpoint to %s", self.output_model_dir)
                            model_to_save = (   
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(self.output_model_dir)
                        
                            print(f"accuracy improved, previous {best_acc}, new one {accuracy}")
                            print(f"f1 improved, previous {best_f1}, new one {f1}") 
                            best_acc = accuracy
                            best_f1 = f1
                        else:
                            print(f"accuracy not improved, best: {best_acc}, this one: {accuracy}")
                            print(f"f1 not improved, best: {best_f1}, this one: {f1}")
                            
        return global_step, tr_loss / global_step

    def predict(self, per_gpu_eval_batch_size):
        test_dataset, _ = self.data_loader.load_dataset('test')
        model = AutoModelForSequenceClassification.from_pretrained(self.output_model_dir, num_labels=2, ignore_mismatched_sizes=True)
        preds = self._predict(eval_dataset=test_dataset,
                             per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                             model=model)
        return preds

    def _predict(self,
                 eval_dataset,
                 model,
                 per_gpu_eval_batch_size):
        
        eval_batch_size = per_gpu_eval_batch_size * max(1, self.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        model.to(self.device)
        if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        preds = []
        confident = 0
        
        if USE_CAUSAL_MODEL:
            model_id = "microsoft/phi-2"
            causal_tokenizer = AutoTokenizer.from_pretrained(model_id)
            pretrained_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        if USE_ENSEMBLE:
            finetuned_model_dir = 'models/classifier'
            bertweet = AutoModelForSequenceClassification.from_pretrained(finetuned_model_dir, num_labels=2)
            bertweet_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir, use_fast=False)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                if self.n_gpu > 1:
                    outs = model.module(input_ids=batch[0].to(self.device), attention_mask=batch[1].to(self.device))                    
                else:
                    outs = model(input_ids=batch[0].to(self.device), attention_mask=batch[1].to(self.device))
                logits = outs.logits.to(self.device)
                probs = torch.softmax(logits, dim=1) 

                if not (USE_ENSEMBLE or USE_CAUSAL_MODEL):
                    probas = torch.softmax(logits, dim=1)    
                    for proba in probas:
                        preds.append(torch.argmax(proba).cpu().detach().numpy())
                        continue
                    
                else:
                    for i, prob in enumerate(probs):
                        max_proba = torch.max(prob)
                        if max_proba > 0.85:
                            confident+=1
                            preds.append(torch.argmax(prob).cpu().detach().numpy())
                            continue
                        else:
                            if USE_CAUSAL_MODEL:
                                tweet = self.tokenizer.decode(batch[0][i], skip_special_tokens=True)
                                prompt = f"""You are a sentiment classifier. What is the sentiment of {tweet}. Reply with one word: positive or negative """
                                inputs = causal_tokenizer(prompt, return_tensors="pt")
                                outputs = pretrained_model.generate(**inputs, max_new_tokens=5)
                                output_text = causal_tokenizer.decode(outputs[0], skip_special_tokens=True)
                                if 'negative' in output_text:
                                    preds.append(0)
                                    continue
                                elif 'positive' in output_text:
                                    preds.append(1)
                                    continue
                                else:
                                    print("NOT OKAY")
                                    preds.append(1)
                                    continue
                                
                            elif USE_ENSEMBLE:
                                tweet = self.tokenizer.decode(batch[0][i], skip_special_tokens=True)
                                bertweet_inputs = bertweet_tokenizer(tweet, return_tensors="pt")
                                output = bertweet(**bertweet_inputs)
                                bertweet_logits = output.logits.to(self.device)
                                bertweet_probas = torch.softmax(bertweet_logits, dim=1).cpu().detach().numpy()[0]
                                import numpy as np
                                argmax_bertweet = np.argmax(bertweet_probas)
                                
                                if bertweet_probas[argmax_bertweet] > 0.85:
                                    preds.append(argmax_bertweet)
                                    continue
                            
                                prob = prob.cpu().detach().numpy()
                                PROBAS = 0.7*prob + 0.3*bertweet_probas
                                preds.append(np.argmax(PROBAS))       
        return preds