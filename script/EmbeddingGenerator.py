import os
import torch
import pandas as pd
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from metrics import nearest_neighbor_analysis


model_fullnames = {  'llama2-7b':'meta-llama/Meta-Llama-2-7b-chat-hf',
                     'mistral-7B':"mistralai/Mistral-7B-Instruct-v0.3",
                     'llama3.1-8b':'meta-llama/Llama-3.1-8B-Instruct',
                     'Phi3.5-4b':"microsoft/Phi-3.5-mini-instruct",
                     'Qwen2.5-7B': "Qwen/Qwen2.5-7B-Instruct",
                     'gemma2-9b':"google/gemma-2-9b-it",
                     }

class EmbeddingGenerator():

    data_dir = 'dataset'

    def __init__(self, model, context, num_type_name, output_dir, token_method, results_dir=None, is_probing=False):
        self.prompt_data_name = context+'_'+num_type_name
        self.num_type_name = num_type_name
        self.model = model
        self.method = token_method
        self.data_df = self.read_prompt_data(self.prompt_data_name)
        self.embedding_dir = os.path.join(output_dir, 'embeddings')
        self.embedding = None
        self.results_dir = os.path.join(output_dir, results_dir)
        self.is_probing = is_probing

        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def read_prompt_data(self, prompt_data_name):
        prompt_data_file = prompt_data_name + '.jsonl'
        data_path = os.path.join(self.data_dir, prompt_data_file)
        assert os.path.exists(data_path), f"Data file {data_path} not found."
        data_df = pd.read_json(data_path, lines=True)
        return data_df
    
    def _single_input_for_single_token_prompt(self, instance):
            return str(instance['prompt']) + str(instance['number'])
    
    
    def get_number_token_idx_multi_token(self, input_text, tokenizer):
        # target_number = instance['number']
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"Original input text: {input_text}")
        re_expr = self.data_df.get('re', r'\d+').iloc[0]
        numbers = [match.span() for match in re.finditer(rf'{re_expr}', input_text)]
        print(f"Numbers: {numbers}")

        number_token_idx = []
        for start, end in numbers:
            digit_str = input_text[start:end]
            print(f"Number '{digit_str}'")
            current_token_indices = []
            token_accumulator = ""

            for i, token in enumerate(tokens):
                decoded_token = tokenizer.decode([input_ids[i]]).strip()
                token_accumulator += decoded_token
                if token_accumulator == digit_str:
                    current_token_indices.append(i)
                    number_token_idx.append(current_token_indices)
                    break
                elif digit_str.startswith(token_accumulator):
                    current_token_indices.append(i)
                else:
                    token_accumulator = ""
                    current_token_indices = []
        for num_span, token_idx in zip(numbers, number_token_idx):
            token_texts = [tokens[i] for i in token_idx]
            print(f"Number '{input_text[num_span[0]:num_span[1]]}' is represented as'{token_texts}', is  at index {token_idx}")
        print(f"Number token indices: {number_token_idx}")
        return number_token_idx
    

    def process_single_num_type(self, tokenizer, model, all_layer_embeddings, inputs_lst):
        emb_idx = self.data_df.get('idx', -1).iloc[0]
        print(f"emb_idx: {emb_idx}")
        for input_text in inputs_lst:
            layer_embedding = []
            number_token_indices = self.get_number_token_idx_multi_token(input_text, tokenizer)
            print(f"Number token indices: {number_token_indices}")
            encoded_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**encoded_inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                for layer in hidden_states:
                    if self.method == 'last':
                        # number_token_indices = [[1,2,3,4]], extract the last token at"4"
                        embedding = layer[:, number_token_indices[emb_idx][-1], :]
                        embedding = embedding.squeeze(dim=0) # remove batch dimension
                        layer_embedding.append(embedding)
                        print(f"embeddings shape: {np.array(embedding).shape}")
                    elif self.method == 'mean':
                        embedding = layer[0, number_token_indices, :].mean(dim=1)
                        embedding = embedding.squeeze(dim=0)
                        print(f"embeddings shape: {np.array(embedding).shape}")
                        layer_embedding.append(embedding)
            print(f"Layer embeddings shape: {np.array(layer_embedding).shape}") #should be (num_layers, embedding_dim)
            all_layer_embeddings.append(layer_embedding)        

    def run_pipeline(self):
        embedding_result_fpath = os.path.join(self.embedding_dir, self.model+"."+self.prompt_data_name+'.pt')
        if os.path.exists(embedding_result_fpath):
            print(f"Embedding file {embedding_result_fpath} already exists. loading pt...")
            self.embedding = torch.load(embedding_result_fpath)
            return self.embedding
        
        # method = 'mean'
        print(f'Start loading model: {self.model} and generating embedding: {self.prompt_data_name}.')
        tokenizer = AutoTokenizer.from_pretrained(model_fullnames[self.model],padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_fullnames[self.model])
        model.eval()
        all_layer_embeddings = []
        if self.num_type_name == 'single':
            inputs_list = self.data_df.apply(self._single_input_for_single_token_prompt, axis=1).to_list()
            self.process_single_num_type(tokenizer, model, all_layer_embeddings, inputs_list)
        elif self.num_type_name == 'singleDIY':
            inputs_list = self.data_df['prompt'].astype(str).to_list()
            self.process_single_num_type(tokenizer, model, all_layer_embeddings, inputs_list)
        else:
            raise Exception("Invalid num_type_name.")
        all_layer_embeddings = np.transpose(all_layer_embeddings, (1, 0, 2))
        torch.save(all_layer_embeddings, embedding_result_fpath)
        print(f"Embeddings saved. Shapes:{np.array(all_layer_embeddings).shape}")
        self.embedding = all_layer_embeddings
        return self.embedding
        
    def evaluate(self, start_idx=0, end_idx=None):
        results_fpath = os.path.join(self.results_dir, self.model+"."+self.prompt_data_name+'.jsonl')
        if end_idx is None:
            end_idx = len(self.embedding[0])
        embedding = self.embedding[:, start_idx:end_idx, :]
        if self.num_type_name == 'single' or self.num_type_name == 'singleDIY':
            labels = self.data_df['label'].tolist()[start_idx:end_idx]
        assert len(labels) == len(embedding[0]), f"Number of labels {len(labels)} does not match number of embeddings {len(embedding[0])}"

        layers = len(embedding)
        print(f"Number of layers: {layers}")
        results = []
        for i in range(layers):
            exp_info = {
                'model': self.model,
                't_method': self.method,
                'prompt': self.prompt_data_name,
                'layer': i,
            }
            print(f"Layer {i}")
            embs = embedding[i]
            orderness = nearest_neighbor_analysis(embs, labels)
            result = {
                        'model': self.model,
                        'method': self.method,
                        'prompt_data_name': self.prompt_data_name,
                        'embedding_layer': i,    
                        'orderness': orderness, 
                    }
            print(result)
            results.append(result)
        with open(results_fpath, 'w') as f:
            for result in results:
                f.write(json.dumps(result)+'\n')
