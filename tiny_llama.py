import math
import json
import torch
from tqdm import tqdm
from typing import List, Union
from tinygrad import Tensor
from safetensors import safe_open

folder = "./Llama-3.1-8B-Instruct/"

with open(folder+"tokenizer.json", "r") as file:
    data = json.load(file)

vocab = data["model"]["vocab"]

#very basic tokenizer, not as good as the transformers one but outputs the exact same for most inputs
class tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.devocab = {value: key for key, value in vocab.items()}

    def tokenize(self, x: str):
        #put in special characters
        lx = list(x.replace(" ", "Ġ").replace("\n", "Ċ"))
        
        #make largest possible tokens
        while True:
            lxn = []
            index = 0
            wasUpdated = False
            while index < len(lx):
                if index < len(lx) - 1 and lx[index] + lx[index+1] in self.vocab:
                    lxn.append(lx[index] + lx[index+1])
                    index+=2
                    wasUpdated = True
                else:
                    lxn.append(lx[index])
                    index+=1
            lx = lxn
            if not wasUpdated:
                break
        
        #encode them and add start token
        return Tensor([128000] + [self.vocab[i] for i in lx])
    
    def detokenize(self, x: Union[int, List[int]]):
        if isinstance(x, int):
            raw = self.devocab[x]
        else:
            raw = "".join([self.devocab[i] for i in x])
        
        #take out special characters
        raw = raw.replace("Ġ", " ").replace("Ċ", "\n")
        return raw

num_shards = 4

def five_digit(x):
    res = str(x)
    assert len(res) <= 5
    while len(res) < 5:
        res = "0" + res
    return res

weight_files = [f"model-{five_digit(i+1)}-of-{five_digit(num_shards)}.safetensors" for i in range(num_shards)]

all_weights = {}

for file in weight_files:
    with safe_open(folder+file, framework="pt") as f: 
        for key in f.keys():
            all_weights[key] = Tensor(f.get_tensor(key).to(torch.float32).numpy())

with open(folder+"config.json", "r") as file:
    config = json.load(file)

class tiny_llama:
    def __init__(self, config, weights, tokenizer):
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self.end_tokens = [128001, 128008, 128009]
        self.inv_freq = self.get_rope_params()

    def get_rope_params(self):
        head_dim = self.config["hidden_size"] / self.config["num_attention_heads"]
        inv_freq = 1.0 / (self.config["rope_theta"] ** (Tensor.arange(0, head_dim, 2) / head_dim))

        factor = config["rope_scaling"]["factor"]
        low_freq_factor = config["rope_scaling"]["low_freq_factor"]
        high_freq_factor = config["rope_scaling"]["high_freq_factor"]
        old_context_len = config["rope_scaling"]["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = (wavelen > low_freq_wavelen).where(inv_freq / factor, inv_freq)

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = is_medium_freq.where(smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def rms_norm(self, x, gamma) :
        mean_square = (x ** 2).mean(axis=-1, keepdim=True)
        rms = (mean_square + self.config["rms_norm_eps"]).sqrt()
        normalized = x / rms
        return normalized * gamma

    def rotate_half(self, x):
        head_dim = x.shape[-1]
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        return Tensor.cat(-x2, x1, dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def infer_on_tokens(self, tokens):
        #make embeddings
        input_len = tokens.shape[0]
        head_size = self.config["hidden_size"] // self.config["num_attention_heads"]
        oh_tokens = tokens.one_hot(self.config["vocab_size"])
        embeddings = oh_tokens @ self.weights["model.embed_tokens.weight"]

        #get the stuff ready for RoPE
        position_ids = Tensor.arange(0, input_len).unsqueeze(0)
        inv_freq_expanded = self.inv_freq.reshape(1, -1, 1)
        position_ids_expanded = position_ids.unsqueeze(1)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1,2)
        emb = freqs.cat(freqs, dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        for i in range(self.config["num_hidden_layers"]):
            normed = self.rms_norm(embeddings, self.weights[f"model.layers.{i}.input_layernorm.weight"])

            #make the queries, keys and values
            queries = normed @ self.weights[f"model.layers.{i}.self_attn.q_proj.weight"].T
            keys = normed @ self.weights[f"model.layers.{i}.self_attn.k_proj.weight"].T
            values = normed @ self.weights[f"model.layers.{i}.self_attn.v_proj.weight"].T
            
            #reshape them
            queries = queries.reshape(input_len, self.config["num_attention_heads"], head_size).transpose(0, 1).unsqueeze(0)
            keys = keys.reshape(input_len, self.config["num_key_value_heads"], head_size).transpose(0, 1).unsqueeze(0)
            values = values.reshape(input_len, self.config["num_key_value_heads"], head_size).transpose(0, 1).unsqueeze(0)

            #apply RoPE
            queries, keys = self.apply_rotary_pos_emb(queries, keys, cos, sin)
            
            #repeat keys and values
            repeat_num = self.config["num_attention_heads"] // self.config["num_key_value_heads"]
            keys = keys[:, :, None, :, :].expand(1, self.config["num_key_value_heads"], repeat_num, input_len, head_size).reshape(1, self.config["num_attention_heads"], input_len, head_size)
            values = values[:, :, None, :, :].expand(1, self.config["num_key_value_heads"], repeat_num, input_len, head_size).reshape(1, self.config["num_attention_heads"], input_len, head_size)

            #gets scores, apply causal mask, softmax to get attention weights
            scores = queries @ keys.transpose(-2, -1) / (head_size ** 0.5)
            causal_mask = Tensor.ones(input_len, input_len).tril().reshape(1, 1, input_len, input_len)
            scores = scores + (1 - causal_mask) * -1e9 #negative "infinity" mask
            attention_weights = scores.softmax(axis=-1)

            #matmul with values and output
            attention_mid_output = (attention_weights @ values).transpose(1,2).reshape(input_len, -1)
            output_weights = self.weights[f"model.layers.{i}.self_attn.o_proj.weight"].T
            attention_output = attention_mid_output @ output_weights

            #residual connection, norm
            attended = embeddings + attention_output
            normed_attended = self.rms_norm(attended, self.weights[f"model.layers.{i}.post_attention_layernorm.weight"])

            #ffwd with down(silu(x @ gate) * x @ up)
            gate_weights = self.weights[f"model.layers.{i}.mlp.gate_proj.weight"].T
            up_weights = self.weights[f"model.layers.{i}.mlp.up_proj.weight"].T
            down_weights = self.weights[f"model.layers.{i}.mlp.down_proj.weight"].T

            gate_output = (normed_attended @ gate_weights).silu()
            up_output = normed_attended @ up_weights

            hidden_layer = gate_output * up_output
            
            ff_output = hidden_layer @ down_weights

            #residual
            embeddings = attended + ff_output
        
        #final norm
        final_normed = self.rms_norm(embeddings, self.weights["model.norm.weight"])

        #llm head
        logits = final_normed @ self.weights["lm_head.weight"].T
        last_token = logits[-1].argmax() #deterministic (temperature=0) for now
        return last_token
    
    def __call__(self, x):
        tokens = self.tokenizer.tokenize(x)
        for _ in range(100):
            new_token = self.infer_on_tokens(tokens)
            tokens = tokens.cat(new_token.reshape(1), dim=0)
        print(self.tokenizer.detokenize(tokens.tolist()[1:]))

t = tokenizer(vocab)
tl = tiny_llama(config, all_weights, t)

tl("Bob: What's up, man? How are you doing? \nCharlie:")
