from flask import Flask
from flask import jsonify
from flask import request
from src.loren import Loren
from nltk.tokenize import sent_tokenize
from huggingface_hub import snapshot_download

import os
import torch

model = snapshot_download('Jiangjie/loren')
config = {
    "n_gpu": torch.cuda.device_count(),
    "model_type": "roberta",
    "model_name_or_path": "roberta-large",
    "logic_lambda": 0.5,
    "prior": "random",
    "mask_rate": 0.0,
    "cand_k": 3,
    "max_seq1_length": 256,
    "max_seq2_length": 128,
    "max_num_questions": 8,
    "do_lower_case": False,
    "seed": 42,
    'fc_dir': os.path.join(model, 'fact_checking/roberta-large/'),
    'mrc_dir': os.path.join(model, 'mrc_seq2seq/bart-base/'),
    'er_dir': os.path.join(model, 'evidence_retrieval/')
}

loren = Loren(config, verbose=False)
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to LOREN.'

@app.route('/check', methods=['POST'])
def check():
    req = request.get_json()
    text = req.get('text', None)
    claims = []

    if text:
        for idx, sentence in enumerate(sent_tokenize(text)):
            js = loren.check(sentence)
            claims.append({
                'claim': js['claim'],
                'evidence': js['evidence'],
                'score': veracityWeight(js['claim_veracity'])
            })

    return jsonify({ 'claims': claims })

def veracityWeight(veracity):
    if veracity == "SUPPORTS": return 1
    elif veracity == "REFUTES": return -1
    else: return 0

if __name__ == "__main__":
    app.run(host="0.0.0.0")