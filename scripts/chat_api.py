#!/usr/bin/env python3
"""HTTP API for chatting with Athena via Phi-3 LoRA adapter."""
import os, sys, json, time, traceback
import numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify
from flask_cors import CORS
from brain_client import BrainProxy

app = Flask(__name__)
CORS(app)
brain = BrainProxy(timeout=60)

# Lazy globals
_encoder = None
_phi3 = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print('[CHAT] Sentence transformer loaded', flush=True)
    return _encoder

def get_phi3():
    global _phi3
    if _phi3 is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        adapter_dir = os.environ.get('ADAPTER_DIR',
            '/workspace/nimcp/checkpoints/athena/phi3_adapter')
        base_model = 'microsoft/phi-3-mini-4k-instruct'

        print('[CHAT] Loading Phi-3 + LoRA...', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32,
            device_map='auto', trust_remote_code=False)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        projection = torch.nn.Linear(4096, model.config.hidden_size).cuda()
        proj_path = os.path.join(adapter_dir, 'projection.pt')
        if os.path.exists(proj_path):
            projection.load_state_dict(torch.load(proj_path, map_location='cuda'))
        projection.eval()

        _phi3 = {'model': model, 'tokenizer': tokenizer, 'projection': projection}
        print(f'[CHAT] Phi-3 ready (hidden={model.config.hidden_size})', flush=True)
    return _phi3


def decode_brain_output(brain_vec, prompt_text, max_tokens=128):
    """Decode brain output vector to text via Phi-3 LoRA."""
    import torch
    phi = get_phi3()
    model, tokenizer, projection = phi['model'], phi['tokenizer'], phi['projection']

    # Normalize brain vector to prevent overflow
    bv = torch.tensor(brain_vec[:4096], dtype=torch.float32).unsqueeze(0).cuda()
    bv = bv / (torch.norm(bv) + 1e-8)

    # Project to embedding space (FP32)
    brain_emb = projection(bv).unsqueeze(1)  # (1, 1, hidden)

    # Encode prompt
    prompt = f'Respond to: {prompt_text[:80]}\nResponse:'
    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = tokens['input_ids'].cuda()
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # Prepend brain embedding
    combined = torch.cat([brain_emb, inputs_embeds], dim=1)
    attn = torch.ones(1, combined.shape[1], device='cuda', dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined, attention_mask=attn,
            max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)

    text = tokenizer.decode(out[0][combined.shape[1]:], skip_special_tokens=True).strip()
    return text


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        # Encode to brain input
        emb = get_encoder().encode(message).tolist()
        features = [0.0] * 1024
        for i in range(0, 1024, len(emb)):
            n = min(len(emb), 1024 - i)
            features[i:i+n] = emb[:n]

        # Brain inference
        r = brain._send({'cmd': 'decide_full', 'features': features})
        result = r.get('result', r)
        output_vec = result.get('output_vector', [])

        norm = float(np.linalg.norm(output_vec[:100])) if output_vec else 0.0
        response_text = ''

        if output_vec and any(abs(x) > 1e-8 for x in output_vec[:100]):
            # Method 1: Try Phi-3 LoRA decode
            try:
                response_text = decode_brain_output(output_vec, message)
            except Exception as e:
                print(f'[CHAT] Phi-3 decode failed: {e}', flush=True)

            # Method 2: Try brain's native speak/generate_text
            if not response_text:
                try:
                    gen = brain._send({'cmd': 'generate_text',
                                       'semantic_input': output_vec[:1024]})
                    response_text = gen.get('text', '')
                except Exception:
                    pass

            # Method 3: Try vocabulary decode via speak
            if not response_text:
                try:
                    speak = brain._send({'cmd': 'speak', 'features': features})
                    response_text = speak.get('text', speak.get('decoded', ''))
                except Exception:
                    pass

            # Method 4: Describe what the brain produced
            if not response_text:
                label = result.get('label', '')
                conf = result.get('confidence', 0)
                response_text = (f'[Neural pattern: "{label}" '
                                 f'(confidence={conf:.2f}, norm={norm:.1f})]')

            print(f'[CHAT] "{message[:50]}" → "{response_text[:80]}"', flush=True)

        if not response_text:
            response_text = '[No neural activity — brain may be loading]'

        return jsonify({
            'response': response_text,
            'confidence': min(norm / 10.0, 1.0),
            'output_norm': norm,
            'timestamp': time.time()
        })
    except Exception as e:
        print(f'[CHAT] Error: {e}', flush=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    try:
        return jsonify(brain._send({'cmd': 'status'}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('[CHAT] Pre-warming...', flush=True)
    get_encoder()
    get_phi3()
    print('[CHAT] Starting on port 8080', flush=True)
    app.run(host='0.0.0.0', port=8080, threaded=True)
