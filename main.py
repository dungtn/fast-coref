import os
import json
import tqdm
import wandb
import argparse
import jsonlines

import sys
sys.path.append('fast-coref/src')

from inference.model_inference import Inference


def main(args):
    # init model
    inference_model = Inference(
        args.model_path, 
        encoder_name=args.encoder_name
    )
    # run inference
    inp_fn = os.path.join(args.data_dir, args.filename)
    out_fn = os.path.join(args.out_dir, args.filename)
    mode = 'w'
    excluded_doc_ids = set()
    # to skip over some doc ids, put them in the 
    # `doc_ids.json` located in the output dir
    doc_fn = os.path.join(args.out_dir, 'doc_ids.json')
    if os.path.exists(doc_fn):
        excluded_doc_ids = json.load(open(doc_fn))
    # automatically skip over processed files
    if os.path.exists(out_fn):
        mode = 'a'
        with jsonlines.open(out_fn) as reader:
            for obj in tqdm.tqdm(reader):
                excluded_doc_ids.add(int(obj['doc_id']))
    with jsonlines.open(out_fn, mode=mode) as writer:
        with jsonlines.open(inp_fn) as reader:
            doc_id = 0
            doc = list()
            num_doc = 0
            for obj in tqdm.tqdm(reader):
                if int(obj['doc_id']) in excluded_doc_ids:
                    continue
                if int(obj['doc_id']) == doc_id:
                    doc.append(obj['text'])
                else:
                    if len(doc) > 0:
                        output = inference_model.perform_coreference(
                            '\n'.join(doc)
                        )
                        writer.write({
                            'doc_id': doc_id,
                            'coref_output': output
                        })
                        num_doc += 1
                        if args.use_wandb:
                            wandb.log({'num_doc': num_doc})
                    doc_id = int(obj['doc_id'])
                    doc = [obj['text']]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
        help="Path to the input data directory")
    parser.add_argument('--out_dir', type=str, required=True,
        help="Path to the output data directory")
    parser.add_argument('--filename', type=str, required=True, 
        help="The input filename")
    parser.add_argument('--model_path', type=str, required=True, 
        help="Path to the pretrained model")
    parser.add_argument('--encoder_name', type=str, default='', 
        help="The encoder's name")
    parser.add_argument("--use_wandb", type=int, default=0,
        help="Use WandDB?")
    args = parser.parse_args()
    args.use_wandb = args.use_wandb == 1
    if args.use_wandb:
        wandb.init(project='FastCoref')
    main(args)

