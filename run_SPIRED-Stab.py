import os
import click
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from scripts.model import SPIRED_Stab
from scripts.utils_train_valid import getStabDataTest


aa_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X':'ALA'}

working_directory = os.path.abspath(os.path.dirname(__file__))

@click.command()
@click.option('--fasta_file', required = True, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(fasta_file, saved_folder):
    
    # load parameter
    model = SPIRED_Stab(device_list = ['cpu', 'cpu', 'cpu', 'cpu'])
    model.load_state_dict(torch.load(f'{working_directory}/data/model/SPIRED-Stab.pth'))
    model.eval()
    
    # load ESM-2 650M model
    esm2_650M, _ = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
    esm2_650M.eval()
    
    # load ESM-2 3B model
    esm2_3B, esm2_alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
    esm2_3B.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()
    
    # save sequence information
    # load fasta file
    id_list = []
    seq_list = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        id_list.append(record.id)
        seq_list.append(str(record.seq))

    # write pred value by appending to a csv file
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    
    with open(f'{saved_folder}/pred.csv', 'w') as f:
        f.write('id,ddG,dTm\n')

        for id, seq in zip(id_list, seq_list):
            wt_seq, mut_seq = seq.split(':')

            # wt_seq = str(list(SeqIO.parse(fasta_file, 'fasta'))[0].seq)
            # mut_seq = str(list(SeqIO.parse(fasta_file, 'fasta'))[1].seq)
            
            mut_pos_torch_list = torch.tensor((np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist())
            
            # predict
            with torch.no_grad():
                
                # data
                f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(wt_seq, esm2_3B, esm2_650M, esm2_batch_converter)
                wt_data = {
                    'target_tokens': target_tokens,
                    'esm2-3B': f1d_esm2_3B,
                    'embedding': f1d_esm2_650M
                }
                f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(mut_seq, esm2_3B, esm2_650M, esm2_batch_converter)
                mut_data = {
                    'target_tokens': target_tokens,
                    'esm2-3B': f1d_esm2_3B,
                    'embedding': f1d_esm2_650M
                }
                ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)

                # write to csv
                f.write(f'{id},{ddG.item()},{dTm.item()}\n')
        

if __name__ == '__main__':
    main()
