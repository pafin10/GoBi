from transformers import AutoTokenizer, AutoModel
import torch
import scanpy as sc
import os


def load_sequences(fasta_dir):
    all_sequences = []  # A single list to store all sequences
    for fasta_file in os.listdir(fasta_dir):  # Iterate through all files in the directory
        if fasta_file.endswith('.fasta') or fasta_file.endswith('.txt'):  # Only process FASTA files
            file_path = os.path.join(fasta_dir, fasta_file)
            with open(file_path, 'r') as f:
                seq = ''
                for line in f:
                    if line.startswith('>'):  # Sequence header line
                        if seq:  # If there's a current sequence, add it
                            all_sequences.append(seq)
                        seq = ''  # Reset for the new sequence
                    else:
                        seq += line.strip()  # Add sequence lines to seq
                if seq:  # Don't forget to append the last sequence
                    all_sequences.append(seq)
    return all_sequences  # Return a single list of all sequences


if __name__ == '__main__':
    fasta_dir = 'fasta_dir'
    # Load a pretrained protein language model
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    model = AutoModel.from_pretrained("Rostlab/prot_bert")

    # Example FASTA sequence
    sequences = load_sequences(fasta_dir)
    seqs = [str(v) for v in sequences]
    print(type(seqs))  # Show type and the first few elements

    #print(seqs)

    encoded_input = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
    print(encoded_input['input_ids'].shape) 
    output = model(**encoded_input)

    # Extract embeddings (e.g., CLS token representation)
    embeddings = output.last_hidden_state.mean(dim=1).detach().numpy()

    print(embeddings.shape)  # Show the shape of the embeddings
    # Create AnnData object
    adata = sc.AnnData(embeddings)
    adata.obs['sequence_id'] = seqs  # This will assign the sequence ids directly
    print(adata.X.shape)

    # UMAP
    sc.pp.pca(adata, n_comps=10)  # You can adjust n_comps as needed
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    sc.pl.umap(adata, save='umap_plot.png')
