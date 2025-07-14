import anndata as ad
import mudata as mu
import pandas as pd
import awkward as ak
from typing import Optional # Import Optional for type hinting
import numpy as np
import plotnine as p9
import sys
sys.path.insert(0,'~/lib')
import utils
import os

#====
import hashlib

def step1_load_tcr_annotation(filename, format_type='cellranger', barcode_suffix=None):
    if format_type == 'cellranger':
        df = pd.read_csv(filename)
    elif format_type == 'airr':
        df = pd.read_csv(filename, sep='\t', header=0)
    else:
        raise ValueError("Invalid format_type. Choose 'airr' or 'cellranger'.")

    if format_type == 'cellranger':
        column_mapping = {
            'chain': 'locus',
            'v_gene': 'v_call',
            'd_gene': 'd_call',
            'j_gene': 'j_call',
            'cdr3': 'junction_aa'
        }
        df.rename(columns=column_mapping, inplace=True)
    
    if 'barcode' not in df.columns:
        if format_type == 'airr':
            df['barcode'] = barcode_suffix if barcode_suffix else os.path.basename(filename).replace('.tsv', '')
        else:
            raise ValueError(f"'{filename}' ({format_type}) is missing the required 'barcode' column.")
    
    if barcode_suffix:
        df['barcode'] = df.barcode.astype(str) + '_' + barcode_suffix

    if 'productive' in df.columns:
        df['productive'] = df['productive'].astype(str).str.upper().map({'T': True, 'TRUE': True, 'F': False, 'FALSE': False}).fillna(False)
    else:
        df['productive'] = True

    df_processed = df[df['productive'] == True].copy()
    
    print(f'Read {df_processed.shape[0]} productive chains from {filename}')

    def build_custom_chain_id(row):
        d_gene_val = row['d_call']
        d_gene_part = str(d_gene_val) if pd.notna(d_gene_val) and str(d_gene_val).strip() != '' else ''
        return f"{row['locus']}:{row['v_call']}__{d_gene_part}__{row['j_call']}__{row['junction_aa']}"

    df_processed['custom_clone_id'] = df_processed.apply(build_custom_chain_id, axis=1)
    
    df_processed['TRA_custom_id'] = ''
    df_processed['TRB_custom_id'] = ''
    df_processed['custom_clonotype_id_full'] = ''
    df_processed['category'] = None 

    df_processed.loc[df_processed['locus'].isin(['TRA', 'TRAC']), 'TRA_custom_id'] = df_processed['custom_clone_id']
    df_processed.loc[df_processed['locus'].isin(['TRB', 'TRBC']), 'TRB_custom_id'] = df_processed['custom_clone_id']

    df_processed['custom_clonotype_id_full'] = df_processed['custom_clone_id']

    # Add has_tra and has_trb columns
    df_processed['has_tra'] = df_processed['locus'].isin(['TRA', 'TRAC'])
    df_processed['has_trb'] = df_processed['locus'].isin(['TRB', 'TRBC'])

    result_df = df_processed[['barcode', 'TRA_custom_id', 'TRB_custom_id', 'custom_clonotype_id_full', 'category', 'has_tra', 'has_trb']].copy()
    
    return result_df

#-
import hashlib

def create_unique_clonotype_id_across_samples(list_of_dfs_with_sample_info):
    '''
    input is a list of df, where item is a dictionary containing two keys df and sample_id
    '''
    all_clonotypes_data = []
    sample_names = []

    for item in list_of_dfs_with_sample_info:
        df_sample = item['df']
        sample_id = item['sample_id']

        clonotype_counts = df_sample['custom_clonotype_id_full'].value_counts().reset_index()
        clonotype_counts.columns = ['custom_clonotype_id_full', sample_id]
        all_clonotypes_data.append(clonotype_counts)
        sample_names.append(sample_id)

    if not all_clonotypes_data:
        return pd.DataFrame()

    merged_clonotypes = all_clonotypes_data[0]
    for i in range(1, len(all_clonotypes_data)):
        merged_clonotypes = pd.merge(merged_clonotypes, all_clonotypes_data[i], on='custom_clonotype_id_full', how='outer')

    merged_clonotypes.fillna(0, inplace=True)

    for sample in sample_names:
        merged_clonotypes[sample] = merged_clonotypes[sample].astype(int)

    unique_clonotypes = merged_clonotypes['custom_clonotype_id_full'].unique()
    uci_mapping = {
        clonotype: f"uci_{i:06d}" for i, clonotype in enumerate(unique_clonotypes, 1)
    }
    merged_clonotypes['uci_clonotype_id'] = merged_clonotypes['custom_clonotype_id_full'].map(uci_mapping)

    # Calculate 'is_present_in_multiple_samples'
    merged_clonotypes['is_present_in_multiple_samples'] = (merged_clonotypes[sample_names] > 0).sum(axis=1) > 1

    # Calculate 'num_samples_present_in'
    merged_clonotypes['num_samples_present_in'] = (merged_clonotypes[sample_names] > 0).sum(axis=1)

    final_columns = ['uci_clonotype_id'] + sample_names + ['is_present_in_multiple_samples', 'num_samples_present_in', 'custom_clonotype_id_full']
    result_df = merged_clonotypes[final_columns].copy()
    
    return result_df

# This is for 10x cellranger filtered_sample_contig
def step1_load_filtered_sample_contig( filename_filtered_contig_annotation, barcode_suffix=None): # if barcode_suffix is x then barcode = barcode + '_' + barcode_suffix
    df = pd.read_csv(filename_filtered_contig_annotation)
    
    if barcode_suffix:
        df['barcode'] = df.barcode + '_' + barcode_suffix
        
    sample_id=utils.parse_sample_id ( filename_filtered_contig_annotation )
    print(f'Read {df.shape[0]} clones ({sample_id })')
    def build_custom_chain_id(row):
        return f"{row['chain']}:{row['v_gene']}__{row['d_gene']}__{row['j_gene']}__{row['cdr3']}"
    
    # Helper function to determine the category based on counts of TRA and TRB chain IDs.
    def determine_category(tra_ids, trb_ids):
        n_tra = len(tra_ids)
        n_trb = len(trb_ids)
        if n_tra > 0 and n_trb > 0:
            if n_tra == 1 and n_trb == 1:
                return "proper_pair"
            else:
                return "multichain"
        elif n_tra > 0 and n_trb==0:
            return "orphan" if n_tra == 1 else "multi_orphan" # 
        elif n_trb > 0 and n_tra==0:
            return "orphan" if n_trb == 1 else "multi_orphan"
        else:
            return None

    # Work on a copy to avoid modifying the original DataFrame.
    #d
    f = df.copy()
    
    # Create the custom clone ID for every row.
    df['custom_clone_id'] = df.apply(build_custom_chain_id, axis=1) # updates input df
    
    records = []
    
    # Group by barcode. Each group represents a cell with one or more chains.
    for barcode, group in df.groupby('barcode'):
        # Extract custom IDs for TRA and TRB based on the 'chain' column.
        tra_ids = group.loc[group['chain'].str.startswith("TRA"), 'custom_clone_id'].tolist()
        trb_ids = group.loc[group['chain'].str.startswith("TRB"), 'custom_clone_id'].tolist()
        
        # Combine multiple IDs with a semicolon; otherwise, leave blank if none.
        tra_combined = ";".join(tra_ids) if tra_ids else ""
        trb_combined = ";".join(trb_ids) if trb_ids else ""
        
        # Determine the category for the cell.
        category = determine_category(tra_ids, trb_ids)
        
        records.append({
            "barcode": barcode,

            "TRA_custom_id": tra_combined,
            "TRB_custom_id": trb_combined,
            "custom_clonotype_id_full": tra_combined + "//" + trb_combined, # Only do this if it is not an orphan
            "category": category
        })
        
    result_df = add_promiscuity_columns( pd.DataFrame(records))
    #result_df['sample_id']=result_df.barcode.str.split('_').str[1]

    return result_df



def add_promiscuity_columns(df,col1='TRA_custom_id',col2='TRB_custom_id'): 
    """
    For each value in col1 and col2 of the DataFrame, compute its "promiscuity": 
    i.e. the number of unique partners it has in the other column.
    
    For example, if the DataFrame contains pairs such as:
      (A, X), (A, Y), (A, Y), (B, X), (B, Z)
    then:
      - Value A in col1 is paired with X and Y (2 partners)
      - Value B in col1 is paired with X and Z (2 partners)
      - Value X in col2 is paired with A and B (2 partners),
      - and so on.
    
    Note:
      - The function first drops duplicate pairs.
      - It then computes the number of unique partners per value.
    
    Args:
      df (pd.DataFrame): Input DataFrame containing at least columns col1 and col2.
      col1 (str): Name of the first column (default "column1").
      col2 (str): Name of the second column (default "column2").
      
    Returns:
      pd.DataFrame: The input DataFrame with two new columns: 
                    "col1_promiscuity" and "col2_promiscuity".
    """
    # Remove duplicates based on the pair (col1, col2)
    df_unique = df.drop_duplicates(subset=[col1, col2])
    
    # Compute the number of unique partners for each value in col1:
    # (Since df_unique now contains only unique (col1, col2) pairs, we can use groupby + count.)
    col1_prom_map = df_unique.groupby(col1)[col2].count()
    
    # Compute the number of unique partners for each value in col2:
    col2_prom_map = df_unique.groupby(col2)[col1].count()
    
    # Map the computed counts onto the original DataFrame.
    # Every row gets the same promiscuity value according to its col1 and col2 values.
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame.
    df[f'{col1}_promiscuity'] = df[col1].map(col1_prom_map)
    df[f'{col2}_promiscuity'] = df[col2].map(col2_prom_map)
    
    return df

####-
#--------------------------------------------------------------------------------------------------
# sampleinfo

def enclone_plot(samples_list, sampleinfo, min_cells=5, output_filename=None, barcode_filename=None,WORKING_DIR='/scratch/nbatada/ccr8/p01e04/data/P2008326_03232025/MULTI_GEX_VDJ'):
    '''
    CELLRANGER AGRR must be run for this to work
    
    Usage:
    from IPython.display import Image

    enclone_plot(['p01e04s02','p01e04s10'], sampleinfo)
    '''
    print(f'cd {WORKING_DIR}')
    s1=samples_list[0].upper()
    s2=samples_list[1].upper()
    
    s1_tissue=sampleinfo.set_index('sample_id').loc[s1]['tissue']
    s2_tissue=sampleinfo.set_index('sample_id').loc[s2]['tissue']

    s1_treatment=sampleinfo.set_index('sample_id').loc[s1]['treatment']
    s2_treatment=sampleinfo.set_index('sample_id').loc[s2]['treatment']

    s1_celltype=sampleinfo.set_index('sample_id').loc[s1]['celltype']
    s2_celltype=sampleinfo.set_index('sample_id').loc[s2]['celltype']
    
    if not output_filename:
        output_filename=f'{s1}_{s2}_enclone.png'
    
        
    samples_list_str=':'.join([f'{sample_id}_results_MULTI' for sample_id in [s1,s2]]) #just take the first 2
    #print(f'enclone PRE={WORKING_DIR} TCR={samples_list_str} MIN_CELLS={min_cells} PLOT="{output_filename},s1->blue,s2->red" NOPRINT LEGEND=blue,{s1},red,{s2}')
    s1_legend=f'{s1}_{s1_tissue}_{s1_treatment}_{s1_celltype}'
    s2_legend=f'{s2}_{s2_tissue}_{s2_treatment}_{s1_celltype}'
    # add BARCODE=foxp3_bcs.txt
    if barcode_filename:
        print(f'enclone PRE={WORKING_DIR} TCR={samples_list_str} MIN_CELLS={min_cells} BARCODE={barcode_filename} PLOT="{output_filename},s1->blue,s2->red" NOPRINT LEGEND=blue,{s1_legend},red,{s2_legend}')
    else:
        print(f'enclone PRE={WORKING_DIR} TCR={samples_list_str} MIN_CELLS={min_cells} PLOT="{output_filename},s1->blue,s2->red" NOPRINT LEGEND=blue,{s1_legend},red,{s2_legend}')
    
    print('# ---')
    print(f'image_filename="{WORKING_DIR}/{output_filename}"')
    print(f'Image(image_filename, width=500,height=500)')
    



#--------------------------------------------------------------------------------------------------

def shared_clonotype_expansion(df, sample_col, clone_col='airr:clone_id', title=''):
    '''
    Example usage:
    TREATMENTS=['iso+fty','accr8+fty']
    SORT="cd4"
    CELLTYPE_ID='tregs' # only for sor=="cd4"
    utils_scrnaseq.pl.shared_clonotype_expansion( mdata.obs.query(f'`gex:celltype_id`=="{CELLTYPE_ID}"').query(f'sort=="{SORT}"').query('treatment.isin(@TREATMENTS)'), sample_col='sample_id_augmented',title=f'{CELLTYPE_ID} ({" versus ".join(TREATMENTS)})')

    '''
    df_t = utils.tabulate(df, clone_col, sample_col)
    df_t = df_t[(df_t > 0).sum(axis=1) > 1]
    # check if it is non-zero
    if df_t.shape[0]==0:
        print('No shared clonotypes found')
        return 
    df_long = df_t.reset_index().rename(columns={'index': 'clonotype_id'})
    df_long = df_long.melt(id_vars='clonotype_id', var_name=sample_col, value_name='expansion')
    
    plot = (
        p9.ggplot(df_long, p9.aes(x='clonotype_id', y=sample_col, fill='clonotype_id', size='expansion'))
        + p9.geom_point(alpha=0.4)
        + p9.scale_size_continuous(range=(1, 10))
        # + utils.theme_nizar()
        + p9.ggtitle(title)
    )
    
    return plot


#----------------------------------------------------------------------------------------
def scirpy_plot_clonal_expansion(mdata, 
                          bins=None, 
                          bin_labels=None, 
                          facet_by='sort', 
                          custom_theme=None,
                          fields_for_sample_id_augmented=['sample_id', 'tissue', 'sort', 'treatment']):
    """
    Creates a stacked bar plot (using plotnine) of clonal expansion (proportion of clones per frequency bin)
    from single-cell gex/airr data stored in mdata.obs.
    
    The function assumes mdata.obs contains at least these columns:
       - "sample_id_augmented": a string that will be split into fields based on underscores.
       - "airr:clone_id", "airr:clone_id_size"
    
    Parameters:
      mdata: An AnnData/mdata object combining gex and airr data.
      bins: List of numeric bin edges for clone frequencies. Defaults to [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0].
      bin_labels: List of labels for the bins. Defaults to 
                  ["1e-5 to 1e-4", "1e-4 to 1e-3", "1e-3 to 1e-2", "1e-2 to 1e-1", "1e-1 to 1e0"].
      facet_by: Name of a column (extracted from sample_id_augmented) to use for faceting; default "sort".
      custom_theme: An optional plotnine theme (for example, utils.theme_nizar()) to add to the plot.
      fields_for_sample_id_augmented: List of expected field names for sample_id_augmented (if provided).
                                      For example: ['sample_id','tissue','sort','treatment'].
                                      If None, defaults to ['sample_id','tissue','sort','treatment'].
                                      
    Returns:
      A plotnine ggplot object.
    """
    # Set default bins/labels if not provided.
    if facet_by not in fields_for_sample_id_augmented:
        print('[ARGUMENT ERROR]: facet_by must be present in fields_for_sample_id_augmented')
        return None
    
    if bins is None:
        bins = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    if bin_labels is None:
        bin_labels = ["1e-5 to 1e-4", "1e-4 to 1e-3", "1e-3 to 1e-2", "1e-2 to 1e-1", "1e-1 to 1e0"]
    #if fields_for_sample_id_augmented is None:
    #    fields_for_sample_id_augmented = ['sample_id', 'tissue', 'sort', 'treatment']

        
    # --- Preprocess mdata ---
    # (These functions are assumed to update mdata in place and are specific to your workflow.)
    import scirpy as ir  # Ensure modules are available in PYTHONPATH
    import muon as mu

    # Already done in QC
    ir.pp.index_chains(mdata)
    ir.tl.chain_qc(mdata)
    mu.pp.filter_obs(mdata, "airr:chain_pairing", lambda x: ~np.isin(x, ["two full chains", "orphan VDJ", "orphan VJ"]))
    ir.pp.ir_dist(mdata)
    ir.tl.define_clonotypes(mdata, receptor_arms="all", dual_ir="primary_only")
    
    # --- Build working DataFrame from mdata.obs ---
    df = mdata.obs[['gex:sample_id_augmented', 'airr:clone_id', 'airr:clone_id_size']].copy()
    df = df.dropna(subset=["airr:clone_id_size"])
    df["airr:clone_id_size"] = df["airr:clone_id_size"].astype(float)
    
    # Normalize clone sizes by sample:
    # Each clone's frequency is its size divided by the sum of clone sizes in that sample.
    df["clone_frequency"] = df.groupby("gex:sample_id_augmented")["airr:clone_id_size"]\
                              .transform(lambda x: x / x.sum())
    
    # Bin the clone frequencies into discrete intervals.
    df["frequency_bin"] = pd.cut(df["clone_frequency"], bins=bins, labels=bin_labels, include_lowest=True)
    
    # --- Aggregate data per sample and compute proportions ---
    bin_counts = df.groupby(["gex:sample_id_augmented", "frequency_bin"]).size().unstack(fill_value=0)
    bin_proportions = bin_counts.div(bin_counts.sum(axis=1), axis=0)
    
    # Melt the proportions table into long format for plotnine.
    bin_proportions = bin_proportions.reset_index().melt(
        id_vars="gex:sample_id_augmented", 
        var_name="frequency_bin", 
        value_name="proportion"
    )
    
    # --- Extract additional fields from sample_id_augmented based on user provided list ---
    num_fields = len(fields_for_sample_id_augmented)
    split_fields = bin_proportions["gex:sample_id_augmented"].str.split('_', expand=True)
    if split_fields.shape[1] < num_fields:
        raise ValueError(f"Expected sample_id_augmented to have at least {num_fields} underscore-separated fields, "
                         f"but got {split_fields.shape[1]}.")

    for i, field in enumerate(fields_for_sample_id_augmented):
        bin_proportions[field] = split_fields[i]
    
    # Optionally create a treatment_group column if "treatment" is provided.
    if "treatment" in fields_for_sample_id_augmented:
        bin_proportions['treatment_group'] = bin_proportions['treatment'].str.split('+').str[0]
    
    # --- Order the sample_id_augmented ---
    # If the expected fields include "sort" and "treatment", use these for sorting.
    sample_ids = bin_proportions["gex:sample_id_augmented"].unique()
    if "sort" in fields_for_sample_id_augmented and "treatment" in fields_for_sample_id_augmented:
        sort_index = fields_for_sample_id_augmented.index("sort")
        treatment_index = fields_for_sample_id_augmented.index("treatment")
        ordered_samples = sorted(sample_ids, key=lambda s: (s.split('_')[sort_index], s.split('_')[treatment_index]))
    else:
        ordered_samples = sorted(sample_ids)

    bin_proportions["gex:sample_id_augmented"] = pd.Categorical(
        bin_proportions["gex:sample_id_augmented"], categories=ordered_samples, ordered=True
    )
    
    # --- Build the plot using plotnine ---
    p = (p9.ggplot(bin_proportions, 
                   p9.aes(x="gex:sample_id_augmented", y="proportion", fill="frequency_bin"))
         + p9.geom_bar(stat="identity")
         + p9.labs(x="Sample",
                   y="Proportion of Clones",
                   title="Distribution of Clonal Expansion",
                   fill="Frequency Bin")
         + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
    )
    
    if custom_theme is not None:
        p = p + custom_theme
    
    if facet_by:
        p = p + p9.facet_grid(f'. ~ {facet_by}', scales='free_x')
    
    return p


#-----------------------------------------------------------------------------------------------------------------------------------
import awkward as ak

def scirpy_clone_id_to_cdr3(mdata, target_clone_id):
    airr_data = mdata.mod['airr']
    all_clone_cells = mdata.obs[mdata.obs['airr:clone_id'].astype(str) == target_clone_id].copy()

    if all_clone_cells.empty:
        return pd.DataFrame()

    airr_clone_cells_index = all_clone_cells.index.intersection(airr_data.obs.index)

    if airr_clone_cells_index.empty:
        return pd.DataFrame()

    airr_clone_ilocs = airr_data.obs.index.get_indexer(airr_clone_cells_index)
    airr_details_array = airr_data.obsm['airr']
    clone_airr_array = airr_details_array[airr_clone_ilocs]

    cell_data_list = []

    for cell_name, cell_chains in zip(airr_clone_cells_index, clone_airr_array):
        cell_info = {
            'barcode': cell_name,
            'clone_id': target_clone_id,
            'TRA_V': "Not found", 'TRA_J': "Not found", 'TRA_C': "Not found", 'TRA_CDR3_AA': "Not found",
            'TRB_V': "Not found", 'TRB_D': "Not found", 'TRB_J': "Not found", 'TRB_C': "Not found", 'TRB_CDR3_AA': "Not found",
        }

        for chain in cell_chains:
            chain_fields = ak.fields(chain)
            is_tra = False
            is_trb = False

            if 'locus' in chain_fields:
                if isinstance(chain['locus'], str):
                    if chain['locus'] == 'TRA':
                        is_tra = True
                    elif chain['locus'] == 'TRB':
                        is_trb = True
            elif 'c_call' in chain_fields:
                 if isinstance(chain['c_call'], str):
                      if 'TRA' in chain['c_call']:
                           is_tra = True
                      elif 'TRB' in chain['c_call']:
                           is_trb = True

            if is_tra:
                if 'v_call' in chain_fields and chain['v_call'] is not None:
                    cell_info['TRA_V'] = chain['v_call']
                if 'j_call' in chain_fields and chain['j_call'] is not None:
                    cell_info['TRA_J'] = chain['j_call']
                if 'c_call' in chain_fields and chain['c_call'] is not None:
                     cell_info['TRA_C'] = chain['c_call']
                if 'cdr3_aa' in chain_fields and chain['cdr3_aa'] is not None:
                    cell_info['TRA_CDR3_AA'] = chain['cdr3_aa']

            elif is_trb:
                if 'v_call' in chain_fields and chain['v_call'] is not None:
                    cell_info['TRB_V'] = chain['v_call']
                if 'd_call' in chain_fields and chain['d_call'] is not None:
                    cell_info['TRB_D'] = chain['d_call']
                if 'j_call' in chain_fields and chain['j_call'] is not None:
                    cell_info['TRB_J'] = chain['j_call']
                if 'c_call' in chain_fields and chain['c_call'] is not None:
                     cell_info['TRB_C'] = chain['c_call']
                if 'cdr3_aa' in chain_fields and chain['cdr3_aa'] is not None:
                    cell_info['TRB_CDR3_AA'] = chain['cdr3_aa']

        cell_data_list.append(cell_info)

    vdj_df = pd.DataFrame(cell_data_list)
    return vdj_df

#---------------------------------------------------------------------------------------

import types
pp=types.SimpleNamespace(
    step1_load_tcr_annotation=step1_load_tcr_annotation,
    create_unique_clonotype_id_across_samples=create_unique_clonotype_id_across_samples,
)
tl=types.SimpleNamespace(
)

pl=types.SimpleNamespace(
    plot_venn_diagram=plot_venn_diagram,
)
