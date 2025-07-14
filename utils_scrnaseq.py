import pandas as pd
import numpy as np
import plotnine as p9
from collections import Counter
from importlib import reload
import matplotlib.pyplot as plt
import sys
import os
HOME=f'{os.path.expanduser("~")}'
sys.path.insert(0,f'{HOME}/lib')
import utils
import utils_scrnaseq
import plotnine as p9
import scanpy as sc
sc.settings.set_figure_params(dpi=50, facecolor="white")
import scanpy.external as sce
import re
import numpy as np
from scipy.stats import median_abs_deviation as mad

def cellranger_metrics_combine_inhouse(file_paths):
    all_file_data = []
    # Updated the first sample_id_pattern to include the '_X_' part
    sample_id_patterns = [r'SAM[0-9]+_[0-9]+_LIB[0-9]+', r'[pP][0-9][0-9][eE][0-9][0-9][sS][0-9][0-9]']
    
    sample_col_name_1 = 'sample_id_sam_lib'
    sample_col_name_2 = 'sample_id_pes'

    for file in file_paths:
        df_raw = pd.read_csv(file)

        found_id_1 = ''
        found_id_2 = ''

        # Extract sample IDs from the filename/path directly
        match_id_1 = re.search(sample_id_patterns[0], file)
        if match_id_1:
            found_id_1 = match_id_1.group(0)

        match_id_2 = re.search(sample_id_patterns[1], file)
        if match_id_2:
            found_id_2 = match_id_2.group(0)

        metrics_data_for_current_file = []
        for col_name_raw in df_raw.columns:
            metric_value_raw = df_raw[col_name_raw].iloc[0] 

            cleaned_metric_name = utils.clean_str_values([col_name_raw])[0]
            
            final_metric_name = 'gex.' + cleaned_metric_name

            cleaned_metric_value = str(metric_value_raw).replace('%', '').replace(',', '')
            try:
                cleaned_metric_value = float(cleaned_metric_value)
            except ValueError:
                cleaned_metric_value = None 

            metrics_data_for_current_file.append({
                'final_metric_name': final_metric_name,
                'metric_value': cleaned_metric_value
            })

        if metrics_data_for_current_file:
            metrics_df_processed = pd.DataFrame(metrics_data_for_current_file)
            metrics_series = metrics_df_processed.set_index('final_metric_name')['metric_value']
        else:
            metrics_series = pd.Series(dtype='float64')

        file_data_dict = metrics_series.to_dict()
        file_data_dict[sample_col_name_1] = found_id_1
        file_data_dict[sample_col_name_2] = found_id_2
        
        all_file_data.append(file_data_dict)

    combined_df = pd.DataFrame(all_file_data)
    
    id_cols = [sample_col_name_1, sample_col_name_2]
    existing_id_cols = [col for col in id_cols if col in combined_df.columns]
    other_cols = [col for col in combined_df.columns if col not in existing_id_cols]
    
    combined_df = combined_df[existing_id_cols + other_cols]

    return combined_df

def cellranger_metrics_combine_medgenome(qc_files, pattern=r'P[0-9][0-9]E[0-9][0-9]S[0-9][0-9]'):
    all_dfs = []
    
    for file in qc_files:
        # Check if the file exists.
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        
        # Extract sample ID from the file name using regex.
        #sample_ids = re.findall(pattern, file)
        #if not sample_ids:
        #    raise ValueError(f"Sample ID not found in filename: {file}")
        
        #sample_id = sample_ids[0]

        sample_id=utils.parse_sample_id(file)
        
        # Read CSV and keep only rows where Category != "Cells"
        df = pd.read_csv(file)
        
        ## df = df.query('Category != "Cells"') ######
        
        # Clean column names (requires utils.clean_str_values to be defined/imported)
        df.columns = utils.clean_str_values(df.columns)
        
        # Clean/modify library_type and metric_name columns.
        df['library_type'] = utils.clean_str_values(df.library_type) \
                                .str.replace('gene_expression', 'gex') \
                                .str.replace('vdj_t', 'vdj')
        df['metric_name'] = utils.clean_str_values(df.metric_name)
        
        # Clean metric_value: remove '%' and ',' then convert to float.
        df['metric_value'] = df.metric_value.str.replace('%', '') \
                                               .str.replace(',', '') \
                                               .astype('float')
        
        # Remove rows with metric_name starting with non-normalized metrics.
        ## df = df[~df.metric_name.str.startswith('number_of_reads')]
        ## df = df[~df.metric_name.str.startswith('mean_')]
        ## df = df[~df.metric_name.str.startswith('estimated_number')]
        
        # Concatenate library_type and metric_name.
        df['metric_name'] = df.library_type + '.' + df.metric_name
        
        # Retain only the metric name and its value, renaming the value column to sample_id.
        df = df[['metric_name', 'metric_value']]
        df.columns = ['metric_name', sample_id]
        
        # Use metric_name as index.
        df = df.set_index('metric_name') # key 
        all_dfs.append(df)
    
    # Concatenate all dataframes along columns.
    combined_df = pd.concat(all_dfs, axis=1).T
    combined_df.columns.name=''
    combined_df.index.name='sample_id'
    return combined_df


def marker_df_to_dict(df, key='group', value='names'): # here there are many values for each key 

    d=df.groupby(key)[value].apply(lambda x: list(x)).to_dict()
    return(d)

def filter_geneset(S, gene_set):
    # Keep only genes that are present in the adata (S)
    filtered_gene_set = {}
    removed_genes = []
    for key, genes in gene_set.items():
        filtered_genes = []
        for gene in genes:
            if gene in S.var_names:
                filtered_genes.append(gene)
            else:
                removed_genes.append(gene)
        filtered_gene_set[key] = filtered_genes
    if removed_genes:
        print(f"Removed genes: {', '.join(removed_genes)}")
    return filtered_gene_set



#--
def subsample_adata_by_obs_column(S, OBS_COLNAME, n_subsample): ## NOTE: making a copy()
  subsampled_adatas_list = []
  for c in S.obs[OBS_COLNAME].unique():
    adata_c = S[S.obs[OBS_COLNAME] == c]
    n_subsample_cluster = min(n_subsample, adata_c.n_obs)
    adata_c_subsample = sc.pp.subsample(adata_c, n_obs=n_subsample_cluster, copy=True, random_state=0)
    subsampled_adatas_list.append(adata_c_subsample)
  return sc.concat(subsampled_adatas_list)
#--
def find_markers(adata, LEIDEN, FIGSIZE=(12,16), KEY='markers', subsample=False, show_plot=False): # KEY is IGNORED
    PVAL_ADJ_THRESHOLD=0.01
    LOG2FC_MIN=0.5 # 2^(log2(LOG2FC_MIN))
    MIN_PCT_DIFF=0.3
 
    if subsample:
        SMALLEST_CLUSTER_SIZE=min(adata.obs[LEIDEN].value_counts())
        print(f'sample:True (now {SMALLEST_CLUSTER_SIZE} cells per group)')
        
        S=subsample_adata_by_obs_column (adata, LEIDEN, SMALLEST_CLUSTER_SIZE)
    else:
        S=adata

    sc.tl.rank_genes_groups( S, groupby=LEIDEN, groups='all', use_raw=False, layer='log1p', reference='rest', method='wilcoxon', pts=True) # key = rank_genes_groups #, key_added=KEY) #'markers_raw')
    
    # This adds key "rank_genes_groups_filtered" in uns
    # sc.tl.filter_rank_genes_groups(S, groupby=LEIDEN,  use_raw=False, min_in_group_fraction=0.25, min_fold_change=0.5) # key=KEY,

    #----------------------------------------------
    # Copy results back to the original adata.uns[KEY]
    # adata.uns[KEY] = S.uns[KEY] # this is unfiltered
    adata.uns['rank_genes_groups']=S.uns['rank_genes_groups']
    #----------------------------------------------

    # plot
    ## sc.pl.rank_genes_groups(adata, key=KEY, n_genes=10) # plot

    # results are in key=KEY ("marker")
    df = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=PVAL_ADJ_THRESHOLD, log2fc_min=LOG2FC_MIN) # key=KEY, use adata here as well

    df['diff_pct']=np.abs(df['pct_nz_group']-df['pct_nz_reference'])
    df=df.query(f'diff_pct > {MIN_PCT_DIFF}')
    if show_plot:
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, layer='scaled', groupby=LEIDEN, show_gene_labels=True, swap_axes=True, vmax=3, vmin=0, figsize=FIGSIZE) #, key=KEY, cmap='RdBu_r')
    #sc.pl.heatmap(adata, top_genes, groupby=LEIDEN, show_gene_labels=True,swap_axes=True,layer='scaled',vmax=3,vmin=1,figsize=FIGSIZE, cmap='RdBu_r') # added cmap


    return(df)



def plot_genes_separately(adata, marker_genes, PARAM):
    """
    Plot UMAP for each marker gene, grouped by a specified parameter in adata.obs.

    Parameters:
        adata (AnnData): Annotated data matrix.
        marker_genes (list): List of marker genes to plot.
        PARAM (str): Column in adata.obs to group the data (e.g., 'pool', 'treatment','leiden_1.0').
    """
    import matplotlib
    # Get unique groups in the column specified by PARAM
    groups = adata.obs[PARAM].unique()

    # Loop through each group and create UMAP plots for each marker gene
    for group in groups:
        # Subset data for the current group
        subset = adata[adata.obs[PARAM] == group]

        # Create a figure with multiple subplots (one for each marker gene)
        fig, axes = plt.subplots(1, len(marker_genes), figsize=(6 * len(marker_genes), 5))

        if len(marker_genes) == 1:  # If there's only one marker gene, make axes iterable
            axes = [axes]

        for ax, gene in zip(axes, marker_genes):
            # Plot UMAP for the subset and current marker gene
            sc.pl.umap(
                subset,
                color=gene,
                legend_loc='on data',
                use_raw=False,
                ax=ax,
                show=False,
                layer='log1p',
                color_map='binary', #matplotlib.cm.Reds,
            )
            ax.set_title(f'{PARAM.capitalize()}: {group} | Gene: {gene}')

        plt.tight_layout()
        plt.show()
      

import matplotlib.pyplot as plt
import numpy as np

def plot_gplot_gene_expr_umap1(adata, gene, obs_colname, atol=1e-5):
    """
    Plot UMAP for a given gene partitioned by a column in adata.obs.
    Cells with expression effectively zero (using np.isclose) are shown in gray,
    and cells with expression > 0 are shown in red.
    
    Parameters:
      adata       : AnnData object with UMAP coordinates in adata.obsm["X_umap"]
                    and gene expression data in adata[:, gene].X.
      gene        : str, gene symbol (must be present in adata.var_names).
      obs_colname : str, the name of the column in adata.obs to partition the data.
      atol        : float, atol parameter for np.isclose to decide if expression is 0.
    """
    # Check the gene
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names!")
    
    # Get expression values (flatten, convert if sparse)
    expr = adata[:, gene].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray().flatten()
    else:
        expr = np.array(expr).flatten()
    
    # Get UMAP coordinates
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP coordinates not found in adata.obsm['X_umap']!")
    umap = adata.obsm["X_umap"]
    
    # Partition the data based on obs_colname
    groups = adata.obs[obs_colname].unique()
    n_groups = len(groups)
    
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), squeeze=False)
    ax_list = axes[0]
    
    for ax, grp in zip(ax_list, groups):
        # Create a boolean mask for cells in the group
        mask = adata.obs[obs_colname].values == grp
        # Use np.isclose to test for effective zero value (use atol for tolerance)
        mask_zero = mask & np.isclose(expr, 0, atol=atol)
        mask_pos = mask & (~np.isclose(expr, 0, atol=atol))
        
        # Debug prints (optional):
        # print(f"Group {grp}: zero cells = {np.sum(mask_zero)}, positive cells = {np.sum(mask_pos)}")
        
        # Plot gray dots for cells with expression near zero
        ax.scatter(umap[mask_zero, 0], umap[mask_zero, 1], s=10, color="gray", label="expr == 0")
        # Plot red dots for cells with expression > 0
        ax.scatter(umap[mask_pos, 0], umap[mask_pos, 1], s=10, color="red", label="expr > 0")
        
        ax.set_title(f"{grp} (n={np.sum(mask)})")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def split_umap(adata, split_by, ncol=4, nrow=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    categories = adata.obs[split_by].cat.categories
    if nrow is None:
        nrow = int(np.ceil(len(categories) / ncol))
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    axs = axs.flatten()

    color_variables = kwargs.get('color')
    if color_variables is None:
        color_title_suffix = ""
    elif isinstance(color_variables, list):
        color_title_suffix = ", ".join(color_variables)
    else:
        color_title_suffix = color_variables

    for i, cat in enumerate(categories):
        ax = axs[i]
        
        title = f"{color_title_suffix} ({cat})" if color_title_suffix else str(cat)

        sc.pl.umap(adata[adata.obs[split_by] == cat], ax=ax, show=False, title=title, **kwargs)
    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
def plot_umap_split_by_variable(adata, leiden_col='leiden_0.1', var_name='donor', genes=None, layer_use='scaled'):
    """
    Generates UMAP plots split by a variable, with optional gene expression plots.
    Creates separate subplots for each variable value and gene combination.

    Args:
        adata: AnnData object.
        leiden_col (str, optional): Column in adata.obs to color clusters by. Defaults to 'leiden_0.1'.
        var_name (str, optional): Column in adata.obs to split plots by. Defaults to 'donor'.
        genes (list, optional): List of genes to overlay as expression plots. Defaults to None.
        layer_use (str, optional): Layer to use for gene expression. Defaults to 'scaled'.
    """

    if var_name not in adata.obs.columns:
        print(f"Warning: Variable column '{var_name}' not found in adata.obs. Please check column name.")
        return

    unique_vars = adata.obs[var_name].unique().tolist()
    num_vars = len(unique_vars)
    num_genes = len(genes) if genes else 0

    # Calculate total number of subplots needed
    total_subplots = num_vars * (1 + num_genes) # 1 for leiden, + num_genes for gene overlays

    fig, axs = plt.subplots(nrows=total_subplots // num_vars, ncols=num_vars,  # Adjust rows and cols
                             figsize=(5 * num_vars, 5 * (1 + num_genes) )) # Adjust figsize

    axs = axs.ravel() # Flatten axes array

    subplot_index = 0 # Track subplot index

    for var_val in unique_vars:
        adata_subset = adata[adata.obs[var_name].isin([var_val])] # Subset AnnData for current variable value

        # Plot Leiden clusters for the current variable value
        ax_leiden = axs[subplot_index]
        sc.pl.umap(
            adata_subset,
            color=leiden_col, # Color by Leiden clusters
            legend_loc='on data',
            title=f'{var_name}: {var_val}\nClusters ({leiden_col})', # Title with variable name and value
            size=2, cmap='magma', #viridis',
            show=False, alpha=1,
            ax=ax_leiden, # Use current subplot axes
        )
        subplot_index += 1 # Increment subplot index

        if genes: # Overlay gene expression if genes are provided
            for gene in genes:
                ax_gene = axs[subplot_index] # Get the next subplot axes
                if gene not in adata.var_names:
                    print(f"Warning: Gene '{gene}' not found in adata.var_names. Skipping gene overlay for {var_name} '{var_val}'.")
                    continue
                sc.pl.umap(
                    adata_subset,
                    color=gene, # Color by gene expression
                    ax=ax_gene, # Use the new axes for gene plot
                    show=False,
                    cmap='viridis',
                    size=2,
                    alpha=1,
                    colorbar_loc='right',
                    title=f'{var_name}: {var_val}\nGene: {gene}' # Title with gene name
                )
                subplot_index += 1 # Increment subplot index


    plt.tight_layout()
    plt.show()
    

#---

def step1_load_with_layers_no_concat(df_info, subtract_1_umi=False, MIN_CELLS=30, MIN_GENE_COUNTS_TOTAL=60, h5=True, colname_h5='data_path', colnames_meta=['donor', 'timepoint', 'treatment','tissue']): # subtract_1_umi because of ambient RNA
    qc_data = [['sample_id', 'numcells_raw', 'numcells_filtered', 'percent_filtered']]
    adata_dict = {} #[]
    dict_mean_expression = {}
    all_genes = set()
    union_highly_expressed = set()

    for sample_id in df_info['sample_id'].unique():
        sample_data = df_info[df_info['sample_id'] == sample_id]
        data_paths = sample_data[colname_h5].tolist()
        print(f'>> Processing {sample_id}') # ({data_paths})')
        if h5:
            libraries = [sc.read_10x_h5(x) for x in data_paths]
        else:
            libraries = [sc.read_10x_mtx(x, var_names='gene_symbols', cache=True) for x in data_paths]

        for lib_adata in libraries:
            lib_adata.var_names_make_unique()
        
            
        merged_sample = libraries[0].concatenate(*libraries[1:], batch_key="library")
        merged_sample.layers["counts"] = merged_sample.X.copy()

        # merged_sample.var_names_make_unique()
        all_genes.update(merged_sample.var_names)

        # Add sample-specific metadata to obs
        for col in colnames_meta: #['donor', 'timepoint', 'treatment','tissue']: #, 'pool']:
            merged_sample.obs[col] = sample_data[col].iloc[0]
        merged_sample.obs['sample_id'] = sample_id
        # merged_sample.obs['sample_id_augmented'] = f"{sample_id}_{sample_data['donor'].iloc[0]}_{sample_data['treatment'].iloc[0]}_{sample_data['timepoint'].iloc[0]}"
        joined_metadata_string = "_".join(  [str(sample_data[col_name].iloc[0]) for col_name in colnames_meta] )
        merged_sample.obs['sample_id_augmented'] = f"{sample_id}_{joined_metadata_string}"

    
        merged_sample.obs_names = [f"{barcode}_{sample_id}" for barcode in merged_sample.obs_names]
        n_raw = merged_sample.n_obs
        
        ### correct for ambient RNA
        if subtract_1_umi:
          merged_sample.X.data[merged_sample.X.data > 0] -= 1  # Subtract 1 only from non-zero values in .data, in place
          #merged_sample.X.data = np.clip(merged_sample.X.data, a_min=0, a_max=None) # Optional safety clipping

        #print("** [before filtering] num cells: ", merged_sample.shape[0])

        # Basic QC filtering
        sc.pp.filter_cells(merged_sample, min_genes=800)
        sc.pp.filter_genes(merged_sample, min_counts=MIN_GENE_COUNTS_TOTAL) #60)
        sc.pp.filter_genes(merged_sample, min_cells=MIN_CELLS) #30) # cannot nest it nor can use both parameters together

        #print("** [after filtering] num cells: ", merged_sample.shape[0])

        # Identify mitochondrial, ribosomal, and hemoglobin genes
        merged_sample.var['mt'] = merged_sample.var_names.str.startswith(('MT-','mt-'))
        merged_sample.var["ribo"] = merged_sample.var_names.str.startswith(("RPS", "RPL","Rps","Rpl"))
        merged_sample.var["hb"] = merged_sample.var_names.str.contains("^HB[^(P)]","^Hb[^(p)]")

        # Create layers and normalize
        #merged_sample.layers["counts"] = merged_sample.X.copy()
        merged_sample.layers["normalized"] = sc.pp.normalize_total(merged_sample, target_sum=1e4, inplace=False)['X']

        # Calculate mean expression per gene (for each sample)
        ## dict_mean_expression[sample_id] = pd.Series(np.ravel(merged_sample.layers['normalized'].mean(0)), index=merged_sample.var_names)
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(merged_sample, qc_vars=['mt','ribo','hb'], percent_top=[20], log1p=True, inplace=True)
        qc_data.append([sample_id, n_raw, merged_sample.n_obs, (1 - merged_sample.n_obs / n_raw) * 100])

        # Identify highly expressed genes (per sample)
        # sample_he_genes = get_highest_expressed_genes(merged_sample, n_top=50)
        # union_highly_expressed.update(sample_he_genes)

        adata_dict[sample_id]=merged_sample ## .append(merged_sample) # Append processed sample to list
    #
    for sample_id in adata_dict:
        print(sample_id, )
    df_qc = pd.DataFrame(qc_data[1:], columns=qc_data[0])
    print(df_qc)
    print('#-------------- Finished ---------------')
    
    return adata_dict #, df_qc, union_highly_expressed



#----
import matplotlib.pyplot as plt


def qc_plot_violin(adata, groupby='sample_id', keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt','pct_counts_in_top_20_genes'], LOG_SCALE=True):

    sc.set_figure_params(figsize=(8,5)) # Set figure size for plots

    if isinstance(adata, dict):
        print("Concatenating AnnData objects from dictionary for violin plot...")
        adata = sc.concat(
            list(adata.values()),
            join='outer',
            label="sample_id_concat",
            keys=list(adata.keys()),
            index_unique=None
        )

    ymin_per_key = {}
    ymax_per_key = {}

    for key in keys:
        ymin_values = []
        ymax_values = []
        if groupby in adata.obs.columns: # Check if groupby column exists in adata.obs
            if adata.obs[groupby].nunique() > 0: # Check if groupby column has unique values
                for sample_id in adata.obs[groupby].unique():
                    group_data = adata[adata.obs[groupby] == sample_id, :].obs[key]
                    ymin_values.append( np.percentile(group_data,  1)) # 1st percentile
                    ymax_values.append( np.percentile(group_data, 95)) # 99th percentile
                ymin_per_key[key] = min(ymin_values) if ymin_values else 0 # Handle empty list case
                ymax_per_key[key] = max(ymax_values) if ymax_values else 0 # Handle empty list case
            else: # Handle case where groupby column has no unique values
                print(f"Warning: Groupby column '{groupby}' has no unique values. Violin plot might not be informative.")
        else: # Handle case where groupby column is missing
            print(f"Warning: Groupby column '{groupby}' not found in adata.obs. Violin plot might not be grouped.")
    
    ax=sc.pl.violin(adata,
                    keys=keys,
                    jitter=0.2,
                    multi_panel=True,
                    rotation=25,
                    groupby=groupby,
                    log=LOG_SCALE,
                    size=0.2,
                    alpha=0.5,
                    show=False,
                    inner='box', cut=0, # show median and IQR
                    ) 

    if ax is not None: # Check if ax is not None before proceeding
        if 'n_genes_by_counts' in ymin_per_key and 'n_genes_by_counts' in ymax_per_key and ax[0] is not None:
            ax[0].set_ylim(bottom=ymin_per_key['n_genes_by_counts'], top=ymax_per_key['n_genes_by_counts'])
        if 'total_counts' in ymin_per_key and 'total_counts' in ymax_per_key and ax[1] is not None:
            ax[1].set_ylim(bottom=ymin_per_key['total_counts'], top=ymax_per_key['total_counts'])
        if 'pct_counts_mt' in ymin_per_key and 'pct_counts_mt' in ymax_per_key and ax[2] is not None:
            ax[2].set_ylim(bottom=ymin_per_key['pct_counts_mt'], top=ymax_per_key['pct_counts_mt'])

        for axis in ax: # loop through each axes if multi_panel=True
            if axis is not None: # Check if axis is not None before proceeding
                axis.tick_params(axis='x', labelrotation=10) # if you want to keep rotation
                axis.set_xticklabels(axis.get_xticklabels(), ha='right', fontsize=6) # adjust horizontal alignment
                axis.yaxis.set_major_locator(plt.MaxNLocator(nbins='auto'))
    return ax



#----

def step2_filter_data_using_qc_metrics(adata,
                                             n_genes_by_counts_cutoff_lower=None, n_genes_by_counts_cutoff_upper=None,
                                             total_counts_cutoff_lower=None, total_counts_cutoff_upper=None,
                                             pct_counts_mt_cutoff_lower=0, pct_counts_mt_cutoff_upper=10):

    adata_filtered = adata.copy() # Create a copy to avoid modifying original adata
    adata_filtered.obs['filter'] = '' # Initialize filter reason column
    filter_reasons = [] # List to store boolean Series for each filter reason

    if n_genes_by_counts_cutoff_lower is not None: # Check n_genes_by_counts lower cutoff
        filter_ngenes_lower = adata.obs['n_genes_by_counts'] < n_genes_by_counts_cutoff_lower # Identify cells BELOW the lower cutoff
        adata_filtered.obs.loc[filter_ngenes_lower, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';n_genes_by_counts_lower' # Add reason to 'filter' column
        filter_reasons.append(filter_ngenes_lower) # Append boolean series to list
    if n_genes_by_counts_cutoff_upper is not None: # Check n_genes_by_counts upper cutoff
        filter_ngenes_upper = adata.obs['n_genes_by_counts'] > n_genes_by_counts_cutoff_upper # Identify cells ABOVE the upper cutoff
        adata_filtered.obs.loc[filter_ngenes_upper, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';n_genes_by_counts_upper' # Add reason to 'filter' column
        filter_reasons.append(filter_ngenes_upper) # Append boolean series to list

    if total_counts_cutoff_lower is not None: 
        filter_tcounts_lower = adata.obs['total_counts'] < total_counts_cutoff_lower # Identify cells BELOW the lower cutoff
        adata_filtered.obs.loc[filter_tcounts_lower, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';total_counts_lower' # Add reason to 'filter' column
        filter_reasons.append(filter_tcounts_lower) # Append boolean series to list
    if total_counts_cutoff_upper is not None: # Check total_counts upper cutoff
        filter_tcounts_upper = adata.obs['total_counts'] > total_counts_cutoff_upper # Identify cells ABOVE the upper cutoff
        adata_filtered.obs.loc[filter_tcounts_upper, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';total_counts_upper' # Add reason to 'filter' column
        filter_reasons.append(filter_tcounts_upper) # Append boolean series to list

    if pct_counts_mt_cutoff_lower is not None:
        filter_pctmt_lower = adata.obs['pct_counts_mt'] < pct_counts_mt_cutoff_lower # Identify cells ABOVE the cutoff (mitochondrial percentage is usually upper bound)
        adata_filtered.obs.loc[filter_pctmt_lower, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';pct_counts_mt_lower' # Add reason to 'filter' column
        filter_reasons.append(filter_pctmt_lower) # Append boolean series to list

    if pct_counts_mt_cutoff_upper is not None:
        filter_pctmt_upper = adata.obs['pct_counts_mt'] > pct_counts_mt_cutoff_upper # Identify cells ABOVE the cutoff (mitochondrial percentage is usually upper bound)
        adata_filtered.obs.loc[filter_pctmt_upper, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';pct_counts_mt_upper' # Add reason to 'filter' column
        filter_reasons.append(filter_pctmt_upper) # Append boolean series to list
        
    #if pct_counts_mt_cutoff is not None: # Check pct_counts_mt cutoff
    #    filter_pctmt = adata.obs['pct_counts_mt'] > pct_counts_mt_cutoff # Identify cells ABOVE the cutoff (mitochondrial percentage is usually upper bound)
    #    adata_filtered.obs.loc[filter_pctmt, 'filter'] = adata_filtered.obs['filter'].astype(str) + ';pct_counts_mt' # Add reason to 'filter' column
    #    filter_reasons.append(filter_pctmt) # Append boolean series to list

    if filter_reasons: # If any filtering criteria were applied
        combined_filter = pd.concat(filter_reasons, axis=1).any(axis=1) # Combine filter conditions using OR logic
        adata_filtered.obs.loc[~combined_filter, 'filter'] = "PASS" # Set 'filter' to "PASS" for cells that pass filters
        adata_filtered.obs.loc[combined_filter, 'filter'] = adata_filtered.obs['filter'].str.strip(';') # Remove trailing semicolons if any
        adata_filtered.obs.loc[combined_filter, 'filter'] = adata_filtered.obs['filter'].str.replace(';', ';', regex=False) # clean up
        adata_filtered.obs.loc[combined_filter, 'filter'] = adata_filtered.obs['filter'].str.strip()

        n_filtered_cells_removed = combined_filter.sum() # Count removed cells
        percent_filtered_cells_removed = 100 * (n_filtered_cells_removed) / adata_filtered.n_obs # Calculate percentage removed

        adata_filtered = adata_filtered[~combined_filter, :].copy() # Subset AnnData object to keep only "PASS" cells

        print(f"Filtered out {percent_filtered_cells_removed:.2f}% of cells, {adata_filtered.n_obs} cells remaining.") # Print filtering summary after removal

    # adata_filtered.obs['filter'] = adata_filtered.obs['filter'].str.replace('PASS', '') # Clean up 'PASS' if no reason added
    # adata_filtered.obs['filter'] = adata_filtered.obs['filter'].str.strip(';') # Remove trailing semicolons if any
    adata_filtered.obs['filter'] = adata_filtered.obs['filter'].astype('category') # Convert 'filter' column to category
    return adata_filtered # Return the filtered AnnData object
  #----
#=== 


 
  

import scipy.sparse

def find_housekeeping_genes(S, n_top_genes=10):
    # 1. High Presence (Relaxed Threshold)
    print("\nStep 1: Calculating gene presence across cells (>= 90% presence)...")
    gene_presence_fraction = (S.X > 0).sum(axis=0) / S.n_obs
    gene_presence_fraction = np.array(gene_presence_fraction).flatten()
    presence_threshold = 0.90  # Relaxed presence threshold
    high_presence_genes_mask = gene_presence_fraction >= presence_threshold
    high_presence_genes = S.var_names[high_presence_genes_mask].tolist()
    print(f"  - Genes present in >= {presence_threshold*100}% of cells: {len(high_presence_genes)}")

    # 2. Low Inter-sample Variation (CV of log-transformed data - Relaxed Threshold)
    print("\nStep 2: Calculating inter-sample variation (CV of log-transformed data - relaxed threshold)...")
    if 'log1p' not in S.layers:
        raise ValueError("Layer 'log1p' not found in the Scanpy object. Please ensure log-normalization is applied and stored in S.layers['log1p'].")

    log1p_data = S.layers['log1p']
    sample_ids = S.obs['sample_id']
    sample_means_log = pd.DataFrame()

    for sample in sample_ids.unique():
        sample_mask = sample_ids == sample
        sample_data_log = log1p_data[sample_mask, :]
        sample_gene_means_log = np.mean(sample_data_log, axis=0)
        sample_means_log[sample] = sample_gene_means_log.A.flatten() if scipy.sparse.isspmatrix(sample_gene_means_log) else np.array(sample_gene_means_log).flatten()

    gene_means_across_samples_log = sample_means_log.mean(axis=1)
    gene_std_across_samples_log = sample_means_log.std(axis=1)
    gene_cv_log = gene_std_across_samples_log / gene_means_across_samples_log

    cv_threshold_relaxed = gene_cv_log.quantile(0.75) # Relaxed CV threshold (75th percentile - including more genes with moderate CV)
    low_cv_genes_mask_relaxed = gene_cv_log <= cv_threshold_relaxed
    low_cv_genes_relaxed = gene_cv_log[low_cv_genes_mask_relaxed].index.tolist()
    print(f"  - Genes with CV of log-transformed data in the top 75th percentile (CV <= {cv_threshold_relaxed:.3f}): {len(low_cv_genes_relaxed)}")


    # 3. High Basal Expression (Relaxed Threshold)
    print("\nStep 3: Calculating basal expression (mean expression of original counts - relaxed threshold)...")
    gene_mean_expression = np.mean(S.X, axis=0)
    gene_mean_expression = np.array(gene_mean_expression).flatten()

    expression_threshold_relaxed = np.quantile(gene_mean_expression, 0.25) # Relaxed expression threshold (25th percentile - including more genes with moderate expression)
    high_expression_genes_mask_relaxed = gene_mean_expression >= expression_threshold_relaxed
    high_expression_genes_relaxed = S.var_names[high_expression_genes_mask_relaxed].tolist()
    print(f"  - Genes with mean expression in the bottom 75th percentile (Mean Expr >= {expression_threshold_relaxed:.3f}): {len(high_expression_genes_relaxed)}")


    # 4. Rank within each criterion
    print("\nStep 4: Ranking genes within each criterion...")

    presence_rank = pd.Series(gene_presence_fraction, index=S.var_names).rank(ascending=False) # Higher presence, better rank (lower rank value)
    cv_rank = gene_cv_log.rank(ascending=True) # Lower CV, better rank
    expression_rank = pd.Series(gene_mean_expression, index=S.var_names).rank(ascending=False) # Higher expression, better rank

    # 5. Combine Ranks (Average Rank)
    print("\nStep 5: Combining ranks (averaging ranks)...")
    combined_rank = (presence_rank + cv_rank + expression_rank) / 3.0

    # 6. Select Top N
    print("\nStep 6: Selecting top genes based on combined rank...")
    ranked_genes = combined_rank.sort_values(ascending=True) # Lower combined rank is better
    top_housekeeping_genes = ranked_genes.head(n_top_genes).index.tolist()

    print(f"  - Top {n_top_genes} housekeeping genes selected based on combined rank.")
    print("\nHousekeeping Gene Identification Complete (Relaxed Criteria).")
    return top_housekeeping_genes
  

####=====


#---
def step3_integrate(adata, batch_key, n_pca=20, resolutions=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]):
    import scanpy as sc
    import scanpy.external as sce
    import sys

    sc.settings.verbosity = 3
    sc.settings.logfile = sys.stdout
    adata.X = adata.layers["log1p"]
    '''
    adata.obs["batch"] = (
        #adata.obs["pool"].astype(str) + "_" +
        adata.obs["donor"].astype(str) + "_" +
        #adata.obs["sample_id"].astype(str) + "_" +
        #adata.obs["timepoint"].astype(str) + "_" +
        adata.obs["treatment"].astype(str)
    )
    '''
    ##sc.pp.highly_variable_genes(adata, batch_key="batch")
    sc.pp.highly_variable_genes(adata, batch_key=batch_key)

    if "scaled" not in adata.layers:
        sc.pp.scale(adata)
        adata.layers["scaled"] = adata.X.copy()
    else:
        adata.X = adata.layers["scaled"]

    sc.tl.pca(adata, use_highly_variable=True, n_comps=n_pca, svd_solver='arpack')

    sce.pp.harmony_integrate(adata, key=batch_key, max_iter_harmony=50)

    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)

    for res in resolutions:
        print(f'Processing leiden with resolution {res}', flush=True)
        sc.tl.leiden(
            adata,
            resolution=res,
            key_added=f"leiden_{res}",
            flavor="igraph",
            n_iterations=20,
        )
    print('#--------------------------- Finished integration --------------')
    # doesn't return anything becuase it worksin place
        




    
def plot_tcell_markers(adata, res=1.0, genes=None):
    if genes is None:
        genes = ['CD4','CD8A','CXCR4','CXCR5','IL21','FOXP3','PRF1', 'GZMK','IFNG','NKG7','TCF7']

    leiden_col = f'leiden_{str(res)}'  # Ensure res is treated as a string

    if leiden_col not in adata.obs.columns:
        raise ValueError(f"Key '{leiden_col}' not found in adata.obs. Ensure leiden clustering was run with this resolution.")

    #sc.pl.heatmap(
    #    adata, var_names=genes, groupby=leiden_col, layer="log1p",
    #    figsize=(20, 3), show_gene_labels=True, swap_axes=True, dendrogram=True,
    #    show=True
    #)

    #sc.pl.dotplot(
    #    adata, var_names=genes, groupby=leiden_col, layer="log1p",
    #    figsize=(6, 8), dot_min=0.1, dot_max=1, standard_scale="var",
    #    title=leiden_col, dendrogram=True,
    #    show=True
    #)

    sc.pl.umap(
        adata, color=[leiden_col, 'donor', 'treatment', 'timepoint','CD4', 'CD8A', 'FOXP3','PRF1','PDCD1','CCR7','CD27','TCF7','IL2RA','GZMB'], use_raw=False, ncols=4
    )




import matplotlib.pyplot as plt

def sc_plot_combine(adata, plotting_commands, ncol=2, figure_size_ratio=(2, 1), figure_size_total_inch=(10, 4)):
    """
    Combines several Scanpy plotting commands into a single plot with subplots.

    Parameters:
        adata: AnnData object
        plotting_commands: list of strings, each a Scanpy plotting command
        ncol: int, number of columns in the grid
        figure_size_ratio: tuple, (width_ratio, height_ratio) for each subplot
        figure_size_total_inch: tuple, (total_width, total_height) of the figure in inches
    """
    # Calculate number of rows based on ncol
    n_plots = len(plotting_commands)
    nrow = (n_plots + ncol - 1) // ncol  # Ceiling division

    # Calculate individual subplot size
    total_width, total_height = figure_size_total_inch
    subplot_width = total_width / ncol
    subplot_height = total_height / nrow
    aspect_ratio = figure_size_ratio[0] / figure_size_ratio[1]
    subplot_height = subplot_width / aspect_ratio

    # Create the figure
    fig, axes = plt.subplots(nrow, ncol, figsize=(total_width, nrow * subplot_height))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Loop through commands and axes
    for i, command in enumerate(plotting_commands):
        ax = axes[i]
        # Ensure "ax=ax, show=False" is appended properly
        if "ax=" not in command and "show=" not in command:
            if command.endswith(")"):
                command = command[:-1] + ", ax=ax, show=False)"
            else:
                command += ", ax=ax, show=False"
        # Execute the command
        try:
            exec(command, {'adata': adata, 'sc': sc, 'plt': plt, 'ax': ax})
        except Exception as e:
            print(f"Error executing command: {command}\n{e}")
    
    # Remove empty subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


import scanpy as sc
import pandas as pd
import plotnine as p9

def sample_pca_plot(adata, fill="treatment", meta_colnames=['tissue', 'celltype', 'treatment']):
    """
    Compute a sample-level PCA using the top 500 highly variable genes and plot the samples
    on the first two principal components. The input AnnData object must have a column
    'sample_id_augmented' in its .obs, whose values are underscore-delimited strings.

    For example, if a sample_id_augmented string is:
         "P01E04S09_tumor_cd4_iso"
    then with meta_colnames = ['tissue','celltype','treatment'] the tokens are interpreted as:
         - Base sample id: "P01E04S09"
         - tissue       : "tumor"
         - celltype     : "cd4"
         - treatment    : "iso"

    After computing PCA on the top 500 HVGs, the function aggregates cells by sample_id_augmented,
    computes the median PC1/PC2 coordinates for each sample, and creates a plotnine plot.
    The points are colored by the specified fill column (from the extracted metadata), and the text labels
    (showing sample_id_augmented) are jittered to reduce overlap.
    
    Parameters:
      adata         : AnnData
          A Scanpy object with raw counts (or a "counts" layer) and a "sample_id_augmented" column in .obs.
      fill          : str (default "treatment")
          The name of the metadata column to use (from meta_colnames) for the fill color.
      meta_colnames : list of str (default ['tissue','celltype','treatment'])
          The metadata keys (after the base sample id) encoded in the sample_id_augmented strings.
    
    Returns:
      A plotnine ggplot object showing the sample PCA with jittered text labels.
    """
    # Make a copy so as not to modify the original AnnData
    adata = adata.copy()

    # --- Preprocessing: Normalization/Log Transformation ---
    if "log1p" not in adata.layers:
        if "counts" in adata.layers:
            sc.pp.normalize_total(adata, target_sum=1e4, layer="counts", inplace=True)
        else:
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()

    # --- HVG Selection ---
    adata_hvg = adata.copy()
    adata_hvg.X = adata_hvg.layers["log1p"]
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=500, flavor="cell_ranger", inplace=True)
    hvgs = adata_hvg.var_names[adata_hvg.var['highly_variable']].tolist()
    del adata_hvg

    # --- PCA Computation ---
    adata_pca = adata[:, hvgs].copy()
    adata_pca.X = adata_pca.layers["log1p"]
    sc.pp.pca(adata_pca, n_comps=10)
    
    # Extract PC1 and PC2 coordinates
    pca_coords = pd.DataFrame(
        adata_pca.obsm["X_pca"][:, :2],
        index=adata_pca.obs.index,
        columns=["PC1", "PC2"]
    )
    if "sample_id_augmented" not in adata_pca.obs.columns:
        raise ValueError("The AnnData object must have a 'sample_id_augmented' column in .obs")
    pca_coords["sample_id_augmented"] = adata_pca.obs["sample_id_augmented"]
    
    # Variance explained (for axis labels)
    var_ratio = adata_pca.uns["pca"]["variance_ratio"]
    pc1_var = var_ratio[0] * 100
    pc2_var = var_ratio[1] * 100

    # --- Aggregate to Sample Level ---
    sample_df = pca_coords.groupby("sample_id_augmented").agg({"PC1": "median", "PC2": "median"}).reset_index()

    # --- Parse sample_id_augmented into metadata ---
    tokens = sample_df["sample_id_augmented"].str.split("_", expand=True)
    required_tokens = 1 + len(meta_colnames)
    if tokens.shape[1] < required_tokens:
        raise ValueError(
            f"sample_id_augmented does not have enough tokens. Expected at least {required_tokens}, "
            f"but got {tokens.shape[1]}"
        )
    sample_df["sample_id"] = tokens[0]
    for i, meta in enumerate(meta_colnames):
        sample_df[meta] = tokens[i + 1]
        
    # --- Check fill column ---
    if fill not in sample_df.columns:
        raise ValueError(f"Fill column '{fill}' not found in the parsed metadata. "
                         f"Available columns: {list(sample_df.columns)}")
    
    # --- Build Plotnine Plot ---
    # Using geom_text with position_jitter to reduce overlapping labels.
    plot = (
        p9.ggplot(sample_df, p9.aes(**{'x': 'PC1', 'y': 'PC2', 'fill': fill}))
        + p9.geom_point(size=3, color="white", alpha=0.75)
        + p9.geom_text(p9.aes(label="sample_id_augmented"),size=8, angle=1, nudge_x=0.05, nudge_y=0.05, va="top", ha="left" )
        + p9.ggtitle("Sample PCA")
        + p9.xlab(f"PC1 ({pc1_var:.2f}% variance explained)")
        + p9.ylab(f"PC2 ({pc2_var:.2f}% variance explained)")
        + p9.theme(
            figure_size=(6, 6),
            legend_position="top",
            text=p9.element_text(size=10)
        )
    )
    
    return plot
    
def add_temp_variables_for_plotting(adata, obs_variables=["sample_id", "tissue", "treatment"]):
    new_vars = []
    
    for var_col_name in obs_variables:
        unique_vals = adata.obs[var_col_name].astype(str).unique()
        for var in unique_vals:
            new_col = f"{var_col_name}_{var}"
            adata.obs[new_col] = adata.obs[var_col_name].map({var: 1}, na_action="ignore").astype("category")
            new_vars.append(new_col)
    return new_vars

#temporary_obs_vars=add_variables_for_plotting(adata_post_integration, obs_variables=["sample_id", "donor", "tissue", "treatment"])

from scipy.sparse import issparse

def summarize_macrophage_monocyte_subsets(
    adata,
    leiden_columnname,
    species='human',
    expression_threshold=0.1,    # Min gene expression value to consider a cell "positive"
    low_pct_threshold=10,        # Max % of "negative" marker expression allowed
    
    # Specific marker thresholds for classification (default values are suggestions)
    cd14_high_pct=70,            # Min % CD14 for Classical Monocytes (human)
    cd16_high_pct=50,            # Min % CD16 for Non-classical Monocytes (human)
    ly6c_high_pct=70,            # Min % Ly6c for Classical Monocytes (mouse)
    cx3cr1_high_pct=50,          # Min % Cx3cr1 for Non-classical Monocytes (mouse)
    
    f480_high_pct=60,            # Min % F4/80 for Pan-Macrophages (mouse)
    cd64_high_pct=50,            # Min % CD64 for Macrophages/activated Monocytes (human/mouse)
    cd68_high_pct=40,            # Min % CD68 for general Macrophages (human/mouse, often intracellular)
    cd163_high_pct=40,           # Min % CD163 for M2-like Macrophages
    cd206_high_pct=40,           # Min % CD206 for M2-like Macrophages

    # Optional: broad lineage markers to exclude non-myeloid cells if run on a whole dataset
    check_lymphoid_neg=True, # Ensure myeloid cells are negative for lymphoid markers
    cd3_gene_check=None,     # CD3E/Cd3e
    cd19_gene_check=None,    # CD19/Cd19
    nk_gene_check=None       # NKG7/Klrb1c
):
    # Define gene names based on species
    if species.lower() == 'human':
        markers_to_use = {
            'cd14': "CD14",
            'cd16': "FCGR3A",    # CD16
            'cd64': "FCGR1A",    # CD64
            'cd68': "CD68",
            'cd163': "CD163",
            'cd206': "MRC1",     # CD206
        }
        if check_lymphoid_neg:
            markers_to_use['cd3_lymph'] = cd3_gene_check if cd3_gene_check else "CD3E"
            markers_to_use['cd19_lymph'] = cd19_gene_check if cd19_gene_check else "CD19"
            markers_to_use['nk_lymph'] = nk_gene_check if nk_gene_check else "NKG7"
            
    elif species.lower() == 'mouse':
        markers_to_use = {
            'cd14': "Cd14",
            'ly6c': "Ly6c1",     # Common Ly6c marker, consider Ly6c2 if needed
            'cx3cr1': "Cx3cr1",
            'f480': "Adgre1",    # F4/80
            'cd64': "Fcgr1",     # CD64
            'cd68': "Cd68",
            'cd163': "Cd163",
            'cd206': "Mrc1",     # CD206
        }
        if check_lymphoid_neg:
            markers_to_use['cd3_lymph'] = cd3_gene_check if cd3_gene_check else "Cd3e"
            markers_to_use['cd19_lymph'] = cd19_gene_check if cd19_gene_check else "Cd19"
            markers_to_use['nk_lymph'] = nk_gene_check if nk_gene_check else "Klrb1c"
    else:
        raise ValueError("Species must be either 'human' or 'mouse'")

    # Validate if all required genes are in adata.var_names
    for key, gene_name in markers_to_use.items():
        if gene_name not in adata.var_names:
            raise ValueError(f"Required gene '{gene_name}' (for {key}) not found in adata.var_names. Please check your gene list and adata.var_names.")

    # Extract expression values for all relevant genes
    gene_expressions = {}
    for gene_key, gene_name in markers_to_use.items():
        raw_expr = adata[:, gene_name].X
        if issparse(raw_expr):
            gene_expressions[gene_key] = raw_expr.toarray().ravel() 
        else:
            gene_expressions[gene_key] = raw_expr.ravel()

    # Store the exact temporary column names
    temp_obs_cols = []
    for gene_key in markers_to_use.keys():
        col_name = f'__{gene_key}_expr__'
        adata.obs[col_name] = pd.Series(gene_expressions[gene_key], index=adata.obs.index)
        temp_obs_cols.append(col_name)

    def compute_metrics(group):
        n_total = len(group)
        
        # Calculate percentages for all relevant markers
        # For simplicity, using .get() with 0 default if marker not in group (e.g., if check_lymphoid_neg is False)
        p_cd14  = 100 * (group.get("__cd14_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
        p_cd64  = 100 * (group.get("__cd64_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
        p_cd68  = 100 * (group.get("__cd68_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
        p_cd163 = 100 * (group.get("__cd163_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
        p_cd206 = 100 * (group.get("__cd206_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
        
        # Species-specific monocyte markers
        if species.lower() == 'human':
            p_cd16  = 100 * (group.get("__cd16_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
            p_ly6c = 0 # Not applicable for human
            p_cx3cr1 = 0 # Not applicable for human
            p_f480 = 0 # Not applicable for human
        elif species.lower() == 'mouse':
            p_cd16 = 0 # Not applicable for mouse (different roles)
            p_ly6c = 100 * (group.get("__ly6c_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
            p_cx3cr1 = 100 * (group.get("__cx3cr1_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
            p_f480 = 100 * (group.get("__f480_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
        
        # Lymphoid negative checks (if enabled)
        is_lymphoid_neg = True
        if check_lymphoid_neg:
            p_cd3_lymph = 100 * (group.get("__cd3_lymph_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
            p_cd19_lymph = 100 * (group.get("__cd19_lymph_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
            p_nk_lymph = 100 * (group.get("__nk_lymph_expr__", pd.Series([0])) > expression_threshold).sum() / n_total
            
            if p_cd3_lymph >= low_pct_threshold or p_cd19_lymph >= low_pct_threshold or p_nk_lymph >= low_pct_threshold:
                is_lymphoid_neg = False

        label = "unclassified_myeloid" # Default label for myeloid-like cells

        # --- Classification Logic ---
        if is_lymphoid_neg: # Only classify if lymphoid markers are low
            if species.lower() == 'human':
                # 1. Macrophages (prioritize M2-like, then general macrophage markers)
                if (p_cd163 >= cd163_high_pct) or (p_cd206 >= cd206_high_pct):
                    label = "mp_m2"
                elif (p_cd64 >= cd64_high_pct) or (p_cd68 >= cd68_high_pct): # General macrophage
                    label = "mp"
                # 2. Monocytes (Classical, Non-Classical, Intermediate)
                elif (p_cd14 >= cd14_high_pct) or (p_cd16 >= cd16_high_pct):
                    if p_cd14 >= cd14_high_pct and p_cd16 < low_pct_threshold:
                        label = "mono_cd16low"
                    elif p_cd14 < low_pct_threshold and p_cd16 >= cd16_high_pct:
                        label = "mono_cd16high"
                    elif p_cd14 >= cd14_high_pct and p_cd16 >= low_pct_threshold: 
                        label = "mono_int"
                    else:
                        label = "mono" # Catch-all for monocyte-like but not clearly defined
                else:
                    label = "unknown" # Cells that are myeloid but don't fit defined markers
            
            elif species.lower() == 'mouse':
                # 1. Macrophages (prioritize M2-like, then general macrophage markers like F4/80, CD64, CD68)
                if (p_cd163 >= cd163_high_pct) or (p_cd206 >= cd206_high_pct):
                    label = "mp_m2"
                elif (p_f480 >= f480_high_pct) or (p_cd64 >= cd64_high_pct) or (p_cd68 >= cd68_high_pct):
                    label = "mp"
                # 2. Monocytes (Classical, Non-Classical)
                elif (p_ly6c >= ly6c_high_pct) or (p_cx3cr1 >= cx3cr1_high_pct):
                    if p_ly6c >= ly6c_high_pct and p_cx3cr1 < low_pct_threshold:
                        label = "mono_cd16low"
                    elif p_ly6c < low_pct_threshold and p_cx3cr1 >= cx3cr1_high_pct:
                        label = "mono_cd16high"
                    else:
                        label = "mono"
                else:
                    label = "unknown"
        else: # If lymphoid markers were high
            label = "unknown" #"likely_lymphoid_contamination"


        # Return all calculated percentages
        metrics = {
            "cd14_pct": p_cd14,
            "cd64_pct": p_cd64,
            "cd68_pct": p_cd68,
            "cd163_pct": p_cd163,
            "cd206_pct": p_cd206,
            "n_total": n_total,
            "majority_vote": label
        }
        if species.lower() == 'human':
            metrics["cd16_pct"] = p_cd16
        elif species.lower() == 'mouse':
            metrics["ly6c_pct"] = p_ly6c
            metrics["cx3cr1_pct"] = p_cx3cr1
            metrics["f480_pct"] = p_f480

        if check_lymphoid_neg:
            metrics["cd3_lymph_pct"] = p_cd3_lymph
            metrics["cd19_lymph_pct"] = p_cd19_lymph
            metrics["nk_lymph_pct"] = p_nk_lymph
            
        return pd.Series(metrics)

    summary_table = adata.obs.groupby(leiden_columnname).apply(compute_metrics).reset_index()

    # Clean up: Remove all temporary columns
    for col_name in temp_obs_cols:
        if col_name in adata.obs.columns: 
            del adata.obs[col_name]

    #print("\n--- Summary Table of Cluster Percentages for Macrophages/Monocytes ---")
    #print(summary_table)

    mapping = {str(row[leiden_columnname]): row["majority_vote"]
               for _, row in summary_table.iterrows()}
    mapping_ = {key: value for key, value in mapping.items() if value != 'unknown'}
    
    print(mapping_)

    # Store original numeric cluster IDs for proper plotting order
    original_cluster_ids_numeric = summary_table[leiden_columnname].astype(int)

    # Create formatted labels for the x-axis
    summary_table['formatted_leiden_label'] = (
        summary_table[leiden_columnname].astype(str) + " (" + summary_table['n_total'].astype(str) + ")"
    )

    # Prepare plotting order based on sorted numeric IDs
    summary_table_sorted_for_order = summary_table.sort_values(by=leiden_columnname, key=lambda x: x.astype(int))
    plotting_order = summary_table_sorted_for_order['formatted_leiden_label'].tolist()

    # Adjust summary_table for melting
    summary_table = summary_table.drop([leiden_columnname, 'n_total'], axis=1)
    summary_table = summary_table.rename(columns={'formatted_leiden_label': leiden_columnname})

    # Melt the DataFrame for plotnine
    percentage_columns = [col for col in summary_table.columns if '_pct' in col]
    id_vars = ['majority_vote', leiden_columnname] 
    
    dfl = summary_table.melt(id_vars=id_vars, value_vars=percentage_columns)

    # Set leiden_columnname as a Categorical type with explicit order
    dfl[leiden_columnname] = pd.Categorical(dfl[leiden_columnname], categories=plotting_order, ordered=True)

    plot = p9.ggplot(dfl, p9.aes(x=leiden_columnname, y='value', fill='variable')) + \
           p9.geom_bar(position='dodge', stat='identity', width=0.8) + \
           p9.facet_grid('majority_vote~.', scales='free_x') + utils.theme_nizar() + p9.theme(figure_size=(10,6))
           
    return plot


def summarize_immune_subsets(
    adata,
    leiden_columnname,
    species='human',
    expression_threshold=0.1,    # Minimum gene expression value to consider a cell "positive"
    low_pct_threshold=10,        # Max % of "negative" marker expression allowed (e.g., CD8 in CD4 T cells)
    
    # T cell subset thresholds
    cd4_pos_pct=50,              # Min % of CD4 expression for CD4 T cells
    cd8_pos_pct=70,              # Min % of CD8 expression for CD8 T cells
    foxp3_pos_pct=50,            # Min % of Foxp3 expression for Tregs (within CD4+)
    
    # Broad cell type thresholds
    cd3_pos_pct=50,              # Min % of CD3 expression for Pan T cells
    cd14_pos_pct=50,             # Min % of CD14 expression for Macrophages/Monocytes
    cd19_pos_pct=50,             # Min % of CD19 expression for B cells
    trdc_pos_pct=50,             # Min % of TRDC expression for Gamma-delta T cells
    nk_pos_pct=50                # Min % of NK cell marker expression for NK cells
):
    # Define gene names based on species
    if species.lower() == 'human':
        cd4_gene = "CD4"
        cd8_gene = "CD8A"
        foxp3_gene = "FOXP3"
        cd3_gene = "CD3E"    
        cd14_gene = "CD14"   
        cd19_gene = "CD19"   
        trdc_gene = "TRDC"   
        nk_gene = "NKG7"     # Human NK marker
    elif species.lower() == 'mouse':
        cd4_gene = "Cd4"
        cd8_gene = "Cd8a"
        foxp3_gene = "Foxp3"
        cd3_gene = "Cd3e"    
        cd14_gene = "Cd14"   
        cd19_gene = "Cd19"   
        trdc_gene = "Trdc"   
        nk_gene = "Klrb1c"   # Mouse NK marker
    else:
        raise ValueError("Species must be either 'human' or 'mouse'")

    # List all genes to check and extract, using simplified keys for consistent temp column naming
    all_genes_markers = { 
        'cd4': cd4_gene,
        'cd8': cd8_gene,
        'foxp3': foxp3_gene,
        'cd3': cd3_gene,
        'cd14': cd14_gene,
        'cd19': cd19_gene,
        'trdc': trdc_gene,
        'nk': nk_gene 
    }

    for gene_key, gene_name in all_genes_markers.items():
        if gene_name not in adata.var_names:
            raise ValueError(f"Gene '{gene_name}' (for {gene_key}) not found in adata.var_names. Please check your gene list and adata.var_names.")

    # Extract expression values for all relevant genes
    gene_expressions = {}
    for gene_key, gene_name in all_genes_markers.items():
        raw_expr = adata[:, gene_name].X
        if issparse(raw_expr):
            gene_expressions[gene_key] = raw_expr.toarray().ravel() 
        else:
            gene_expressions[gene_key] = raw_expr.ravel()

    # Store the exact temporary column names
    temp_obs_cols = []
    for gene_key in all_genes_markers.keys():
        col_name = f'__{gene_key}_expr__'
        adata.obs[col_name] = pd.Series(gene_expressions[gene_key], index=adata.obs.index)
        temp_obs_cols.append(col_name)

    def compute_metrics(group):
        n_total = len(group)
        
        # Calculate percentages for all relevant markers using the consistent temp column names
        p_cd4   = 100 * (group["__cd4_expr__"] > expression_threshold).sum() / n_total
        p_cd8   = 100 * (group["__cd8_expr__"] > expression_threshold).sum() / n_total
        p_foxp3 = 100 * (group["__foxp3_expr__"] > expression_threshold).sum() / n_total
        p_cd3   = 100 * (group["__cd3_expr__"] > expression_threshold).sum() / n_total
        p_cd14  = 100 * (group["__cd14_expr__"] > expression_threshold).sum() / n_total
        p_cd19  = 100 * (group["__cd19_expr__"] > expression_threshold).sum() / n_total
        p_trdc  = 100 * (group["__trdc_expr__"] > expression_threshold).sum() / n_total
        p_nk    = 100 * (group["__nk_expr__"] > expression_threshold).sum() / n_total 
        
        # Double positive check (can be expanded if needed for other definitions)
        p_dp    = 100 * ((group["__cd4_expr__"] > expression_threshold) & (group["__cd8_expr__"] > expression_threshold)).sum() / n_total

        label = "unknown" 

        # Hierarchical classification logic (most specific to least specific)
        # 1. Gamma-delta T cells (TRDC positive, CD3 positive, and low other T cell markers)
        if p_trdc >= trdc_pos_pct and p_cd3 >= cd3_pos_pct and p_cd4 < low_pct_threshold and p_cd8 < low_pct_threshold:
            label = "t_gd"
        # 2. Pan T cells (CD3 positive, and not classified as gamma-delta T already)
        elif p_cd3 >= cd3_pos_pct:
            if p_cd4 >= cd4_pos_pct and p_cd8 < low_pct_threshold and p_foxp3 >= foxp3_pos_pct:
                label = "tregs"
            elif p_cd8 >= cd8_pos_pct and p_cd4 < low_pct_threshold:
                label = "t_cd8"
            elif p_cd4 >= cd4_pos_pct and p_cd8 < low_pct_threshold:
                label = "t_cd4"
            else: # CD3 positive but doesn't fit specific T cell criteria
                label = "t_other"
        # 3. NK cells (NK marker positive, and low CD3, CD14, CD19 to exclude other major types)
        elif p_nk >= nk_pos_pct and p_cd3 < low_pct_threshold and p_cd14 < low_pct_threshold and p_cd19 < low_pct_threshold:
            label = "nk"
        # 4. Macrophages/Monocytes (CD14 positive and CD3/CD19/NK negative)
        elif p_cd14 >= cd14_pos_pct and p_cd3 < low_pct_threshold and p_cd19 < low_pct_threshold and p_nk < low_pct_threshold:
            label = "mp"
        # 5. B cells (CD19 positive and CD3/CD14/NK negative)
        elif p_cd19 >= cd19_pos_pct and p_cd3 < low_pct_threshold and p_cd14 < low_pct_threshold and p_nk < low_pct_threshold:
            label = "b"
        # 6. Remaining cells are unknown

        return pd.Series({
            "cd4_pct": p_cd4,
            "cd8_pct": p_cd8,
            "foxp3_pct": p_foxp3,
            "cd3_pct": p_cd3,     
            "cd14_pct": p_cd14,
            "cd19_pct": p_cd19,
            "trdc_pct": p_trdc,
            "nk_pct": p_nk,       
            "dp_pct": p_dp,
            "n_total": n_total,
            "majority_vote": label
        })

    summary_table = adata.obs.groupby(leiden_columnname).apply(compute_metrics).reset_index()

    # Clean up: Remove all temporary columns using the list of actual column names
    for col_name in temp_obs_cols:
        if col_name in adata.obs.columns: 
            del adata.obs[col_name]

    #print(summary_table)

    mapping = {str(row[leiden_columnname]): row["majority_vote"]
               for _, row in summary_table.iterrows()}
    mapping_ = {key: value for key, value in mapping.items() if value != 'unknown'}

    print(mapping_)

    # Store the original numeric cluster IDs before formatting them
    original_cluster_ids_numeric = summary_table[leiden_columnname].astype(int)

    # Create the formatted strings for the x-axis labels (e.g., "0 (123)")
    summary_table['formatted_leiden_label'] = (
        summary_table[leiden_columnname].astype(str) + " (" + summary_table['n_total'].astype(str) + ")"
    )

    # Generate the desired plotting order based on sorted numeric IDs
    summary_table_sorted_for_order = summary_table.sort_values(by=leiden_columnname, key=lambda x: x.astype(int))
    
    # Extract the formatted labels in the correct numerical order
    plotting_order = summary_table_sorted_for_order['formatted_leiden_label'].tolist()

    # Drop the original leiden_columnname and n_total, and rename the new formatted column
    summary_table = summary_table.drop([leiden_columnname, 'n_total'], axis=1)
    summary_table = summary_table.rename(columns={'formatted_leiden_label': leiden_columnname})

    # Melt the DataFrame for plotnine, ensuring all percentage columns are included
    percentage_columns = [col for col in summary_table.columns if '_pct' in col]
    id_vars = ['majority_vote', leiden_columnname] 
    
    dfl = summary_table.melt(id_vars=id_vars, value_vars=percentage_columns)

    # Convert the leiden_columnname in dfl to a Categorical type
    # and explicitly set the categories to enforce numerical order on the x-axis.
    dfl[leiden_columnname] = pd.Categorical(dfl[leiden_columnname], categories=plotting_order, ordered=True)

    plot = p9.ggplot(dfl, p9.aes(x=leiden_columnname, y='value', fill='variable')) + \
           p9.geom_bar(position='dodge', stat='identity', width=0.8) + \
           p9.facet_grid('majority_vote~.', scales='free_x') + utils.theme_nizar() + p9.theme(figure_size=(10,6))
           
    return plot


####===
def infer_t_celltypes(
    adata,
    leiden_columnname,
    species              = 'human',
    expression_threshold = 0.3,
    low_pct_threshold    = 10,
    cd4_pos_pct          = 40,
    cd8_pos_pct          = 50,
    foxp3_pos_pct        = 50,
    gamma_delta_pos_pct  = 30,
    cyt_score_pos_pct    = 30,
    cd3e_pos_pct         = 30,
    cd3e_expr_threshold  = 0.1,
    print_table          = False
):
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse
    import plotnine as p9
    # Assume utils has been imported elsewhere
    
    # Define species-specific gene names
    if species.lower() == 'human':
        cd3e_gene          = "CD3E"
        cd4_gene           = "CD4"
        cd8_gene           = "CD8A"
        foxp3_gene         = "FOXP3"
        gamma_delta_gene   = "TRDC"
        sell_gene          = "SELL"
        ifng_gene          = "IFNG"
        cytolytic_genes    = ["PRF1", "GZMA", "IFNG"]
    else:
        cd3e_gene        = "Cd3e"
        cd4_gene         = "Cd4"
        cd8_gene         = "Cd8a"
        foxp3_gene       = "Foxp3"
        gamma_delta_gene = "Trdc"
        sell_gene        = "Sell"
        ifng_gene        = "Ifng"
        cytolytic_genes  = ["Prf1", "Gzma", "Ifng"]
    
    def get_expr(adata, gene):
        expr_raw = adata[:, gene].X
        return expr_raw.toarray().ravel() if issparse(expr_raw) else expr_raw.ravel()
    
    # Get gene expression for all markers
    cd3e_expr  = get_expr(adata, cd3e_gene)
    cd4_expr   = get_expr(adata, cd4_gene)
    cd8_expr   = get_expr(adata, cd8_gene)
    foxp3_expr = get_expr(adata, foxp3_gene)
    trdc_expr  = get_expr(adata, gamma_delta_gene)
    sell_expr  = get_expr(adata, sell_gene)
    # Get a separate measurement of IFNG (in addition to its inclusion in the cyt score)
    ifng_expr_separate = get_expr(adata, ifng_gene)
    
    # Compute a cytolytic score as the mean expression of the cytolytic genes list
    cyt_expr_list = [get_expr(adata, gene) for gene in cytolytic_genes]
    cyt_expr      = np.vstack(cyt_expr_list).mean(axis=0)
    
    # Store temporary expression values in adata.obs for grouping/counting
    adata.obs['__cd3e_expr__']  = pd.Series(cd3e_expr, index=adata.obs.index)
    adata.obs['__cd4_expr__']   = pd.Series(cd4_expr, index=adata.obs.index)
    adata.obs['__cd8_expr__']   = pd.Series(cd8_expr, index=adata.obs.index)
    adata.obs['__foxp3_expr__'] = pd.Series(foxp3_expr, index=adata.obs.index)
    adata.obs['__trdc_expr__']  = pd.Series(trdc_expr, index=adata.obs.index)
    adata.obs['__cyt_score__']  = pd.Series(cyt_expr, index=adata.obs.index)
    adata.obs['__sell_expr__']  = pd.Series(sell_expr, index=adata.obs.index)
    adata.obs['__ifng_expr__']  = pd.Series(ifng_expr_separate, index=adata.obs.index)
    
    def compute_metrics(group):
        n_total = len(group)
        p_cd3e  = 100 * (group["__cd3e_expr__"]  > cd3e_expr_threshold).sum() / n_total
        p_cd4   = 100 * (group["__cd4_expr__"]   > expression_threshold).sum() / n_total
        p_cd8   = 100 * (group["__cd8_expr__"]   > expression_threshold).sum() / n_total
        p_foxp3 = 100 * (group["__foxp3_expr__"] > expression_threshold).sum() / n_total
        p_trdc  = 100 * (group["__trdc_expr__"]  > expression_threshold).sum() / n_total
        p_cyt   = 100 * (group["__cyt_score__"]  > expression_threshold).sum() / n_total
        #p_dp    = 100 * ((group["__cd4_expr__"] > expression_threshold) & (group["__cd8_expr__"] > expression_threshold)).sum() / n_total
        
        # New metrics for SELL and IFNG
        p_sell  = 100 * (group["__sell_expr__"] > expression_threshold).sum() / n_total
        p_ifng  = 100 * (group["__ifng_expr__"] > expression_threshold).sum() / n_total
        
        # Decide label based on the metrics
        if p_cd3e < cd3e_pos_pct:
            label = "x"
        elif p_trdc >= gamma_delta_pos_pct:
            label = "gd"
        elif p_foxp3 >= foxp3_pos_pct and p_cd4 >= cd4_pos_pct: #and p_cd8 < low_pct_threshold:
            label = "cd4_tregs"
        elif (p_cd8 >= cd8_pos_pct): # or p_cyt >= cyt_score_pos_pct):
            # For CD8 clusters, subdivide into memory (mem), cytotoxic (ctl) or default label
            if p_cd8 >= cd8_pos_pct and p_ifng < low_pct_threshold and p_sell > low_pct_threshold:
                label = "cd8_mem"
            elif p_cd8 >= cd8_pos_pct and p_sell < low_pct_threshold and p_ifng > low_pct_threshold:
                label = "cd8_ctl"
            else:
                label = "cd8"
        elif p_cd4 >= cd4_pos_pct:
            # For CD4 clusters (that are not tregs)
            if p_cd4 >= cd4_pos_pct and p_ifng < low_pct_threshold and p_sell > low_pct_threshold:
                label = "cd4_mem"
            elif p_cd4 >= cd4_pos_pct and p_sell < low_pct_threshold and p_ifng > low_pct_threshold:
                label = "cd4_th1"
            else:
                label = "cd4_tconv"
        else:
            label = "x"
            
        return pd.Series({
            "cd3e_pct": p_cd3e,
            "cd4_pct": p_cd4,
            "cd8_pct": p_cd8,
            "foxp3_pct": p_foxp3,
            "trdc_pct": p_trdc,
            "cyt_pct": p_cyt,
            #"dp_pct": p_dp,
            "sell_pct": p_sell,
            "ifng_pct": p_ifng,
            "n_total": n_total,
            "majority_vote": label
        })
    
    summary_table = adata.obs.groupby(leiden_columnname).apply(compute_metrics).reset_index()
    
    # Remove temporary expression columns
    for col in ['__cd3e_expr__', '__cd4_expr__', '__cd8_expr__', '__foxp3_expr__',
                '__trdc_expr__', '__cyt_score__', '__sell_expr__', '__ifng_expr__']:
        adata.obs.drop(col, axis=1, inplace=True)
    
    mapping = {str(row[leiden_columnname]): row["majority_vote"] for _, row in summary_table.iterrows()}
    print('cluster_mapping=',mapping)
    
    summary_table['cluster_id']    = summary_table[leiden_columnname].astype(str)
    summary_table['cluster_label'] = summary_table['cluster_id'] + "\n(" + summary_table['n_total'].astype(str) + ")"
    sorted_list = utils.sort_natural(list(adata.obs[leiden_columnname].unique()))
    categories_order = [
        s + "\n(" + str(summary_table.loc[summary_table['cluster_id'] == s, 'n_total'].iloc[0]) + ")" 
        for s in sorted_list
    ]
    summary_table[leiden_columnname] = pd.Categorical(summary_table['cluster_label'], categories=categories_order, ordered=True)
    summary_table = summary_table.drop(['n_total', 'cluster_id', 'cluster_label'], axis=1)
    
    # Update possible labels (categories) to reflect the new nomenclature
    maj_vote_categories = ['x', 'gd', 'cd4_tregs', 'cd4_mem', 'cd4_th1', 'cd4_tconv', 'cd8_ctl', 'cd8_mem', 'cd8']
    summary_table['majority_vote'] = pd.Categorical(summary_table['majority_vote'], categories=maj_vote_categories, ordered=True)
    
    if print_table:
        print(summary_table)
    # (Needed for compatibility with older versions of pandas/plotnine)
    pd.DataFrame.iteritems = pd.DataFrame.items
    
    plot = (
        p9.ggplot(summary_table.melt(id_vars=['majority_vote', leiden_columnname]),
                  p9.aes(x=leiden_columnname, y='value', fill='variable')) +
        p9.geom_bar(position='dodge', stat='identity', width=0.8) +
        p9.facet_grid(f'majority_vote ~ {leiden_columnname}', scales='free_x', drop=False) +
        utils.theme_nizar() +
        p9.theme(figure_size=(12, 8), axis_text_x=p9.element_blank())
    )
    return plot

####====
##--
##--

#-------------------------------------------------------------
#
def score_clusters_by_gene_sets(adata,
                                celltype_geneset,
                                groupby_key='leiden_0.3',
                                title='',
                                min_cells_expressed_pct=0.005,
                                y_axis_order=None,
                                split_by=None,
                                split_by_order=None,
                                one_plot_per_geneset=False,
                                rescale_scores=False,
                                rescale_method="minmax",
                                weight_method="equal",
                                filter_low_var_genes=False,
                                min_cluster_range=0.1,
                                use_faceting=True,
                                facet_wrap_ncol=2,
                                color_axis_limits=(-2, 2)):
    if y_axis_order is not None:
        final_categories = y_axis_order
    else:
        final_categories = adata.obs[groupby_key].unique().tolist()

    adata.obs[groupby_key] = pd.Categorical(adata.obs[groupby_key], categories=final_categories, ordered=True)

    def compute_avg_scores(adata_sub, genes):
        original_gene_count = len(genes)
        genes_present = [gene for gene in genes if gene in adata_sub.var_names]
        filtered_gene_count = len(genes_present)

        if filtered_gene_count < original_gene_count:
            dropped_genes = set(genes) - set(genes_present)
            warnings.warn("Gene filtering: {} genes provided, but only {} remain after filtering. Dropped genes: {}".format(
                original_gene_count, filtered_gene_count, ", ".join(dropped_genes)
            ))

        if filtered_gene_count == 0:
            cell_scores = np.full(adata_sub.n_obs, np.nan)
            score_series = pd.Series(cell_scores, index=adata_sub.obs.index)
            avg_scores = score_series.groupby(adata_sub.obs[groupby_key]).mean().reindex(final_categories)
            return avg_scores

        bin_expr = adata_sub[:, genes_present].X
        if hasattr(bin_expr, "toarray"):
            bin_expr = bin_expr.toarray()
        if bin_expr.ndim == 1:
            bin_expr = bin_expr[:, None]
        if weight_method == "variance":
            gene_vars = bin_expr.var(axis=0)
            weights = gene_vars + 1e-6
        else:
            weights = np.ones(bin_expr.shape[1])

        cell_scores = (bin_expr * weights).sum(axis=1) / weights.sum()
        score_series = pd.Series(cell_scores, index=adata_sub.obs.index)
        avg_scores = score_series.groupby(adata_sub.obs[groupby_key]).mean().reindex(final_categories)
        return avg_scores

    records = []
    if split_by is not None:
        split_groups = adata.obs[split_by].unique().tolist()
    else:
        split_groups = [None]

    for gs_name, genes in celltype_geneset.items():
        for sp in split_groups:
            if sp is not None:
                adata_sub = adata[adata.obs[split_by] == sp]
            else:
                adata_sub = adata

            if adata_sub.n_obs < min_cells_expressed_pct * adata.n_obs:
                continue

            avg_scores = compute_avg_scores(adata_sub, genes)
            for cat in final_categories:
                records.append({
                    "GeneSet": gs_name,
                    groupby_key: cat,
                    "Score": avg_scores.get(cat, np.nan),
                    split_by: sp
                })

    if len(records) == 0:
        warnings.warn("No records to plot. Check your filters and gene sets.")
        return None

    df = pd.DataFrame(records)

    if split_by is not None and split_by_order is not None:
        df[split_by] = pd.Categorical(df[split_by], categories=split_by_order, ordered=True)

    if rescale_scores:
        if split_by is not None:
            group_cols = ["GeneSet", split_by]
        else:
            group_cols = ["GeneSet"]
        if rescale_method == "minmax":
            def minmax_scale(x):
                rng = x.max() - x.min()
                if rng == 0:
                    return x * 0
                return 2 * ((x - x.min()) / rng) - 1
            df["Score"] = df.groupby(group_cols)["Score"].transform(minmax_scale)
        elif rescale_method == "zscore":
            def zscore_scale(x):
                std = x.std(ddof=0)
                if std == 0:
                    return x * 0
                return (x - x.mean()) / std
            df["Score"] = df.groupby(group_cols)["Score"].transform(zscore_scale)
        else:
            raise ValueError("Unknown rescale_method: {}. Use 'minmax' or 'zscore'.".format(rescale_method))

    df["Score_text"] = df["Score"].round(2).astype(str)
    lower_limit, upper_limit = color_axis_limits
    df["Score_fill"] = df["Score"].apply(lambda x: max(min(x, upper_limit), lower_limit))

    p = (p9.ggplot(df, p9.aes(x="GeneSet", y=groupby_key, fill="Score_fill"))
         + p9.geom_tile()
         + p9.geom_text(p9.aes(label="Score_text"), size=8, color="black")
         + p9.scale_fill_gradient2(low="red", mid="white", high="blue", midpoint=0, limits=color_axis_limits)
         + p9.labs(title=title, x="", y="")
         + p9.theme_bw()
         + p9.scale_y_discrete(limits=final_categories))

    if use_faceting and split_by is not None:
        p = p + p9.facet_wrap("~" + split_by, ncol=facet_wrap_ncol)

    if one_plot_per_geneset:
        plots = []
        for gs in df["GeneSet"].unique():
            subdf = df[df["GeneSet"] == gs].copy()
            subdf["Score_text"] = subdf["Score"].round(2).astype(str)
            subdf["Score_fill"] = subdf["Score"].apply(lambda x: max(min(x, upper_limit), lower_limit))
            plot = (p9.ggplot(subdf, p9.aes(x="GeneSet", y=groupby_key, fill="Score_fill"))
                    + p9.geom_tile()
                    + p9.geom_text(p9.aes(label="Score_text"), size=8, color="black")
                    + p9.scale_fill_gradient2(low="red", mid="white", high="blue", midpoint=0, limits=color_axis_limits)
                    + p9.labs(title="Geneset: {}".format(gs), x="", y="")
                    + p9.theme_bw()
                    + p9.scale_y_discrete(limits=final_categories))
            if use_faceting and split_by is not None:
                plot = plot + p9.facet_wrap("~" + split_by, ncol=facet_wrap_ncol)
            plots.append(plot)
        return plots
    else:
        return p
    
#-------------------------------------------------------------

import matplotlib.pyplot as plt

def deg_by_condition(
    adata,
    condition_column,
    condition_treatment,
    condition_reference,
    cluster_column,
    cluster_name,
    n_cells_per_group=500,
    pval_cutoff=0.05,
    log2fc_min=1.0,
    min_pct_nz_in_group=0.3,
    n_top_genes_plot=10,
    plot_layer='log1p',
    method='wilcoxon',
    random_state=42,
    do_subsample=False
):
    def subsample_balanced(adata, groupby_col, group1, group2, n_cells=100, rs=None):
        sampled_indices = []
        rng = np.random.default_rng(rs)
        group1_indices = adata.obs.index[adata.obs[groupby_col] == group1].tolist()
        n_group1 = min(n_cells, len(group1_indices))
        sampled_indices.extend(rng.choice(group1_indices, size=n_group1, replace=False))
        group2_indices = adata.obs.index[adata.obs[groupby_col] == group2].tolist()
        n_group2 = min(n_cells, len(group2_indices))
        sampled_indices.extend(rng.choice(group2_indices, size=n_group2, replace=False))
        return adata[sampled_indices, :].copy()

    adata_subset = adata[
        (adata.obs[cluster_column] == cluster_name) &
        (adata.obs[condition_column].isin([condition_treatment, condition_reference]))
    ].copy()

    if do_subsample:
        adata_subset_sampled = subsample_balanced(
            adata_subset,
            condition_column,
            condition_treatment,
            condition_reference,
            n_cells=n_cells_per_group,
            rs=random_state
        )
    else:
        adata_subset_sampled = adata_subset.copy()

    rgg_key = f"rank_genes_groups_{cluster_name}_{condition_treatment}_vs_{condition_reference}"
    if rgg_key in adata_subset_sampled.uns:
        del adata_subset_sampled.uns[rgg_key]

    sc.tl.rank_genes_groups(
        adata_subset_sampled,
        layer='normalized',
        groupby=condition_column,
        groups=[condition_treatment],
        reference=condition_reference,
        method=method,
        pts=True,
        key_added=rgg_key,
        n_jobs=-1
    )

    df_deg_full = sc.get.rank_genes_groups_df(
        adata_subset_sampled,
        group=condition_treatment,
        key=rgg_key,
        pval_cutoff=1,
        log2fc_min=0
    )

    pts_df = pd.DataFrame(adata_subset_sampled.uns[rgg_key]['pts'])
    df_deg_full = df_deg_full.set_index('names')
    df_deg_full[f'pct_nz_{condition_treatment}'] = pts_df[condition_treatment]
    df_deg_full[f'pct_nz_{condition_reference}'] = pts_df[condition_reference]
    df_deg_full['pct_nz_diff'] = df_deg_full[f'pct_nz_{condition_treatment}'] - df_deg_full[f'pct_nz_{condition_reference}']
    df_deg_full = df_deg_full.reset_index()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        df_deg_full['logfoldchanges'],
        -np.log10(df_deg_full['pvals_adj']),
        s=10,
        alpha=0.5,
        c='gray'
    )
    
    sig_genes_plot = df_deg_full[
        (df_deg_full['pvals_adj'] < pval_cutoff) &
        (df_deg_full['logfoldchanges'].abs() > log2fc_min)
    ]
    
    plt.scatter(
        sig_genes_plot['logfoldchanges'],
        -np.log10(sig_genes_plot['pvals_adj']),
        s=15,
        alpha=0.8,
        c='red'
    )

    plt.axhline(y=-np.log10(pval_cutoff), color='blue', linestyle='--', linewidth=1)
    plt.axvline(x=log2fc_min, color='blue', linestyle='--', linewidth=1)
    plt.axvline(x=-log2fc_min, color='blue', linestyle='--', linewidth=1)
    
    plt.title(f'Volcano Plot: {cluster_name} ({condition_treatment} vs {condition_reference})')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10(Adjusted p-value)')
    
    genes_to_label = df_deg_full.sort_values('logfoldchanges', ascending=False).head(n_top_genes_plot)
    for i, row in genes_to_label.iterrows():
        plt.text(row['logfoldchanges'], -np.log10(row['pvals_adj']), row['names'], fontsize=8)

    plt.show()

    df_deg_sig = df_deg_full[
        (df_deg_full['logfoldchanges'] > log2fc_min) &
        (df_deg_full['pvals_adj'] < pval_cutoff) &
        (df_deg_full[f'pct_nz_{condition_treatment}'] >= min_pct_nz_in_group)
    ].copy()

    df_deg_sig = df_deg_sig.sort_values(by='logfoldchanges', ascending=False)
    
    print(f"Found {df_deg_sig.shape[0]} significant up-regulated genes after filtering.")

    if not df_deg_sig.empty:
        top_genes_for_plot = df_deg_sig['names'].head(n_top_genes_plot).tolist()
        
        sc.pl.dotplot(
            adata_subset_sampled,
            var_names=top_genes_for_plot,
            groupby=condition_column,
            use_raw=False,
            layer=plot_layer,
            title=f'Top {len(top_genes_for_plot)} DEGs in {cluster_name}',
            swap_axes=True, vmin=0, vmax=4,
        )

    df_deg_sig['cluster'] = cluster_name
    df_deg_sig['comparison'] = f'{condition_treatment} versus {condition_reference}'
    
    return df_deg_sig.drop('scores',axis = 'columns')
#===

#===
immune_markers_human={
    't_cd8_naive': ['SELL', 'IFNG', 'NKG7', 'PTPRC', 'LEF1', 'CCR7', 'CD3E', 'CD8A', 'TCF7'],
    't_cd8_effector': ['PRF1', 'IFNG', 'GZMB', 'NKG7', 'PTPRC', 'CD3E', 'CD8A', 'EOMES', 'GZMA'],
    't_cd8_exhausted': ['HAVCR2', 'LAG3', 'TOX', 'PTPRC', 'TIGIT', 'CTLA4', 'PDCD1', 'LAYN', 'CD3E', 'CD8A', 'CD200', 'NR4A1'],
    't_cd8_trm': ['PTPRC', 'CXCL13', 'ZNF683', 'CD3E', 'ITGAE', 'CD8A'],
    't_cd8_circulating': ['CD8A', 'CX3CR1', 'CD3E', 'PTPRC'],
    't_cd4_naive': ['CD4', 'SELL', 'PTPRC', 'LEF1', 'CCR7', 'CD3E', 'TCF7'],
    't_cd4_effector': ['CD4', 'CD3E', 'GZMA', 'PTPRC'],
    't_cd4_trm_th_th1': ['CD4', 'PTPRC', 'CXCL13', 'CD3E', 'ITGAE'],
    't_cd4_circulating': ['CD4', 'CX3CR1', 'CD3E', 'PTPRC'],
    't_treg_naive': ['CD4', 'PTPRC', 'TNFRSF4', 'CCR7', 'FOXP3'],
    't_treg_activated': ['CD4', 'PTPRC', 'TNFRSF9', 'CCR8', 'TNFRSF4', 'FOXP3'],
    't_treg_peripheral': ['CD4', 'PTPRC', 'CCR4', 'TNFRSF4', 'FOXP3'],
    'nk_cd56_naive': ['KLRF1', 'KLRD1', 'GNLY', 'KLRB1', 'NCAM1'],
    'nk1_xcl+': ['XCL1', 'XCL2'],
    'nk1_cytotox_xcl-': ['GZMB', 'PRF1'],
    'b_naive': ['SELL', 'CD79A', 'MS4A1', 'CD19'],
    'b_memory': ['IGJ', 'MS4A1', 'CD79A', 'CD19'],
    'b_plasma': ['IGJ', 'MS4A1', 'CD79A', 'CD19'],
    'monocyte': ['FCGR1A', 'S100A8', 'CD14', 'CSF1R', 'S100A9'],
    'macrophage': ['FCGR1A', 'CD14', 'CSF1R', 'CD68'],
    'mast_cell': ['KIT', 'TPSAB1', 'CPA3', 'TPSB2'],
    'dc_pdc': ['IL3RA', 'GZMB', 'ITGAM', 'HLADRA', 'LILRA4', 'CLEC4C', 'ITGAX'],
    'dc_cdc1': ['CLEC9A', 'ITGAM', 'ITGAX', 'HLADRA'],
    'dc_cdc2_cdc3': ['CD1C', 'ITGAM', 'HLADRA', 'CLEC10A', 'FCER1A', 'ITGAX']
 }
immune_markers_mouse={
    't_cd8_naive': ['Sell', 'Ifng', 'Nkg7', 'Ptprc', 'Lef1', 'Ccr7', 'Cd3e', 'Cd8a', 'Tcf7'],
    't_cd8_effector': ['Prf1', 'Ifng', 'Gzmb', 'Nkg7', 'Ptprc', 'Cd3e', 'Cd8a', 'Eomes', 'Gzma'],
    't_cd8_exhausted': ['Havcr2', 'Lag3', 'Tox', 'Ptprc', 'Tigit', 'Ctla4', 'Pdcd1', 'Laycn', 'Cd3e', 'Cd8a', 'Cd200', 'Nr4a1'],
    't_cd8_trm': ['Ptprc', 'Cxcl13', 'Zfp683', 'Cd3e', 'Itgae', 'Cd8a'],
    't_cd8_circulating': ['Cd8a', 'Cx3cr1', 'Cd3e', 'Ptprc'],
    't_cd4_naive': ['Cd4', 'Sell', 'Ptprc', 'Lef1', 'Ccr7', 'Cd3e', 'Tcf7'],
    't_cd4_effector': ['Cd4', 'Cd3e', 'Gzma', 'Ptprc'],
    't_cd4_trm_th_th1': ['Cd4', 'Ptprc', 'Cxcl13', 'Cd3e', 'Itgae'],
    't_cd4_circulating': ['Cd4', 'Cx3cr1', 'Cd3e', 'Ptprc'],
    't_treg_naive': ['Cd4', 'Ptprc', 'Tnfrsf4', 'Ccr7', 'Foxp3'],
    't_treg_activated': ['Cd4', 'Ptprc', 'Tnfrsf9', 'Ccr8', 'Tnfrsf4', 'Foxp3'],
    't_treg_peripheral': ['Cd4', 'Ptprc', 'Ccr4', 'Tnfrsf4', 'Foxp3'],
    'nk_cd56_naive': ['Klrk1', 'Klrd1', 'Gm3762', 'Klrb1', 'Ncamb'],
    'nk1_xcl+': ['Xcl1', 'Xcl2'],
    'nk1_cytotox_xcl-': ['Gzmb', 'Prf1'],
    'b_naive': ['Sell', 'Cd79a', 'Ms4a1', 'Cd19'],
    'b_memory': ['Igj', 'Ms4a1', 'Cd79a', 'Cd19'],
    'b_plasma': ['Igj', 'Ms4a1', 'Cd79a', 'Cd19'],
    'monocyte': ['Fcgr1', 'S100a8', 'Cd14', 'Csf1r', 'S100a9'],
    'macrophage': ['Fcgr1', 'Cd14', 'Csf1r', 'Cd68'],
    'mast_cell': ['Kit', 'Tpsb2', 'Cpa3', 'Tpsb2'],
    'dc_pdc': ['Il3ra', 'Gzmb', 'Itgam', 'H2-Ab1', 'Lilra4', 'Clec4c', 'Itgax'],
    'dc_cdc1': ['Clec9a', 'Itgam', 'Itgax', 'H2-Ab1'],
    'dc_cdc2_cdc3': ['Cd1d1', 'Itgam', 'H2-Ab1', 'Clec10a', 'Fcer1a', 'Itgax']}

# mouse
signaling_geneset={
    'ifng':'Raf1,Ptpn6,Ifngr1,Camk2g,Prkcd,Ifngr2,Ptpn2,Camk2a,Jak2,Sumo1,Ybx1,Pias1,Socs1,Socs3,Camk2d,Ifng,Camk2b,Mapk3,Mapk1'.split(','), # https://www.gsea-msigdb.org/gsea/msigdb/mouse/geneset/REACTOME_INTERFERON_GAMMA_SIGNALING.html
    'tfna':'Mcl1,Cd80,Traf1,F2rl1,Dusp2,Tnc,Fosl2,Stat5a,Vegfa,Efna1,Relb,Rela,Cebpd,Ptger4,Cdkn1a,Ptx3,Il15ra,Atp2b1,Nfkbia,Tnf,Ier3,Ier2,Hes1,Tnfaip2,Dusp1,Eif1,Fosl1,Bcl6,Ifngr2,Tank,Gadd45b,Gadd45a,Tubb2a,Sqstm1,Il18,Rhob,Cxcl1,Btg2,Nfe2l2,Tsc22d1,Irs2,Atf3,Nfil3,Btg3,Jag1,Ackr3,Cxcl5,Phlda1,Bhlhe40,Per1,Plk2,Nfkb2,Tnfsf9,Tnfrsf9,Klf10,Tgif1,Nfkbie,Tnfaip6,Tnfaip3,Ninj1,Birc3,Birc2,Smad3,Dennd5a,Socs3,Phlda2,Olr1,Snn,Bcl2a1d,Egr3,Sphk1,G0s2,Cd83,Ccl20,Klf9,Msc,Cflar,Ier5,Gfpt2,Csf2,Csf1,Sgk1,Cxcl2,Ehd1,Fjx1,Klf4,Klf2,Tlr2,Klf6,Map2k3,Map3k8,Sdc4,Cxcl10,Nr4a1,Nr4a2,Nr4a3,Icosl,Nfat5,Panx1,Cxcl11,Plek,Rcan1,Ripk2,Dnajb4,Plpp3,Pnrc1,Kynu,Ifih1,Dram1,Ccrl2,Spsb1,Rnf19b,Ccnl1,Tnip1,Ppp1r15a,B4galt5,Pdlim5,Litaf,Pmepa1,Nampt,Clcf1,Il23a,Zbtb10,Slc16a6,Trip10,Tnfaip8,Tiparp,Pfkfb3,Zc3h12a,Tnip2,Yrdc,Gpr183,Dusp4,Rigi,Slc2a6,Trib1,Kdm6b,Dusp5,Areg,Bcl3,Bmp2,Btg1,Ccnd1,Cd44,Cd69,Cebpb,F3,Ccn1,Serpinb8,Edn1,Egr1,Egr2,Ets2,Fos,Fosb,Fut4,Gch1,B4galt1,Slc2a3,Hbegf,Icam1,Id2,Il12b,Il1a,Il1b,Il6,Il6st,Il7r,Inhba,Irf1,Jun,Junb,Ldlr,Lif,Marcks,Mxd1,Maff,Myc,Nfkb1,Serpine1,Serpinb2,Plau,Plaur,Ptgs2,Ptpre,Rel,Sat1,Ccl5,Sod2,Tap1,Zfp36,Ifit2,Pde4b,Abca1,Gem,Lamb3'.split(','),
    'nfkb':'Tnfrsf1a,Tradd,Ikbkb,Ikbkg,Chuk,Tnfaip3,Irak1,Il1r1,Nfkb1,Rela,Ripk1,Traf6,Fadd,Il1a,Myd88,Nfkbia,Tnfrsf1a,Tnfrsf1b,Map3k1,Tnf,Map3k14,Map3k7'.split(','),
    'tgfb':'Ppp1ca,Nog,Cdkn1c,Fnta,Skil,Xiap,Ifngr2,Hdac1,Slc20a1,Smad1,Bmpr2,Rhoa,Smad7,Klf10,Tgif1,Smad3,Hipk2,Cdk9,Smad6,Ncor2,Bmpr1a,Map3k7,Bcar3,Ube2d3,Smurf2,Rab31,Wwtr1,Smurf1,Ppp1r15a,Pmepa1,Trim33,Arid4b,Acvr1,Apc,Bmp2,Ctnnb1,Cdh1,Eng,Fkbp1a,Id1,Id2,Id3,Junb,Furin,Serpine1,Ski,Sptbn1,Tgfb1,Tgfbr1,Thbs1,Tjp1,Ltbp2,Ppm1a'.split(','),
}

#---------------------------
def split_gene_barplot(adata, gene_name, groupby_col='celltype_id', split_by='genotype', layer='log1p'):
    if layer == 'X' or layer is None:
        gene_expression = adata[:, gene_name].X.toarray().flatten()
    elif layer in adata.layers:
        gene_expression = adata[:, gene_name].layers[layer].toarray().flatten()
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers or invalid layer specified.")

    plot_data = pd.DataFrame({
        groupby_col: adata.obs[groupby_col],
        split_by: adata.obs[split_by],
        gene_name: gene_expression
    })

    y_label_suffix = ""
    if layer == 'log1p':
        y_label_suffix = " (log1p)"
    elif layer == 'normalized':
        y_label_suffix = " (Normalized)"

    y_label = f'Mean {gene_name} Expression{y_label_suffix}'

    summary_data = plot_data.groupby([groupby_col, split_by])[gene_name].mean().reset_index()
    summary_data.columns = [groupby_col, split_by, 'mean_expression']

    plot = (
        p9.ggplot(summary_data, p9.aes(x=groupby_col, y='mean_expression', fill=split_by))
        + p9.geom_col(position=p9.position_dodge(width=0.9), stat='identity')
        + p9.labs(title=gene_name,x='',y=y_label) # Changed y label to reflect mean expression
        + p9.coord_flip()
        + utils.theme_nizar() 
        # + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1), figure_size=(12,6))
        # + p9.facet_grid(f'~{groupby_col}') #, scales='free_x')
        # + p9.guides(fill=p9.guide_legend(title=fill_col.replace('_', ' ').title())) # Changed fill_col to split_by
    )
    return(plot)

###-
import sys

def remove_mito_genes(adata, species="human"):
    if species == "human":
        mito_prefix = "MT-"
    elif species == "mouse":
        mito_prefix = "mt-"
    else:
        sys.stderr.write("Warning: Unknown species. No mitochondrial genes will be filtered.\n")
        mito_prefix = ""

    if mito_prefix:
        non_mito_genes = [gene for gene in adata.var_names if not gene.startswith(mito_prefix)]
        adata_filtered = adata[:, non_mito_genes].copy()
    else:
        adata_filtered = adata # Return the original adata if no filtering

    return adata_filtered
###-
def split_adata(adata, split_by):
    unique_values = adata.obs[split_by].unique()
    subset_adatas = []

    for value in unique_values:
        subset_adata = adata[adata.obs[split_by] == value].copy()
        subset_adatas.append(subset_adata)

    return subset_adatas
####-



#-----------------------------------------------------------------------
import types
pl = types.SimpleNamespace(
    add_temp_variables_for_plotting=add_temp_variables_for_plotting,
    qc_plot_violin=qc_plot_violin,
    sample_pca_plot=sample_pca_plot,
    split_umap=split_umap,
    score_clusters_by_gene_sets=score_clusters_by_gene_sets,
    split_gene_barplot=split_gene_barplot,
    #enclone_plot=enclone_plot,
    #shared_clonotype_expansion=shared_clonotype_expansion,
    #plot_clonal_expansion=plot_clonal_expansion,    
    )

tl = types.SimpleNamespace(
    split_adata=split_adata,
    remove_mito_genes=remove_mito_genes,
    infer_t_celltypes=infer_t_celltypes,    
    filter_geneset=filter_geneset,
    deg_by_condition=deg_by_condition,
)

pp = types.SimpleNamespace(
    step1_load_with_layers_no_concat=step1_load_with_layers_no_concat,
    step2_filter_data_using_qc_metrics=step2_filter_data_using_qc_metrics,
    step3_integrate=step3_integrate,
)



