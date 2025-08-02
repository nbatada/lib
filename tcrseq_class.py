import os, re, sys, hashlib
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import plotnine as p9
from scipy.stats import entropy
from upsetplot import from_indicators
# use our unit function from mizani (or fallback defined earlier)

try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    venn2, venn3 = None, None
from upsetplot import from_indicators, plot as upset_plot

try:
    import cowpatch as cp  # Use cp.ggplot_grid to merge plots.
except ImportError:
    cp = None

def unit(value, unit_type):
    # Fallback: simply return a string that ggplot2 might interpret.
    return f"{value} {unit_type}"


# ----------------------
# A “professional” ggplot theme (mimicking a scienceplots–like style)
# ----------------------
def professional_theme():
    return (p9.theme(
        panel_background=p9.element_rect(fill="white", color=None),
        panel_grid_major=p9.element_line(linetype="dotted", color="grey", size=0.5),
        panel_grid_minor=p9.element_line(linetype="dotted", color="grey", size=0.2),
        panel_border=p9.element_rect(color="grey", fill=None),
        legend_title=p9.element_text(face="bold", size=8),
        legend_text=p9.element_text(size=7),
        legend_direction="horizontal",
        legend_position="top",
        axis_title=p9.element_text(size=9, weight="bold"),
        axis_text_x=p9.element_text(rotation=25, hjust=1, size=8),
        axis_text_y=p9.element_text(size=8),
        strip_text=p9.element_text(size=8)
    ))


# ----------------------
# TCRSeq Class
# ----------------------
class TCRSeq:
    def __init__(self, df: pd.DataFrame, obs: pd.DataFrame = None,
                 data_subset: str = "all", project_name: str = "DefaultProject"):
        self.df = df.copy()   # main data (clone-level)
        # If obs is not provided, create an empty DataFrame with index = unique sample_id
        self.obs = obs.copy() if obs is not None else pd.DataFrame(index=self.samples)
        self.data_subset = data_subset  # renamed from data_name
        self.project_name = project_name
        # Create namespace-like attributes for preprocessing (pp), tools (tl), and plotting (pl)
        self.pp = _TCRSeqPreprocessing(self)
        self.tl = _TCRSeqTools(self)
        self.pl = _TCRSeqPlotting(self)
    
    def __repr__(self):
        lines = []
        lines.append(f"<TCRSeq object at {hex(id(self))}>")
        lines.append(f"project_name: {self.project_name}")
        lines.append(f"data_subset: {self.data_subset}")
        lines.append(f"[.df] -- has all raw data (clonotypes)")
        lines.append("----------------------------------------------")
        lines.append(f"\tshape: {self.df.shape}")
        lines.append(f"\tUnique samples: {len(self.samples)}")
        lines.append(f"\tUnique clones: {len(self.clones)}")
        lines.append(f"\tcolumns:\n\t\t{', '.join(self.df.columns.tolist())}")
        lines.append("\n")
        lines.append(f"[.X]  -- counts matrix (clones x samples)")
        lines.append("----------------------------------------------")
        lines.append(f"\tshape: {self.X.shape}")
        lines.append(f"\tcolumns:\n\t\t{', '.join(self.X.columns.tolist())}")
        lines.append("\n")
        if self.obs is not None and not self.obs.empty:
            lines.append(f"[.obs] -- sample metadata (includes stats if available)")
            lines.append("----------------------------------------------")
            lines.append(f"\tshape: {self.obs.shape}")
            lines.append(f"\tcolumns:\n\t\t{', '.join(self.obs.columns.tolist())}")
        return "\n".join(lines)
    
    @property
    def samples(self):
        return sorted(self.df['sample_id'].unique())
    
    @property
    def clones(self):
        return self.df['combined_custom_id'].unique()
    
    @property
    def X(self) -> pd.DataFrame:
        """Counts matrix (clones x samples)."""
        return self.tl.get_counts_matrix()

    def get_stats(self) -> pd.DataFrame:
        """
        Get per-sample reading statistics.
        Returns a DataFrame that merges the sample metadata (.obs) with
        the statistics computed via tl.get_sample_stats(), without modifying self.obs.
        """
        stats_df = self.tl.get_sample_stats()
        if not self.obs.empty:
            # Join on sample_id without updating self.obs
            joined = self.obs.join(stats_df.set_index('sample_id'), how="left").reset_index()
            return joined
        return stats_df

    def get_top_clones(self, sample_ids: list, n: int = 10) -> pd.DataFrame:
        """
        Return a subset of the counts matrix (.X) that corresponds to the union
        of the top n clones (by count) for each sample in sample_ids.
        """
        X_sub = self.X[sample_ids]
        top_clones = set()
        for s in sample_ids:
            # Get the top n clone ids for this sample
            top_ids = X_sub[s].sort_values(ascending=False).head(n).index
            top_clones.update(top_ids)
        return self.X.loc[list(top_clones), sample_ids]

    def get_clone_count(self, clone_ids: list) -> dict:
        """
        Given a list of clonotype IDs, return a dictionary of their overall counts
        (summing across samples) as obtained from the counts matrix.
        """
        counts = {}
        for cid in clone_ids:
            if cid in self.X.index:
                counts[cid] = self.X.loc[cid].sum()
            else:
                counts[cid] = 0
        return counts

    def _get_reading_stats(self) -> dict:
        # Compute total reads and unique clones (using combined_custom_id)
        n_total_clones = self.df.shape[0]
        n_unique_clones = self.df['combined_custom_id'].nunique()
        # For productive clones use combined_custom_id for consistency
        n_clones_prod = (self.df[self.df['productive'] == True]['combined_custom_id']
                         .nunique() if 'productive' in self.df.columns else n_unique_clones)
        n_TRA = (self.df[self.df['TRA_custom_id'] != '']['TRA_custom_id']
                 .nunique() if 'TRA_custom_id' in self.df.columns else 0)
        n_TRB = (self.df[self.df['TRB_custom_id'] != '']['TRB_custom_id']
                 .nunique() if 'TRB_custom_id' in self.df.columns else 0)
        pct_TRB = float(n_TRB / n_clones_prod) if n_clones_prod > 0 else 0
        pct_TRA = float(n_TRA / n_clones_prod) if n_clones_prod > 0 else 0
        sample_id = self.df['sample_id'].unique()[0]
        if 'consensus_count' in self.df.columns:
            clone_counts = self.df.groupby('combined_custom_id')['consensus_count'].sum()
            pct_clones_gt1 = (clone_counts > 1).sum() / n_unique_clones * 100 if n_unique_clones > 0 else np.nan
        else:
            pct_clones_gt1 = np.nan
        return { 'sample_id': sample_id,
                 'n_total_clones': n_total_clones, # total (assembled) clones
                 'n_unique_clones': n_unique_clones, # unique clones count
                 'pct_TRA': pct_TRA,
                 'pct_TRB': pct_TRB,
                 'pct_clones_gt1': pct_clones_gt1}
                     
    @classmethod
    def from_file(cls, filename: str, format_type: str = 'airr', barcode_suffix: str = None,
                  assay_type: str = 'single_cell', filter_productive: bool = True,
                  filter_IG: bool = True, filter_TRD_TRG: bool = True,
                  project_name: str = "DefaultProject"):
        if format_type == 'cellranger':
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, sep='\t', header=0)
        if format_type == 'cellranger':
            mapping = {'chain': 'locus', 'v_gene': 'v_call',
                       'd_gene': 'd_call', 'j_gene': 'j_call',
                       'cdr3': 'junction_aa'}
            df.rename(columns=mapping, inplace=True)
        if 'barcode' not in df.columns:
            if format_type == 'airr':
                df['barcode'] = barcode_suffix if barcode_suffix else os.path.basename(filename).replace('.tsv','')
            else:
                raise ValueError("barcode column missing")
        if barcode_suffix:
            df['barcode'] = df['barcode'].astype(str) + '_' + barcode_suffix
        if 'productive' in df.columns:
            df['productive'] = df['productive'].astype(str).str.upper().map({
                'T': True, 'TRUE': True, 'F': False, 'FALSE': False
            }).fillna(False)
        else:
            df['productive'] = True
        
        # Filter out non-TCR clones if specified
        filtered_ig_count = 0
        filtered_trd_trg_count = 0
        if filter_IG:
            filtered_ig_count = df['locus'].isin(['IGH', 'IGL']).sum()
            df = df[~df['locus'].isin(['IGH', 'IGL'])]
        if filter_TRD_TRG:
            filtered_trd_trg_count = df['locus'].isin(['TRD', 'TRG']).sum()
            df = df[~df['locus'].isin(['TRD', 'TRG'])]
        if filtered_ig_count > 0 or filtered_trd_trg_count > 0:
            print(f"Note: Filtered out {filtered_ig_count} IG clones and {filtered_trd_trg_count} TRD/TRG clones from {filename}.", file=sys.stderr)
        
        # Make a canonical clonotype id (all upper-case)
        df['chain_clonotype_id'] = (df['v_call'].fillna('').astype(str).str.upper() + '_' +
                                    df['junction_aa'].fillna('').astype(str).str.upper() + '_' +
                                    df['j_call'].fillna('').astype(str).str.upper())
        # Option to filter for productive chains (default True)
        if filter_productive:
            proc = df[df['productive'] == True].copy()
        else:
            proc = df.copy()
        if assay_type == 'single_cell':
            proc['TRA_custom_id'] = ''
            proc['TRB_custom_id'] = ''
            proc.loc[proc['locus'].isin(['TRA','TRAC']), 'TRA_custom_id'] = proc['chain_clonotype_id']
            proc.loc[proc['locus'].isin(['TRB','TRBC']), 'TRB_custom_id'] = proc['chain_clonotype_id']
            grouped = proc.groupby('barcode').agg(
                TRA_custom_id = ('TRA_custom_id', lambda x: ','.join(x[x != ''])),
                TRB_custom_id = ('TRB_custom_id', lambda x: ','.join(x[x != ''])),
                n_tra = ('locus', lambda x: (x.isin(['TRA','TRAC'])).sum()),
                n_trb = ('locus', lambda x: (x.isin(['TRB','TRBC'])).sum())
            ).reset_index()
            def get_cat(row):
                ntra, ntrb = row['n_tra'], row['n_trb']
                tra_id, trb_id = row['TRA_custom_id'], row['TRB_custom_id']
                if ntra == 1 and ntrb == 1:
                    return 'paired', f"{tra_id};{trb_id}"
                elif ntra == 1 and ntrb == 0:
                    return 'orphan_TRA', tra_id
                elif ntra == 0 and ntrb == 1:
                    return 'orphan_TRB', trb_id
                elif ntra > 1 and ntrb == 0:
                    return 'extra_TRA', tra_id
                elif ntra == 0 and ntrb > 1:
                    return 'extra_TRB', trb_id
                elif ntra > 1 and ntrb == 1:
                    return 'extra_TRA_and_paired_TRB', f"{tra_id};{trb_id}"
                elif ntra == 1 and ntrb > 1:
                    return 'paired_TRA_and_extra_TRB', f"{tra_id};{trb_id}"
                elif ntra > 1 and ntrb > 1:
                    return 'extra_both', f"{tra_id};{trb_id}"
                else:
                    return 'unknown', ''
            cats = grouped.apply(lambda r: pd.Series(get_cat(r), index=['category','combined_custom_id']), axis=1)
            grouped = pd.concat([grouped, cats], axis=1)
            res = grouped[['barcode','TRA_custom_id','TRB_custom_id','combined_custom_id','category']].copy()
        else:
            proc['TRA_custom_id'] = ''
            proc.loc[proc['locus'].isin(['TRA','TRAC']), 'TRA_custom_id'] = proc['chain_clonotype_id']
            proc['TRB_custom_id'] = ''
            proc.loc[proc['locus'].isin(['TRB','TRBC']), 'TRB_custom_id'] = proc['chain_clonotype_id']
            res = proc.copy()
            res['combined_custom_id'] = res['chain_clonotype_id']
        def safe_md5(text: str) -> str:
            return hashlib.md5(text.encode('utf-8')).hexdigest() if isinstance(text, str) and text.strip() != '' else ''
        res['tra_univ_md5'] = res['TRA_custom_id'].apply(safe_md5)
        res['trb_univ_md5'] = res['TRB_custom_id'].apply(safe_md5)
        res['combined_univ_md5'] = res['combined_custom_id'].apply(safe_md5)
        m = re.search(r'(P\d{2}E\d{2}S\d{2})', filename)
        sid = m.group(1) if m else os.path.basename(filename)
        res['sample_id'] = sid
        # For a single-sample load, data_subset is set to the sample id.
        return cls(res, data_subset=sid, project_name=project_name)
    
    @classmethod
    def from_sample_info(cls, sample_info: pd.DataFrame, filename_col: str = 'airr_filename',
                         sample_id_col: str = 'sample_id', project_name: str = "DefaultProject", **kwargs):
        dfs = []
        stats_list = []
        
        for i, row in sample_info.iterrows():
            sample_id = row[sample_id_col]
            filename = row[filename_col]
            # Print loading info to stderr if desired.
            tcr_sample = cls.from_file(filename, project_name=project_name, **kwargs)
            # Override sample_id using sample_info.
            tcr_sample.df['sample_id'] = sample_id
            stats = tcr_sample._get_reading_stats()
            stats_list.append(stats)
            dfs.append(tcr_sample.df)
        combined_df = pd.concat(dfs, ignore_index=True)
        obs = sample_info.set_index(sample_id_col)
        if 'group' in obs.columns:  
            obs['group'] = obs['group'].astype('category')
        
        print(f"Loaded {len(dfs)} samples with a total of {combined_df.shape[0]} rows.", file=sys.stderr)
        stats_df = pd.DataFrame(stats_list)
        # For multiple samples, data_subset is set to "all".
        return cls(combined_df, obs, data_subset="all", project_name=project_name)

# -----------------------------
# Preprocessing Namespace (pp)
# -----------------------------
class _TCRSeqPreprocessing:
    def __init__(self, adata: TCRSeq):
        self.adata = adata
    
    def filter_by_min_count(self, min_count: int) -> TCRSeq:
        counts = self.adata.df.groupby('combined_custom_id').size()
        keep = counts[counts >= min_count].index
        new_df = self.adata.df[self.adata.df['combined_custom_id'].isin(keep)].copy()
        new_obs = (self.adata.obs.loc[self.adata.obs.index.isin(new_df['sample_id'].unique())]
                   if self.adata.obs is not None and not self.adata.obs.empty
                   else self.adata.obs)
        return TCRSeq(new_df, new_obs, data_subset=self.adata.data_subset, project_name=self.adata.project_name)
    
    def filter_shared_clones(self, min_samples: int, group: str = None) -> TCRSeq:
        if group and self.adata.obs is not None and 'group' in self.adata.obs.columns:
            group_samples = self.adata.obs[self.adata.obs['group'] == group].index.tolist()
        else:
            group_samples = self.adata.samples
        counts = self.adata.df[self.adata.df['sample_id'].isin(group_samples)].groupby('combined_custom_id')['sample_id'].nunique()
        keep = counts[counts >= min_samples].index
        new_df = self.adata.df[self.adata.df['combined_custom_id'].isin(keep)].copy()
        new_obs = (self.adata.obs.loc[self.adata.obs.index.isin(new_df['sample_id'].unique())]
                   if self.adata.obs is not None and not self.adata.obs.empty
                   else self.adata.obs)
        return TCRSeq(new_df, new_obs, data_subset=self.adata.data_subset, project_name=self.adata.project_name)
    
    def subset_by_group(self, group_value: str) -> TCRSeq:
        if self.adata.obs is None or 'group' not in self.adata.obs.columns:
            raise ValueError("Metadata is missing a 'group' column.")
        samples = self.adata.obs[self.adata.obs['group'] == group_value].index.tolist()
        sub_df = self.adata.df[self.adata.df['sample_id'].isin(samples)].copy()
        new_obs = self.adata.obs.loc[samples].copy() if not self.adata.obs.empty else self.adata.obs
        new_data_subset = f"{self.adata.data_subset}_group:{group_value}"
        return TCRSeq(sub_df, new_obs, data_subset=new_data_subset, project_name=self.adata.project_name)
    
    def subset_by_chain(self, chain: str) -> TCRSeq:
        valid = {"TRA": ["TRA", "TRAC"], "TRB": ["TRB", "TRBC"]}
        if chain not in valid:
            raise ValueError("chain must be either 'TRA' or 'TRB'")
        subset_df = self.adata.df[self.adata.df['locus'].isin(valid[chain])].copy()
        sample_ids = subset_df['sample_id'].unique()
        new_obs = (self.adata.obs.loc[self.adata.obs.index.isin(sample_ids)].copy() 
                   if self.adata.obs is not None and not self.adata.obs.empty 
                   else self.adata.obs)
        new_data_subset = f"{self.adata.data_subset}_chain:{chain}"
        return TCRSeq(subset_df, new_obs, data_subset=new_data_subset, project_name=self.adata.project_name)

# -------------------------
# Tools Namespace (tl)
# -------------------------
class _TCRSeqTools:
    def __init__(self, adata: TCRSeq):
        self.adata = adata
    
    def get_counts_matrix(self) -> pd.DataFrame:
        if 'consensus_count' in self.adata.df.columns:
            cm = self.adata.df.pivot_table(index='combined_custom_id', columns='sample_id',
                                           values='consensus_count', aggfunc='sum', fill_value=0)
        else:
            cm = self.adata.df.groupby(['combined_custom_id', 'sample_id']).size().unstack(fill_value=0)
        return cm
    
    def compute_entropy(self) -> pd.Series:
        cm = self.get_counts_matrix()
        ents = {}
        for s in cm.columns:
            arr = cm[s].values
            tot = arr.sum()
            ents[s] = entropy(arr / tot) if tot > 0 else np.nan
        return pd.Series(ents)
    
    def compute_gini(self) -> pd.Series:
        def gini(x):
            if np.all(x == 0):
                return 0
            sx = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sx)
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        cm = self.get_counts_matrix()
        return pd.Series({s: gini(cm[s].values) for s in cm.columns})
    
    def compute_simpson(self) -> pd.Series:
        cm = self.get_counts_matrix()  # counts matrix: clones x samples
        simpson_results = {}
        for s in cm.columns:
            counts = cm[s].values.astype(float)
            tot = counts.sum()
            if tot > 0:
                p = counts / tot
                simpson_results[s] = 1 - np.sum(p ** 2)
            else:
                simpson_results[s] = np.nan
        return pd.Series(simpson_results)

    
    def most_common_clones(self, top_n: int = 5) -> dict:
        cm = self.get_counts_matrix()
        common = {}
        for s in cm.columns:
            common[s] = cm[s].sort_values(ascending=False).head(top_n)
        return common
    
    def get_sample_stats(self) -> pd.DataFrame:
        stats_list = []
        for sample, sub in self.adata.df.groupby('sample_id'):
            temp_obj = TCRSeq(sub,
                              self.adata.obs.loc[[sample]] if (self.adata.obs is not None and sample in self.adata.obs.index) else None,
                              data_subset=self.adata.data_subset, project_name=self.adata.project_name)
            stats_list.append(temp_obj._get_reading_stats())
        return pd.DataFrame(stats_list)
    def get_shared_clones(self, n_top: int = 10) -> pd.DataFrame:
        """
        Returns a DataFrame of the top n shared clones across samples.
        Computation: uses the counts matrix to compute the mean clone count across samples,
        resets the index, and then returns the top n clones sorted in descending order
        by the mean count.
        """
        cm = self.get_counts_matrix()
        # Compute the mean count for each clone
        mean_series = cm.mean(axis=1)
        # Reset the index to create a DataFrame, naming the mean column 'mean_count'
        df_mean = mean_series.reset_index(name='mean_count')
        # Sort by 'mean_count' in descending order and return the top n clones
        df_mean = df_mean.sort_values('mean_count', ascending=False).head(n_top)
        return df_mean

# ----------------------------
# Plotting Namespace (pl)
# ----------------------------
class _TCRSeqPlotting:
    def __init__(self, adata):
        self.adata = adata
    def plot_sequence_stats(self, prefix: str = 'seqstats_', FILL: str = 'sample_id', 
                              x_min: float = None, y_min: float = None):
        """
        Creates scatter plots using sequence statistics from the sample metadata (.obs).
        Two plots are produced:
          1) A scatter plot with x-axis = prefix + "reads" and y-axis = prefix + "umi"
          2) A scatter plot with x-axis = prefix + "umi" and y-axis = n_unique_clones
        A best-fit (linear) line is overlaid and the Spearman rho and p-value are added to the title.
        x_min and y_min (if provided) are respected on the log10-transformed axes.
        """
        try:
            df = self.adata.get_stats()
        except ValueError as e:
            if "columns overlap" in str(e):
                stats_df = self.adata.tl.get_sample_stats().set_index('sample_id')
                overlap = self.adata.obs.columns.intersection(stats_df.columns)
                if len(overlap) > 0:
                    self.adata.obs = self.adata.obs.drop(columns=list(overlap))
                df = self.adata.get_stats()
            else:
                raise e
        
        reads_col = prefix + "reads"
        umi_col = prefix + "umi"
        if reads_col not in df.columns or umi_col not in df.columns:
            raise ValueError(f"Required columns {reads_col} and/or {umi_col} not found in sample metadata (.obs)")
        if "n_unique_clones" not in df.columns:
            raise ValueError("Column n_unique_clones not found in sample metadata (.obs)")
        
        for col in [reads_col, umi_col, "n_unique_clones"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[reads_col, umi_col, "n_unique_clones"])
        
        mapping1 = p9.aes(x=reads_col, y=umi_col)
        mapping2 = p9.aes(x=umi_col, y="n_unique_clones")
        if FILL in df.columns and FILL != "sample_id":
            mapping1 = p9.aes(x=reads_col, y=umi_col, color=FILL)
            mapping2 = p9.aes(x=umi_col, y="n_unique_clones", color=FILL)
        
        from scipy.stats import spearmanr
        rho1, pval1 = spearmanr(df[reads_col], df[umi_col])
        title1 = (f"[{self.adata.project_name}:{self.adata.data_subset}] {reads_col.removeprefix(prefix)} vs {umi_col.removeprefix(prefix)} "
                  f"(ρ={rho1:.2f}, p={pval1:.3g})")
        
        p1 = (p9.ggplot(df, mapping1) +
              p9.geom_point() +
              p9.geom_smooth(method="lm", se=False, color="black") +
              p9.labs(x=reads_col.removeprefix(prefix),
                      y=umi_col.removeprefix(prefix),
                      title=title1) +
              professional_theme() +
              p9.theme(panel_spacing=0.4))
        if x_min is not None:
            p1 = p1 + p9.scale_x_log10(limits=(x_min, None))
        else:
            p1 = p1 + p9.scale_x_log10()
        if y_min is not None:
            p1 = p1 + p9.scale_y_log10(limits=(y_min, None))
        else:
            p1 = p1 + p9.scale_y_log10()
        
        rho2, pval2 = spearmanr(df[umi_col], df["n_unique_clones"])
        title2 = (f"[{self.adata.project_name}:{self.adata.data_subset}] {umi_col.removeprefix(prefix)} vs n_unique_clones "
                  f"(ρ={rho2:.2f}, p={pval2:.3g})")
        
        p2 = (p9.ggplot(df, mapping2) +
              p9.geom_point() +
              p9.geom_smooth(method="lm", se=False, color="black") +
              p9.labs(x=umi_col.removeprefix(prefix),
                      y="n_unique_clones",
                      title=title2) +
              professional_theme() +
              p9.theme(panel_spacing=0.4))
        if x_min is not None:
            p2 = p2 + p9.scale_x_log10(limits=(x_min, None))
        else:
            p2 = p2 + p9.scale_x_log10()
        if y_min is not None:
            p2 = p2 + p9.scale_y_log10(limits=(y_min, None))
        else:
            p2 = p2 + p9.scale_y_log10()
        
        try:
            import cowpatch as cow
            import numpy as np
            vis_patch = cow.patch(p1, p2)
            vis_patch += cow.layout(design=np.array([0, 1]))
            final_plot = vis_patch
        except ImportError:
            final_plot = (p1, p2)
        print(final_plot.show())

    def plot_sample_stats(self, split_by: str = None, FILL: str = 'sample_id',
                          x_min: float = None, y_min: float = None, label_order: list = None):
        """
        Composite plot made entirely with plotnine:
          • A faceted bar plot of per‐sample metrics (excluding n_unique_clones)
          • A stacked bar plot of unique TRA and TRB clone counts per sample.
        Extra spacing between panels is added.
        x_min and y_min allow you to set the lower limits for the x and y axes.
        label_order is a list of sample_id values specifying the order of x-axis labels.
        """
        stats_df = self.adata.tl.get_sample_stats()
        
        if split_by is not None and self.adata.obs is not None and split_by in self.adata.obs.columns:
            obs_reset = self.adata.obs.reset_index()
            if split_by not in stats_df.columns:
                stats_df = stats_df.merge(obs_reset[['sample_id', split_by]], on='sample_id', how='left')
        if FILL != "sample_id" and self.adata.obs is not None and FILL in self.adata.obs.columns:
            if FILL not in stats_df.columns:
                stats_df = stats_df.merge(self.adata.obs.reset_index()[['sample_id', FILL]], on='sample_id', how='left')
        
        stats_for_melt = stats_df.drop(columns=['n_unique_clones'], errors='ignore')
        id_vars = ['sample_id']
        if split_by is not None and split_by in stats_for_melt.columns:
            id_vars.append(split_by)
        if FILL != "sample_id" and FILL in stats_for_melt.columns:
            id_vars.append(FILL)
        value_vars = [col for col in stats_for_melt.columns if col not in id_vars]
        metrics_long = stats_for_melt.melt(id_vars=id_vars, value_vars=value_vars,
                                           var_name="metric", value_name="value")
        mapping = p9.aes(x='sample_id', y='value')
        if FILL in metrics_long.columns and FILL != "sample_id":
            mapping = p9.aes(x='sample_id', y='value', fill=FILL)
        p1 = (p9.ggplot(metrics_long, mapping) +
              p9.geom_col() +
              p9.geom_text(mapping=p9.aes(label='sample_id'),
                           position=p9.position_dodge(width=0.9),
                           va='bottom', size=8) +
              p9.facet_wrap('~metric', scales='free_y') +
              p9.labs(x='Sample', y='Value', title=f"[{self.adata.project_name}:{self.adata.data_subset}] Per‐Sample Metrics") +
              professional_theme() +
              p9.theme(panel_spacing=0.4))
        if x_min is not None or y_min is not None:
            p1 = p1 + p9.coord_cartesian(
                xlim=(x_min, None) if x_min is not None else None,
                ylim=(y_min, None) if y_min is not None else None)
        if label_order is not None:
            p1 = p1 + p9.scale_x_discrete(limits=label_order)
        
        df = self.adata.df.copy()
        tra = df[df['TRA_custom_id'] != ''][['sample_id', 'TRA_custom_id']].drop_duplicates()
        trb = df[df['TRB_custom_id'] != ''][['sample_id', 'TRB_custom_id']].drop_duplicates()
        tra_counts = tra.groupby('sample_id')['TRA_custom_id'].nunique().reset_index()
        tra_counts['chain'] = 'TRA'
        tra_counts.rename(columns={'TRA_custom_id': 'clone_count'}, inplace=True)
        trb_counts = trb.groupby('sample_id')['TRB_custom_id'].nunique().reset_index()
        trb_counts['chain'] = 'TRB'
        trb_counts.rename(columns={'TRB_custom_id': 'clone_count'}, inplace=True)
        counts = pd.concat([tra_counts, trb_counts], ignore_index=True)
        p2 = (p9.ggplot(counts, p9.aes(x='sample_id', y='clone_count', fill='chain')) +
              p9.geom_col() +
              p9.labs(x='Sample', y='Clone Count', title=f"[{self.adata.project_name}:{self.adata.data_subset}] Unique Clone Counts per Sample") +
              professional_theme() +
              p9.theme(panel_spacing=0.4))
        if x_min is not None or y_min is not None:
            p2 = p2 + p9.coord_cartesian(
                xlim=(x_min, None) if x_min is not None else None,
                ylim=(y_min, None) if y_min is not None else None)
        if label_order is not None:
            p2 = p2 + p9.scale_x_discrete(limits=label_order)
        
        return p2
    def plot_shared_clone(self, split_by: str = None, n_top: int = 10):
        """
        Creates a dot plot for shared clones across samples using the top n shared clones.
        The top clones are defined as those with the highest average count across samples,
        computed via the counts matrix (using get_shared_clones).
        X-axis: sample_id, Y-axis: clone (combined_custom_id).
        Dot size reflects the clone count and color is set according to the provided split_by.
        """
        # Retrieve top shared clones (the DataFrame contains 'combined_custom_id' and 'mean_count')
        shared_df = self.adata.tl.get_shared_clones(n_top)
        # Extract the list of clone IDs (assumed to be in the same column name as in the counts matrix)
        top_clones = shared_df['combined_custom_id'].tolist()
        
        # Get the counts matrix and filter to the top clones, then melt for plotting
        cm = self.adata.X
        X_top = cm.loc[top_clones].reset_index().melt(
            id_vars='combined_custom_id',
            var_name='sample_id',
            value_name='clone_count'
        )
        
        # Merge additional group/split information if specified and available
        if split_by is not None and self.adata.obs is not None and split_by in self.adata.obs.columns:
            obs_reset = self.adata.obs.reset_index()
            X_top = X_top.merge(obs_reset[['sample_id', split_by]], on='sample_id', how='left')
            colorField = split_by
        else:
            colorField = 'sample_id'
        
        p = (p9.ggplot(X_top, p9.aes(x='sample_id', y='combined_custom_id',
                                     size='clone_count', color=colorField)) +
             p9.geom_point() +
             p9.labs(x="Sample", y="Clone", size="Clone Count", color=colorField,
                     title=f"[{self.adata.project_name}:{self.adata.data_subset}] Top {n_top} Shared Clones") +
             professional_theme() +
             p9.theme(panel_spacing=0.4, axis_text_x=p9.element_text(rotation=45, ha="right")))
        return p

    def plot_saturation_curve(self, randomize: bool = False, n_iter: int = 100):
        """
        Plots a saturation curve (number of unique clones vs cumulative samples).
        For randomize=True, error bars (std dev) are added over multiple iterations.
        """
        groups = [df for _, df in self.adata.df.groupby('sample_id')]
        if not randomize:
            cum = []
            union_set = set()
            for i, d in enumerate(groups, start=1):
                union_set.update(d['combined_custom_id'].dropna().unique())
                cum.append({'sample_num': i, 'unique_clones': len(union_set)})
            df_cum = pd.DataFrame(cum)
            p = (p9.ggplot(df_cum, p9.aes(x='sample_num', y='unique_clones')) +
                 p9.geom_col(fill="cornflowerblue") +
                 p9.geom_line(color="blue", group=1) +
                 p9.geom_point(color="blue", size=3) +
                 p9.labs(x="Cumulative Samples", y="Unique Clones",
                         title=f"[{self.adata.project_name}:{self.adata.data_subset}] Saturation Curve") +
                 professional_theme() +
                 p9.theme(panel_spacing=0.4))
            return p
        else:
            n = len(groups)
            all_cum = []
            for _ in range(n_iter):
                grp = groups.copy()
                np.random.shuffle(grp)
                union_set = set()
                cum = []
                for i, d in enumerate(grp, start=1):
                    union_set.update(d['combined_custom_id'].dropna().unique())
                    cum.append(len(union_set))
                all_cum.append(cum)
            arr = np.array(all_cum)
            mean_unique = np.mean(arr, axis=0)
            std_unique = np.std(arr, axis=0)
            df_plot = pd.DataFrame({
                'sample_num': np.arange(1, n+1),
                'mean_unique': mean_unique,
                'std_unique': std_unique
            })
            p = (p9.ggplot(df_plot, p9.aes(x='sample_num', y='mean_unique')) +
                 p9.geom_line(color="blue") +
                 p9.geom_point(color="blue", size=3) +
                 p9.geom_errorbar(p9.aes(ymin='mean_unique-std_unique', ymax='mean_unique+std_unique'), width=0.2) +
                 p9.labs(x="Cumulative Samples", y="Mean Unique Clones",
                         title=f"[{self.adata.project_name}:{self.adata.data_subset}] Saturation Curve (Randomized Order)") +
                 professional_theme() +
                 p9.theme(panel_spacing=0.4))
            return p

    def plot_clonal_expansion(self, bins=None, labels=None, normalize=False, label_order: list = None):
        """
        Plots clonal expansion as a stacked bar plot.
        The x-axis shows samples and the fill indicates expansion bins.
        When normalize=True, proportions are plotted.
        label_order is a list specifying the order in which sample IDs should appear on the x-axis.
        """
        if bins is None:
            bins = [0.5, 1.5, 2.5, 5.5, 20.5, np.inf]
        if labels is None:
            labels = ["1", "2", "3-5", "6-20", ">20"]
        if 'consensus_count' in self.adata.df.columns:
            clone_counts = self.adata.df.groupby(['sample_id', 'combined_custom_id'])['consensus_count'].sum().reset_index()
        else:
            clone_counts = self.adata.df.groupby(['sample_id', 'combined_custom_id']).size().reset_index(name='consensus_count')
        clone_counts['expansion_bin'] = pd.cut(clone_counts['consensus_count'],
                                                bins=bins, labels=labels, include_lowest=True)
        exp_df = clone_counts.groupby(['sample_id', 'expansion_bin']).size().reset_index(name='n_clones')
        if normalize:
            total = exp_df.groupby('sample_id')['n_clones'].transform('sum')
            exp_df['proportion'] = exp_df['n_clones'] / total
            y_val = 'proportion'
            ylabel = "Proportion of Clones"
        else:
            y_val = 'n_clones'
            ylabel = "Number of Clones"
        p = (p9.ggplot(exp_df, p9.aes(x='sample_id', y=y_val, fill='expansion_bin')) +
             p9.geom_col() +
             p9.labs(x="Sample", y=ylabel, fill="Clone Count",
                     title=f"[{self.adata.project_name}:{self.adata.data_subset}] Clonal Expansion") +
             professional_theme() +
             p9.theme(panel_spacing=0.4, axis_text_x=p9.element_text(rotation=45, ha="right")))
        if label_order is not None:
            p = p + p9.scale_x_discrete(limits=label_order)
        return p

    def plot_venn(self, group: str = None):
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
        # If a group is specified, restrict sample list accordingly; otherwise use all samples.
        if group and self.adata.obs is not None and "group" in self.adata.obs.columns:
            samples = self.adata.obs[self.adata.obs["group"] == group].index.tolist()
        else:
            samples = self.adata.samples

        # Use the counts matrix so that the venn numbers match get_counts_matrix()
        cm = self.adata.tl.get_counts_matrix()
        if len(samples) == 2:
            s1, s2 = samples[0], samples[1]
            set1 = set(cm.index[cm[s1] > 0]) if s1 in cm.columns else set()
            set2 = set(cm.index[cm[s2] > 0]) if s2 in cm.columns else set()
            plt.figure(figsize=(8, 6))
            v = venn2(subsets=(len(set1 - set2),
                               len(set2 - set1),
                               len(set1 & set2)),
                      set_labels=(s1, s2))
            venn2_circles(subsets=(len(set1 - set2),
                                   len(set2 - set1),
                                   len(set1 & set2)),
                          linestyle='dashed', linewidth=1, color="black")
            plt.title(f"[{self.adata.project_name}:{self.adata.data_subset}]", # Venn Diagram ({s1} vs {s2})",
                      fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

        elif len(samples) == 3:
            s1, s2, s3 = samples[0], samples[1], samples[2]
            set1 = set(cm.index[cm[s1] > 0]) if s1 in cm.columns else set()
            set2 = set(cm.index[cm[s2] > 0]) if s2 in cm.columns else set()
            set3 = set(cm.index[cm[s3] > 0]) if s3 in cm.columns else set()

            A_only = len(set1 - set2 - set3)
            B_only = len(set2 - set1 - set3)
            AB_only = len((set1 & set2) - set3)
            C_only = len(set3 - set1 - set2)
            AC_only = len((set1 & set3) - set2)
            BC_only = len((set2 & set3) - set1)
            ABC = len(set1 & set2 & set3)

            plt.figure(figsize=(8, 6))
            v = venn3(subsets=(A_only, B_only, AB_only, C_only, AC_only, BC_only, ABC),
                      set_labels=(s1, s2, s3))
            venn3_circles(subsets=(A_only, B_only, AB_only, C_only, AC_only, BC_only, ABC),
                          linestyle='dashed', linewidth=1, color="black")
            plt.title(f"[{self.adata.project_name}:{self.adata.data_subset}]", # Venn Diagram ({s1}, {s2}, {s3})",
                      fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

        else:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5,
                     f"Venn diagram supports only 2 or 3 sets. Provided {len(samples)} sets.",
                     horizontalalignment="center", verticalalignment="center",
                     fontsize=12, fontweight="bold")
            plt.title("Venn Diagram", fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()
        
    def plot_upset(self):
        """
        Creates an UpSet plot using the UpSetPlot library.
        Generates a plot showing the set intersections based on the counts matrix
        and returns the current matplotlib figure.
        """
        import matplotlib.pyplot as plt
        from upsetplot import from_indicators, plot as upset_plot

        # Retrieve the counts matrix and convert counts to boolean (presence/absence)
        cm = self.adata.tl.get_counts_matrix()
        pres = (cm > 0)
        # Create an UpSet series from the boolean indicators using the sample names as index
        upset_series = from_indicators(pres.columns.tolist(), pres)
        upset_series.name = "count"
        
        # Plot the upset plot onto the current figure
        upset_plot(upset_series)
        plt.suptitle(f"[{self.adata.project_name}:{self.adata.data_subset}] Upset Plot",
                     fontsize=14, fontweight="bold")
        plt.show()
        return plt.gcf()

    def plot_diversity_measures(self, label_order: list = None):
        """
        Creates a composite ggplot2 figure consisting of three side-by-side bar plots:
          • Left: Gini Unevenness (from compute_gini)
          • Center: Shannon Entropy (from compute_entropy)
          • Right: Simpson's Diversity (from compute_simpson)
        
        Each panel displays per-sample diversity measures.
        If label_order is provided, it defines the order of sample IDs on the x-axis.
        """
        # Compute the three diversity measures
        gini_series = self.adata.tl.compute_gini()
        entropy_series = self.adata.tl.compute_entropy()
        simpson_series = self.adata.tl.compute_simpson()

        # Convert the series to DataFrames for ggplot
        df_gini = pd.DataFrame({
            'sample_id': gini_series.index,
            'value': gini_series.values
        })
        df_entropy = pd.DataFrame({
            'sample_id': entropy_series.index,
            'value': entropy_series.values
        })
        df_simpson = pd.DataFrame({
            'sample_id': simpson_series.index,
            'value': simpson_series.values
        })

        # Create a ggplot bar plot for Gini Unevenness
        p_gini = (p9.ggplot(df_gini, p9.aes(x='sample_id', y='value')) +
                  p9.geom_col(fill="salmon") +
                  p9.labs(title=f"[{self.adata.project_name}:{self.adata.data_subset}] Gini Unevenness",
                          x="Sample", y="Gini") +
                  professional_theme() +
                  p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
                 )
        if label_order is not None:
            p_gini = p_gini + p9.scale_x_discrete(limits=label_order)
        
        # Create a ggplot bar plot for Shannon Entropy
        p_entropy = (p9.ggplot(df_entropy, p9.aes(x='sample_id', y='value')) +
                     p9.geom_col(fill="skyblue") +
                     p9.labs(title=f"[{self.adata.project_name}:{self.adata.data_subset}] Shannon Entropy",
                             x="Sample", y="Entropy") +
                     professional_theme() +
                     p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
                    )
        if label_order is not None:
            p_entropy = p_entropy + p9.scale_x_discrete(limits=label_order)
        
        # Create a ggplot bar plot for Simpson's Diversity
        p_simpson = (p9.ggplot(df_simpson, p9.aes(x='sample_id', y='value')) +
                     p9.geom_col(fill="lightgreen") +
                     p9.labs(title=f"[{self.adata.project_name}:{self.adata.data_subset}] Simpson's Diversity (1 - Σp²)",
                             x="Sample", y="Simpson's Diversity") +
                     professional_theme() +
                     p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
                    )
        if label_order is not None:
            p_simpson = p_simpson + p9.scale_x_discrete(limits=label_order)
        
        # Combine the three plots into one panel using cowpatch if available.
        try:
            import cowpatch as cow
            import numpy as np
            # Create a patch with three plots in one row (design: 1 row, 3 columns)
            vis_patch = cow.patch(p_gini, p_entropy, p_simpson)
            vis_patch += cow.layout(design=np.array([0, 1, 2]).reshape(1, 3))
            final_plot = vis_patch
        except ImportError:
            final_plot = (p_gini, p_entropy, p_simpson)
        print(final_plot.show())
