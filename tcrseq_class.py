import os, re, sys, hashlib
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import plotnine as p9
from scipy.stats import entropy
try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    venn2, venn3 = None, None
from upsetplot import from_indicators, plot as upset_plot

try:
    import cowpatch as cp  # Use cp.ggplot_grid to merge plots.
except ImportError:
    cp = None


try:
    from mizani.unit import unit
except ImportError:
    def unit(value, unit_type):
        # Fallback: simply return a string that ggplot2 might interpret.
        return f"{value} {unit_type}"


# ----------------------
# TCRSeq Class
# ----------------------
class TCRSeq:
    def __init__(self, df: pd.DataFrame, obs: pd.DataFrame = None, data_name: str = "all"):
        self.df = df.copy()   # main data (clone-level)
        # If obs is not provided, create an empty DataFrame with index = unique sample_id
        self.obs = obs.copy() if obs is not None else pd.DataFrame(index=self.samples)
        self.data_name = data_name  # new attribute for dataset naming
        # Create namespace-like attributes for preprocessing (pp), tools (tl), and plotting (pl)
        self.pp = _TCRSeqPreprocessing(self)
        self.tl = _TCRSeqTools(self)
        self.pl = _TCRSeqPlotting(self)
    
    def __repr__(self):
        lines = []
        lines.append(f"<TCRSeq object at {hex(id(self))}>")
        lines.append(f"data_name: {self.data_name}")
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

    # --- New Utility Methods ---

    def get_stats(self) -> pd.DataFrame:
        """
        Get per-sample reading statistics.
        Also update the self.obs metadata by joining in these stats.
        """
        stats_df = self.tl.get_sample_stats()
        if not self.obs.empty:
            # Join stats on index (sample_id)
            stats_df = stats_df.set_index('sample_id')
            self.obs = self.obs.join(stats_df, how="left")
            stats_df = self.obs.reset_index()
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

    # ==== File reading and conversion ====
    @classmethod
    def from_file(cls, filename: str, format_type: str = 'airr', barcode_suffix: str = None,
                  assay_type: str = 'single_cell', filter_productive: bool = True):
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
        # For a single-sample load, data_name is set to the sample_id.
        return cls(res, data_name=sid)
    
    @classmethod
    def from_sample_info(cls, sample_info: pd.DataFrame, filename_col: str = 'airr_filename',
                         sample_id_col: str = 'sample_id', **kwargs):
        dfs = []
        stats_list = []
        
        for i, row in sample_info.iterrows():
            sample_id = row[sample_id_col]
            filename = row[filename_col]
            print(f"Loading sample {sample_id} from file {filename} ...", file=sys.stderr)
            tcr_sample = cls.from_file(filename, **kwargs)
            # Override sample_id using sample_info.
            tcr_sample.df['sample_id'] = sample_id
            stats = tcr_sample._get_reading_stats()
            stats_list.append(stats)
            dfs.append(tcr_sample.df)
        combined_df = pd.concat(dfs, ignore_index=True)
        obs = sample_info.set_index(sample_id_col)
        if 'group' in obs.columns:  obs['group'] = obs['group'].astype('category')
        
        print(f"Loaded {len(dfs)} samples with a total of {combined_df.shape[0]} rows.", file=sys.stderr)
        stats_df = pd.DataFrame(stats_list)
        # For multiple samples, data_name is set to "all".
        return cls(combined_df, obs, data_name="all"), stats_df

    def _get_reading_stats(self) -> dict:
        n_total = self.df.shape[0]
        n_prod = self.df['productive'].sum() if 'productive' in self.df.columns else n_total
        n_clones_all = self.df['chain_clonotype_id'].nunique()
        n_clones_prod = self.df[self.df['productive'] == True]['chain_clonotype_id'].nunique() if 'productive' in self.df.columns else n_clones_all
        
        n_TRA = self.df[self.df['TRA_custom_id'] != '']['TRA_custom_id'].nunique() if 'TRA_custom_id' in self.df.columns else 0
        n_TRB = self.df[self.df['TRB_custom_id'] != '']['TRB_custom_id'].nunique() if 'TRB_custom_id' in self.df.columns else 0
        pct_TRB=float(n_TRB/n_clones_prod)
        sample_id = self.df['sample_id'].unique()[0]
        shannon_diversity = self.tl.compute_entropy().get(sample_id, np.nan)
        gini_unevenness = self.tl.compute_gini().get(sample_id, np.nan)
        if 'consensus_count' in self.df.columns:
            clone_counts = self.df.groupby('combined_custom_id')['consensus_count'].sum()
            pct_clones_gt1 = (clone_counts > 1).sum() / n_clones_all * 100 if n_clones_all > 0 else np.nan
        else:
            pct_clones_gt1 = np.nan
        return {
            'sample_id': sample_id,
            #'n_total': n_total,
            #'n_prod': n_prod,
            'pct_TRB': pct_TRB,
            'n_clones_all': n_total, #n_clones_all,
            #'n_clones_prod': n_prod, #n_clones_prod,
            'n_TRA': n_TRA,
            'n_TRB': n_TRB,
            'shannon_diversity': shannon_diversity,
            'gini_unevenness': gini_unevenness,
            'pct_clones_gt1': pct_clones_gt1,
        }

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
        # data_name remains unchanged when filtering
        new_obs = (self.adata.obs.loc[self.adata.obs.index.isin(new_df['sample_id'].unique())]
                   if self.adata.obs is not None and not self.adata.obs.empty
                   else self.adata.obs)
        return TCRSeq(new_df, new_obs, data_name=self.adata.data_name)
    
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
        return TCRSeq(new_df, new_obs, data_name=self.adata.data_name)
    
    def subset_by_group(self, group_value: str) -> TCRSeq:
        if self.adata.obs is None or 'group' not in self.adata.obs.columns:
            raise ValueError("Metadata is missing a 'group' column.")
        samples = self.adata.obs[self.adata.obs['group'] == group_value].index.tolist()
        sub_df = self.adata.df[self.adata.df['sample_id'].isin(samples)].copy()
        new_obs = self.adata.obs.loc[samples].copy() if not self.adata.obs.empty else self.adata.obs
        new_data_name = f"{self.adata.data_name}_group:{group_value}"
        return TCRSeq(sub_df, new_obs, data_name=new_data_name)
    
    def subset_by_chain(self, chain: str) -> TCRSeq:
        valid = {"TRA": ["TRA", "TRAC"], "TRB": ["TRB", "TRBC"]}
        if chain not in valid:
            raise ValueError("chain must be either 'TRA' or 'TRB'")
        subset_df = self.adata.df[self.adata.df['locus'].isin(valid[chain])].copy()
        sample_ids = subset_df['sample_id'].unique()
        new_obs = (self.adata.obs.loc[self.adata.obs.index.isin(sample_ids)].copy() 
                   if self.adata.obs is not None and not self.adata.obs.empty 
                   else self.adata.obs)
        new_data_name = f"{self.adata.data_name}_chain:{chain}"
        return TCRSeq(subset_df, new_obs, data_name=new_data_name)

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
    
    def most_common_clones(self, top_n: int = 5) -> dict:
        cm = self.get_counts_matrix()
        common = {}
        for s in cm.columns:
            common[s] = cm[s].sort_values(ascending=False).head(top_n)
        return common
    
    def get_sample_stats(self) -> pd.DataFrame:
        stats_list = []
        for sample, sub in self.adata.df.groupby('sample_id'):
            temp_obj = TCRSeq(sub, self.adata.obs.loc[[sample]] if (self.adata.obs is not None and sample in self.adata.obs.index) else None)
            stats_list.append(temp_obj._get_reading_stats())
        return pd.DataFrame(stats_list)

# ----------------------------
# Plotting Namespace (pl)
# ----------------------------
def professional_theme():
    return p9.theme(
        panel_background=p9.element_rect(fill="white"),
        panel_grid_major=p9.element_line(linetype='dotted', color='grey', size=0.2),
        panel_grid_minor=p9.element_line(linetype='dotted', color='grey', size=0.5),
        panel_border=p9.element_rect(color='grey', fill=None),
        legend_title=p9.element_text(weight="bold"),
        legend_direction="horizontal",
        legend_text=p9.element_text(size=8),
        axis_text_x=p9.element_text(rotation=45, hjust=1, size=6),
        axis_text_y=p9.element_text(size=8),
        legend_position='top',
        strip_text=p9.element_text(size=6)
    )


class _TCRSeqPlotting:
    def __init__(self, adata: TCRSeq):
        self.adata = adata

    def plot_sample_stats(self, split_by: str = None, ncol: int = None, nrow: int = None, FILL: str = 'sample_id'):
        """
        Bar plots for per-sample reading statistics.
        Each metric (column) in the stats is shown in its own facet.
        If split_by is provided (the name of a column in obs), it is added to the facet formula.
        ncol and nrow allow you to specify the facet layout.
        FILL is the column name used for the bar fill color.
        """
        stats_df = self.adata.tl.get_sample_stats()
        # If split_by is provided and exists in obs, merge that column into stats_df.
        if split_by is not None and self.adata.obs is not None and split_by in self.adata.obs.columns:
            obs_reset = self.adata.obs.reset_index()
            stats_df = stats_df.merge(obs_reset[['sample_id', split_by]], on='sample_id', how='left')
            id_vars = ["sample_id", split_by]
            facet_formula = "~ metric + " + split_by
        else:
            id_vars = ["sample_id"]
            facet_formula = "~ metric"

        # If you want the fill column from obs, merge that column if missing.
        if FILL != "sample_id" and (split_by is None or FILL not in stats_df.columns):
            if self.adata.obs is not None and FILL in self.adata.obs.columns:
                obs_reset = self.adata.obs.reset_index()
                stats_df = stats_df.merge(obs_reset[['sample_id', FILL]], on='sample_id', how='left')
                if FILL not in id_vars:
                    id_vars.append(FILL)

        # Convert the fill column to a categorical variable for a discrete color scale.
        if FILL in stats_df.columns:
            stats_df[FILL] = stats_df[FILL].astype('category')

        stats_melt = stats_df.melt(id_vars=id_vars, var_name="metric", value_name="value")

        # Ensure that the fill aesthetic is set to a column that exists.
        fill_mapping = FILL if FILL in stats_melt.columns else "sample_id"

        # If data for only TRB exist, force the n_TRA facet to have the same y scale as n_TRB.
        if 'n_TRA' in stats_melt['metric'].unique() and 'n_TRB' in stats_melt['metric'].unique():
            max_trb = stats_melt.loc[stats_melt['metric'] == 'n_TRB', 'value'].max()
            max_tra = stats_melt.loc[stats_melt['metric'] == 'n_TRA', 'value'].max()
            if max_trb > max_tra:
                # Append a dummy row to "n_TRA" so that its y-scale reaches max_trb.
                dummy = pd.DataFrame({
                    'sample_id': [stats_melt['sample_id'].iloc[0]],
                    'metric': ['n_TRA'],
                    'value': [max_trb],
                })
                if split_by is not None:
                    dummy[split_by] = stats_melt[split_by].iloc[0]
                if FILL in stats_melt.columns:
                    dummy[FILL] = stats_melt[FILL].iloc[0]
                stats_melt = pd.concat([stats_melt, dummy], ignore_index=True)

        base_plot = (
            p9.ggplot(stats_melt, p9.aes(x="sample_id", y="value", fill=fill_mapping))
            + p9.geom_bar(stat="identity")
            + p9.labs(
                title=f"[{self.adata.data_name}] Per-Sample Reading Stats",
                x="Sample ID",
                y="Value"
            )
            + p9.theme(
                axis_text_x=p9.element_text(rotation=90, hjust=1, size=4),
                panel_spacing=0.5, figure_size=(26, 3)
            )
            + p9.facet_wrap(facet_formula, scales="free_y", ncol=ncol, nrow=nrow)
            + professional_theme()
        )
        return base_plot


    def plot_upset(self):
        cm = self.adata.tl.get_counts_matrix()
        pres = cm > 0
        upset_data = from_indicators(pres.columns.tolist(), pres)
        fig = upset_plot(upset_data)
        plt.title(f"[{self.adata.data_name}] Upset Plot")
        plt.tight_layout()
        plt.show()
        return fig

    def plot_venn(self, group: str = None):
        if venn2 is None:
            raise ImportError("matplotlib-venn required")
        if group and self.adata.obs is not None and 'group' in self.adata.obs.columns:
            samples = self.adata.obs[self.adata.obs['group'] == group].index.tolist()
        else:
            samples = self.adata.samples
        if len(samples) == 2:
            sets = [set(self.adata.df[self.adata.df['sample_id'] == s]['combined_custom_id'])
                    for s in samples]
            plt.figure(); venn2(sets, set_labels=samples)
            plt.title(f"[{self.adata.data_name}] Venn Diagram")
            plt.show()
        elif len(samples) == 3:
            sets = [set(self.adata.df[self.adata.df['sample_id'] == s]['combined_custom_id'])
                    for s in samples]
            plt.figure(); venn3(sets, set_labels=samples)
            plt.title(f"[{self.adata.data_name}] Venn Diagram")
            plt.show()
        else:
            print("Venn supports only 2 or 3 sets.")

    def plot_pie_common_clones(self, top_n: int = 5):
        # Use matplotlib to create a pie chart.
        cm = self.adata.tl.get_counts_matrix().sum(axis=1)
        top = cm.sort_values(ascending=False).head(top_n)
        others = cm.sum() - top.sum()
        # Pandas append vs concat compatibility:
        data = top.append(pd.Series({'Others': others})) if hasattr(top, 'append') else pd.concat([top, pd.Series({'Others': others})])
        fig, ax = plt.subplots()
        ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(f"[{self.adata.data_name}] Top {top_n} Clones")
        plt.show()
        return fig

    def plot_shared_clones_dot(self, split_by: str = None, n_top: int = 10):
        """
        Scanpy-style dot plot.
        y-axis: combined_custom_id
        x-axis: sample_id
        dot size: clone count
        Only plot the top n_top clones (by average count across all samples).
        Optionally facet by split_by if provided.
        """
        if 'consensus_count' in self.adata.df.columns:
            agg = self.adata.df.groupby(['sample_id', 'combined_custom_id']).agg(
                clone_count=('consensus_count', 'sum')
            ).reset_index()
        else:
            agg = self.adata.df.groupby(['sample_id', 'combined_custom_id']).size().reset_index(name='clone_count')
        # Compute the average clone count (across samples) for each clone.
        avg_df = agg.groupby('combined_custom_id')['clone_count'].mean().reset_index()
        top_clones = avg_df.sort_values(by='clone_count', ascending=False).head(n_top)['combined_custom_id']
        agg = agg[agg['combined_custom_id'].isin(top_clones)]
        facet = None
        if split_by is not None and self.adata.obs is not None and split_by in self.adata.obs.columns:
            obs_reset = self.adata.obs.reset_index()
            agg = agg.merge(obs_reset[['sample_id', split_by]], on='sample_id', how='left')
            facet = f"~ {split_by}"
        
        p = (p9.ggplot(agg, p9.aes(x='sample_id', y='combined_custom_id', size='clone_count'))
             + p9.geom_point()
             + p9.labs(title=f"[{self.adata.data_name}] Dot Plot (Top {n_top} Clones)", 
                       x="Sample", y="Clone", size="Clone Count")
             + professional_theme()
        )
        if facet:
            p = p + p9.facet_wrap(facet, scales="free_y")
        return p

    def plot_clone_bar_by_sample(self, split_by: str = None):
        """
        Bar plot of unique TRA and TRB clone counts per sample.
        If split_by is provided, facet the plot accordingly.
        """
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
        if self.adata.obs is not None:
            counts = counts.merge(self.adata.obs.reset_index(), on='sample_id', how='left')
        base_plot = (p9.ggplot(counts, p9.aes(x='sample_id', y='clone_count', fill='chain'))
                     + p9.geom_bar(stat='identity', position='dodge')
                     + p9.labs(title=f"[{self.adata.data_name}] Unique Clone Counts per Sample",
                               x="Sample", y="Clone Count")
                     + professional_theme()
                     + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
                    )
        if split_by is not None and self.adata.obs is not None and split_by in self.adata.obs.columns:
            base_plot = base_plot + p9.facet_wrap(f"~ {split_by}")
            if cp is not None:
                plots = []
                for grp, sub_df in counts.groupby(split_by):
                    pg = (p9.ggplot(sub_df, p9.aes(x='sample_id', y='clone_count', fill='chain'))
                          + p9.geom_bar(stat='identity', position='dodge')
                          + p9.labs(title=f"{split_by}: {grp}", x="Sample", y="Clone Count")
                          + professional_theme()
                          + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
                         )
                    plots.append(pg)
                base_plot = cp.ggplot_grid(plots, ncol=2)
        return base_plot

    def plot_saturation_curve(self, randomize: bool = False, n_iter: int = 100):
        groups = [df for _, df in self.adata.df.groupby('sample_id')]
        if not randomize:
            cum = []
            union_set = set()
            for i, d in enumerate(groups, start=1):
                union_set.update(d['combined_custom_id'].dropna().unique())
                cum.append({'sample_num': i, 'unique_clones': len(union_set)})
            df_cum = pd.DataFrame(cum)
            p = (p9.ggplot(df_cum, p9.aes(x='sample_num', y='unique_clones'))
                 + p9.geom_bar(stat='identity', fill='cornflowerblue', width=0.7)
                 + p9.geom_line(group=1, color='blue')
                 + p9.geom_point(color='blue', size=3)
                 + p9.labs(title=f"[{self.adata.data_name}] Saturation Curve", 
                           x='Cumulative Samples', y='Unique Clones')
                 + professional_theme()
                 + p9.theme(axis_text_x=p9.element_text(rotation=25, hjust=1))
                )
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
            df_plot = pd.DataFrame({'sample_num': np.arange(1, n+1),
                                    'mean_unique': mean_unique,
                                    'std_unique': std_unique})
            p = (p9.ggplot(df_plot, p9.aes(x='sample_num', y='mean_unique'))
                 + p9.geom_line(color='blue')
                 + p9.geom_point(color='darkblue', size=3)
                 + p9.geom_errorbar(
                        p9.aes(ymin='mean_unique - std_unique',
                               ymax='mean_unique + std_unique'),
                        width=0.2)
                 + p9.labs(title=f"[{self.adata.data_name}] Saturation Curve (Randomized Order)",
                           x='Cumulative Samples', y='Mean Unique Clones')
                 + professional_theme()
                 + p9.theme(axis_text_x=p9.element_text(rotation=25, hjust=1))
                )
            return p

    def plot_clonal_expansion(self, bins=None, labels=None, normalize=False):
        """
        Creates a stacked bar plot showing the distribution of clone counts across samples.
        Default bins represent: 1, 2, 3-5, 6-20, >20 clones. The user may supply custom bins and labels.
        If normalize is True, proportions per sample are shown.
        """
        if bins is None:
            bins = [0.5, 1.5, 2.5, 5.5, 20.5, np.inf]
        if labels is None:
            labels = ["1", "2", "3-5", "6-20", ">20"]
        if 'consensus_count' in self.adata.df.columns:
            clone_counts = self.adata.df.groupby(['sample_id', 'combined_custom_id'])['consensus_count'].sum().reset_index()
        else:
            clone_counts = self.adata.df.groupby(['sample_id', 'combined_custom_id']).size().reset_index(name='consensus_count')
        clone_counts['expansion_bin'] = pd.cut(clone_counts['consensus_count'], bins=bins, labels=labels, include_lowest=True)
        exp_df = clone_counts.groupby(['sample_id', 'expansion_bin']).size().reset_index(name='n_clones')
        if normalize:
            total = exp_df.groupby('sample_id')['n_clones'].transform('sum')
            exp_df['proportion'] = exp_df['n_clones'] / total
            y_val = 'proportion'
            ylabel = "Proportion of Clones"
        else:
            y_val = 'n_clones'
            ylabel = "Number of Clones"
        p = (p9.ggplot(exp_df, p9.aes(x='sample_id', y=y_val, fill='expansion_bin'))
             + p9.geom_bar(stat='identity', position='stack')
             + p9.labs(title=f"[{self.adata.data_name}] Clonal Expansion", x="Sample", y=ylabel, fill="Clone Count")
             + professional_theme()
             + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
        )
        return p
