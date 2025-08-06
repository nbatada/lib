# Updated : 6 August 2025, 3.45pm
import os
import re
import sys
import hashlib
import copy
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Union, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotnine as p9
from scipy.stats import entropy
from upsetplot import from_indicators, plot as upset_plot
import logging
logger = logging.getLogger(__name__)

try:
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
except ImportError:
    venn2 = venn3 = venn2_circles = venn3_circles = None

try:
    import cowpatch as cp
except ImportError:
    cp = None


import plotnine as p9
from plotnine import ggplot, aes, geom_point, theme, element_text, element_rect, element_blank, element_line, theme_minimal,theme_light

def economist_theme() -> p9.theme:
    return (
        p9.theme_matplotlib() +
        p9.theme(
            axis_text_x=p9.element_text(rotation=20, hjust=1, size=9),
            axis_text_y=p9.element_text(size=9),
            plot_title=p9.element_text(text="", margin={"t": 0, "r": 0, "b": 0, "l": 0}),
            legend_title=p9.element_text(weight="bold"),
            subplots_adjust={'wspace': 0.25}  # adds extra horizontal spacing between facet panels
        )
    )



@dataclass
class SampleStats:
    """Summary statistics for a single sample."""
    sample_id: str
    n_total_clones: int
    n_unique_clones: int
    pct_TRA: float
    pct_TRB: float
    pct_clones_gt1: float


def _clone_breakdown(df: pd.DataFrame, sample_id: str, step: str) -> Dict[str, Any]:
    """Internal: capture clone counts at a given processing step."""
    sub = df[df['sample_id'] == sample_id]
    return {
        'sample_id': sample_id,
        'step': step,
        'total_rows': len(sub),
        'unique_clones': sub['combined_custom_id'].nunique(),
        'TRA_clones': sub.loc[sub['TRA_custom_id'] != '', 'combined_custom_id'].nunique(),
        'TRB_clones': sub.loc[sub['TRB_custom_id'] != '', 'combined_custom_id'].nunique()
    }


def unit(value: Union[int, float], unit_type: str) -> str:
    """Attach a unit to a numeric value."""
    return f"{value} {unit_type}"


class _TCRSeqAccessor:
    """Accessor for getting/setting core data and derived metrics."""
    def __init__(self, adata: "TCRSeq"):
        self._adata = adata

    def get_clonotypes_df(self) -> pd.DataFrame:
        return self._adata.clonotypes_df

    def set_clonotypes_df(self, clonotypes_df: pd.DataFrame) -> None:
        self._adata.clonotypes_df = clonotypes_df.copy()
        self._adata._clear_caches()
        self._adata._record_step("manual update")

    def get_obs(self) -> Optional[pd.DataFrame]:
        return self._adata.obs

    def set_obs(self, obs: Optional[pd.DataFrame]) -> None:
        self._adata.obs = obs.copy() if obs is not None else None

    def get_project_name(self) -> str:
        return self._adata.project_name

    def set_project_name(self, name: str) -> None:
        self._adata.project_name = name

    def get_counts_matrix(self) -> pd.DataFrame:
        return self._adata.tl.get_counts_matrix()

    def get_sample_stats(self) -> pd.DataFrame:
        return self._adata.get_stats()

    def compute_entropy(self) -> pd.Series:
        return self._adata.entropy

    def compute_gini(self) -> pd.Series:
        return self._adata.gini

    def compute_simpson(self) -> pd.Series:
        return self._adata.simpson

    def get_shared_clones(self, n_top: int = 10) -> pd.DataFrame:
        return self._adata.tl.get_shared_clones(n_top)


class TCRSeq:
    """
    Core class for TCR sequencing data:
      - clonotypes_df: per-cell or per-read clonotype assignments
      - obs: optional sample metadata (with QC metrics joined in)
      - .pp, .tl, .pl, .acc: preprocessing, tools, plotting, accessor namespaces
    """
    def __init__(
        self,
        clonotypes_df: pd.DataFrame,
        obs: Optional[pd.DataFrame] = None,
        data_subset: str = "all",
        project_name: str = "DefaultProject"
    ) -> None:

        required = {"sample_id", "combined_custom_id"}
        missing = required - set(clonotypes_df.columns)
        if missing:
            raise ValueError(f"TCRSeq initialization error: Missing required columns: {missing}")

        self.clonotypes_df = clonotypes_df.copy()
        self.obs = obs.copy() if obs is not None else None
        self.data_subset = data_subset
        self.project_name = project_name

        # cached computations
        self._cm: Optional[pd.DataFrame] = None
        self._entropy_cache: Optional[pd.Series] = None
        self._gini_cache: Optional[pd.Series] = None
        self._simpson_cache: Optional[pd.Series] = None

        # history of filtering steps
        self._history: Dict[str, List[Dict[str, Any]]] = {
            sid: [ _clone_breakdown(self.clonotypes_df, sid, 'raw') ]
            for sid in sorted(self.clonotypes_df['sample_id'].unique())
        }

        # Namespaces
        self.pp = _TCRSeqPreprocessing(self)
        self.tl = _TCRSeqTools(self)
        self.pl = _TCRSeqPlotting(self)
        self.acc = _TCRSeqAccessor(self)

    def __repr__(self) -> str:
        obs_info = (
            ["", "[.obs] -- sample metadata",
             "-" * 46,
             f"\tshape: {self.obs.shape}",
             f"\tcolumns: {', '.join(self.obs.columns)}"]
            if self.obs is not None and not self.obs.empty else []
        )
        lines = [
            f"<TCRSeq object at {hex(id(self))}>",
            f"project_name: {self.project_name}",
            f"data_subset: {self.data_subset}",
            "[.clonotypes_df] -- raw clonotype data",
            f"\tshape: {self.clonotypes_df.shape}",
            f"\tUnique samples: {len(self.samples)}",
            f"\tUnique clones: {len(self.clones)}",
            f"\tcolumns: {', '.join(self.clonotypes_df.columns)}",
            "",
            "[.X] -- counts matrix (clones × samples)",
            f"\tshape: {self.X.shape}",
            f"\tcolumns: {', '.join(self.X.columns)}"
        ] + obs_info
        return "\n".join(lines)

    @property
    def samples(self) -> List[str]:
        return sorted(self.clonotypes_df['sample_id'].unique())

    @property
    def clones(self) -> np.ndarray:
        return self.clonotypes_df['combined_custom_id'].unique()

    @property
    def X(self) -> pd.DataFrame:
        return self.acc.get_counts_matrix()

    @property
    def entropy(self) -> pd.Series:
        if self._entropy_cache is None:
            self._entropy_cache = self.tl.compute_entropy()
        return self._entropy_cache

    @property
    def gini(self) -> pd.Series:
        if self._gini_cache is None:
            self._gini_cache = self.tl.compute_gini()
        return self._gini_cache

    @property
    def simpson(self) -> pd.Series:
        if self._simpson_cache is None:
            self._simpson_cache = self.tl.compute_simpson()
        return self._simpson_cache

    def _clear_caches(self):
        """Invalidate all derived caches."""
        self._cm = self._entropy_cache = self._gini_cache = self._simpson_cache = None

    def _record_step(self, label: str):
        """Append a new breakdown entry for every sample."""
        for sid in self._history:
            self._history[sid].append(
                _clone_breakdown(self.clonotypes_df, sid, label)
            )

    def get_stats(self) -> pd.DataFrame:
        """
        Merge per-sample reading stats with self.obs (if present).
        """
        stats = self.tl.get_sample_stats()
        if self.obs is not None and not self.obs.empty:
            obs = self.obs.reset_index()
            # Drop any overlapping columns (other than 'sample_id') so that merge doesn't create duplicate columns.
            duplicate_cols = set(stats.columns).intersection(obs.columns) - {"sample_id"}
            if duplicate_cols:
                obs = obs.drop(columns=list(duplicate_cols))
            return stats.merge(obs, on="sample_id", how="right")
        return stats

    def get_top_clones(self, sample_ids: List[str], n: int = 10) -> pd.DataFrame:
        """
        Return the top-n clones (by count) for each sample in sample_ids.
        """
        missing = set(sample_ids) - set(self.X.columns)
        if missing:
            raise ValueError(f"Samples not found: {missing}")
        mat = self.X[sample_ids]
        top_clones = set()
        for s in sample_ids:
            top_clones.update(mat[s].nlargest(n).index)
        return mat.loc[list(top_clones), sample_ids]

    def get_clone_count(self, clone_ids: List[str]) -> Dict[str, int]:
        """
        Sum counts across all samples for each requested clone_id.
        """
        summed = self.X.sum(axis=1).reindex(clone_ids, fill_value=0)
        return summed.astype(int).to_dict()

    @classmethod
    def from_file(
        cls,
        filename: str,
        format_type: str = 'airr',
        barcode_suffix: Optional[str] = None,
        assay_type: str = 'single_cell',
        filter_productive: bool = True,
        filter_IG: bool = True,
        filter_TRD_TRG: bool = True,
        project_name: str = "DefaultProject"
    ) -> "TCRSeq":
        """
        Load a single AIRR or CellRanger file, filter loci & productive,
        build clonotypes, and return a TCRSeq.
        """
        # --- read and standardize columns ---
        if format_type == 'cellranger':
            df = pd.read_csv(filename)
            df.rename(columns={
                'chain':'locus','v_gene':'v_call','d_gene':'d_call',
                'j_gene':'j_call','cdr3':'junction_aa'
            }, inplace=True)
        else:
            df = pd.read_csv(filename, sep='\t', header=0)

        # --- ensure barcode exists ---
        if 'barcode' not in df.columns:
            if format_type == 'airr':
                df['barcode'] = barcode_suffix or os.path.splitext(os.path.basename(filename))[0]
            else:
                raise ValueError("Missing 'barcode' column")
        if barcode_suffix:
            df['barcode'] = df['barcode'].astype(str) + '_' + barcode_suffix

        # --- productive filter ---
        if 'productive' not in df.columns:
            df['productive'] = True
        else:
            df['productive'] = (
                df['productive']
                  .astype(str)
                  .str.upper()
                  .map({'T':True,'TRUE':True,'F':False,'FALSE':False})
                  .fillna(False)
            )

        # --- filter out non-TCR ---
        if filter_IG and 'locus' in df.columns:
            df = df[~df['locus'].isin(['IGH','IGL'])]
        if filter_TRD_TRG and 'locus' in df.columns:
            df = df[~df['locus'].isin(['TRD','TRG'])]

        # log how many were dropped
        if filter_IG or filter_TRD_TRG:
            logger.debug(f"After filtering loci: {df.shape[0]} rows remain in {filename}")

        # --- build chain_clonotype_id ---
        df['chain_clonotype_id'] = (
            df['v_call'].fillna('').str.upper() + '_' +
            df['junction_aa'].fillna('').str.upper() + '_' +
            df['j_call'].fillna('').str.upper()
        )

        # --- dispatch to single-cell vs bulk builder ---
        if assay_type == 'single_cell':
            clonotypes = cls._build_clonotypes_single_cell(df, filter_productive)
        else:
            clonotypes = cls._build_clonotypes_bulk(df, filter_productive)

        # --- MD5 hashes for interoperability ---
        def _md5(x: str) -> str:
            return hashlib.md5(x.encode()).hexdigest() if x else ''
        for col in ['TRA_custom_id','TRB_custom_id','combined_custom_id']:
            clonotypes[f"{col.split('_')[0].lower()}_univ_md5"] = clonotypes[col].apply(_md5)

        # --- extract sample_id from filename ---
        m = re.search(r'(P\d{2}E\d{2}S\d{2})', filename)
        sid = m.group(1) if m else os.path.splitext(os.path.basename(filename))[0]
        clonotypes['sample_id'] = sid

        return cls(clonotypes, data_subset=sid, project_name=project_name)

    @staticmethod
    def _build_clonotypes_single_cell(df: pd.DataFrame, productive_only: bool) -> pd.DataFrame:
        """Group by barcode, build TRA/TRB, categorize into paired/orphan."""
        proc = df[df['productive']] if productive_only else df

        def _join_by_locus(series: pd.Series, loci: List[str]) -> str:
            mask = proc.loc[series.index, 'locus'].isin(loci)
            return ','.join(series[mask])

        grouped = (
            proc
            .groupby('barcode')
            .agg(
                TRA_custom_id=('chain_clonotype_id', lambda s: _join_by_locus(s, ['TRA','TRAC'])),
                TRB_custom_id=('chain_clonotype_id', lambda s: _join_by_locus(s, ['TRB','TRBC'])),
                n_tra=('locus',  lambda s: s.isin(['TRA','TRAC']).sum()),
                n_trb=('locus',  lambda s: s.isin(['TRB','TRBC']).sum())
            )
            .reset_index()
        )

        def _categorize(r: pd.Series) -> pd.Series:
            if r['n_tra']==1 and r['n_trb']==1:
                return pd.Series(['paired', f"{r['TRA_custom_id']};{r['TRB_custom_id']}"],
                                 index=['category','combined_custom_id'])
            if r['n_tra']==1:
                return pd.Series(['orphan_TRA', r['TRA_custom_id']],
                                 index=['category','combined_custom_id'])
            if r['n_trb']==1:
                return pd.Series(['orphan_TRB', r['TRB_custom_id']],
                                 index=['category','combined_custom_id'])
            return pd.Series(['unknown',''], index=['category','combined_custom_id'])

        cats = grouped.apply(_categorize, axis=1)
        return pd.concat([grouped, cats], axis=1)[
            ['barcode','TRA_custom_id','TRB_custom_id','combined_custom_id','category']
        ]

    @staticmethod
    def _build_clonotypes_bulk(df: pd.DataFrame, productive_only: bool) -> pd.DataFrame:
        """Annotate each row with chain_clonotype_id."""
        proc = df[df['productive']].copy() if productive_only else df.copy()
        proc.loc[:,'TRA_custom_id'] = proc['chain_clonotype_id'].where(
            proc['locus'].isin(['TRA','TRAC']), ''
        )
        proc.loc[:,'TRB_custom_id'] = proc['chain_clonotype_id'].where(
            proc['locus'].isin(['TRB','TRBC']), ''
        )
        proc['combined_custom_id'] = proc['chain_clonotype_id']
        return proc.copy()

    @classmethod
    def from_sample_info(
        cls,
        sample_info: pd.DataFrame,
        filename_col: str = 'airr_filename',
        sample_id_col: str = 'sample_id',
        project_name: str = "DefaultProject",
        **kwargs
    ) -> "TCRSeq":
        """
        Load multiple files described in sample_info (with columns filename_col, sample_id_col),
        concatenate into one TCRSeq, and join per-sample QC metrics into obs.
        
        This updated version immediately sets the sample_id in the clonotypes_df
        (based on the sample_info provided) BEFORE computing the sample statistics.
        This ensures that the computed values (n_total_clones, n_unique_clones, etc.)
        use the intended sample id, so that later merging into .obs is successful.
        """
        dfs = []
        stats_list: List[SampleStats] = []

        for _, row in sample_info.iterrows():
            tcr = cls.from_file(row[filename_col], project_name=project_name, **kwargs)
            # Override sample_id immediately so that sample stats use the intended value
            tcr.clonotypes_df['sample_id'] = row[sample_id_col]
            # Capture per-sample stats after updating sample_id
            stats_list.append(tcr._get_reading_stats_obj())
            # Collect clonotype rows, now with the correct sample_id
            dfs.append(tcr.clonotypes_df.copy())

        combined = pd.concat(dfs, ignore_index=True)
        # Build a DataFrame from the computed sample stats.
        sample_stats_df = pd.DataFrame([asdict(s) for s in stats_list]).set_index('sample_id')

        # Create the obs from sample_info.
        obs = sample_info.set_index(sample_id_col)

        # If any of the sample stats columns already exist in obs, drop them
        for col in sample_stats_df.columns:
            if col in obs.columns:
                obs = obs.drop(columns=[col])

        # Merge the computed sample stats into obs.
        obs = obs.join(sample_stats_df, how='left')

        if 'group' in obs.columns:
            obs['group'] = obs['group'].astype('category')

        logger.info(f"Loaded {len(dfs)} samples with {combined.shape[0]} total clonotype rows.")
        return cls(combined, obs, data_subset="all", project_name=project_name)

    def _get_reading_stats_obj(self) -> SampleStats:
        """
        Produce a SampleStats dataclass for the first sample in this clonotypes_df.
        Used internally in from_sample_info.
        """
        sid = self.clonotypes_df['sample_id'].iat[0]
        n_total = len(self.clonotypes_df)
        n_unique = self.clonotypes_df['combined_custom_id'].nunique()
        n_prod = self.clonotypes_df[self.clonotypes_df['productive']].shape[0]
        n_tra = self.clonotypes_df[self.clonotypes_df['TRA_custom_id']!='']\
                    ['combined_custom_id'].nunique()
        n_trb = self.clonotypes_df[self.clonotypes_df['TRB_custom_id']!='']\
                    ['combined_custom_id'].nunique()
        pct_tra = n_tra / n_prod if n_prod else 0
        pct_trb = n_trb / n_prod if n_prod else 0
        if 'consensus_count' in self.clonotypes_df:
            counts = self.clonotypes_df.groupby('combined_custom_id')['consensus_count'].sum()
            pct_gt1 = (counts > 1).sum() / n_unique * 100 if n_unique else np.nan
        else:
            pct_gt1 = np.nan

        return SampleStats(sid, n_total, n_unique, pct_tra, pct_trb, pct_gt1)

    def to_anndata(self) -> "AnnData":
        """Export to an AnnData object (cells×clones counts)."""
        from anndata import AnnData
        cm = self.X
        return AnnData(
            X=cm.T.values,
            obs=self.obs.copy() if self.obs is not None else pd.DataFrame(index=cm.columns),
            var=pd.DataFrame(index=cm.index)
        )

    def to_airr(self) -> pd.DataFrame:
        """Return a pure AIRR-format table (read-level)."""
        cols = ['sample_id','chain_clonotype_id','v_call','d_call','j_call','junction_aa','productive']
        return self.clonotypes_df[cols].copy()

    def summary(self) -> pd.DataFrame:
        """Global summary of the experiment (single row)."""
        stats = {
            'n_samples': len(self.samples),
            'n_total_clones': len(self.clonotypes_df),
            'n_unique_clones': self.clonotypes_df['combined_custom_id'].nunique(),
            'n_productive_unique': self.clonotypes_df[self.clonotypes_df['productive']]
                                        ['combined_custom_id'].nunique(),
            'mean_entropy': float(self.entropy.mean()),
            'mean_gini': float(self.gini.mean()),
            'mean_simpson': float(self.simpson.mean()),
        }
        return pd.DataFrame.from_dict(stats, orient='index', columns=['value'])
    def add_umi_stats(
        self,
        source: Union[str, pd.DataFrame],
        index_col: str = "sample_id",
        qc_cols: Optional[List[str]] = None,
        **read_csv_kwargs
    ) -> None:
        """
        Register columns as sequencing stats and Merge per-sample UMI/QC stats into self.obs.

        USAGE:
            tcr = TCRSeq.from_sample_info(sample_info, …)
            tcr.add_umi_stats("all_samples_umi_count.tsv", qc_cols=["reads_total","reads_left_pct"])
            p = tcr.pl.plot_sequence_stats(
                x_col="reads_total",
                y_col="n_unique_clones",
                color_by="group"
            )

        Parameters:
          source: Either a file path (string) or a pandas DataFrame containing UMI stats.
          index_col: The column to use as the index (default "sample_id").
          qc_cols: Optional list of columns to extract from the source. If these columns
                   already exist in self.obs, a warning is issued and they will be overwritten.
          **read_csv_kwargs: Additional keyword arguments for pd.read_csv if source is a file path.

        Behavior:
          - If source is a DataFrame, it is used directly; otherwise, it is read using pd.read_csv.
          - The function checks that index_col exists in the source and (if provided) that qc_cols are present.
          - Before merging, if any of the qc_cols already exist in self.obs, those columns are dropped (overwritten)
            after logging a warning.
          - The merged DataFrame is stored in self.obs.
        """
        import pandas as pd

        # Load source as a DataFrame if not already one.
        if isinstance(source, pd.DataFrame):
            qc = source.copy()
        else:
            qc = pd.read_csv(source, **read_csv_kwargs)

        # Ensure the index column exists.
        if index_col not in qc.columns:
            raise KeyError(f"add_umi_stats: Required column '{index_col}' not found in source (provided as {source}).")
        qc = qc.set_index(index_col)

        # If qc_cols are provided, select only those columns.
        if qc_cols is not None:
            missing = set(qc_cols) - set(qc.columns)
            if missing:
                raise KeyError(f"add_umi_stats: Requested qc_cols {missing} not found in source.")
            qc = qc[qc_cols]

        # If self.obs exists, check for duplicate columns.
        if self.obs is not None:
            duplicate_cols = set(qc.columns).intersection(self.obs.columns)
            if duplicate_cols:
                logger.warning(f"add_umi_stats: The following columns already exist in obs and will be overwritten: {duplicate_cols}")
                # Drop the duplicate columns from self.obs so they can be overwritten.
                self.obs = self.obs.drop(columns=list(duplicate_cols))

        # Ensure self.obs is mergeable: set the proper index.
        if self.obs is not None:
            if self.obs.index.name != index_col:
                if index_col in self.obs.columns:
                    self.obs = self.obs.set_index(index_col)
                else:
                    raise KeyError(f"add_umi_stats: Cannot merge because '{index_col}' is missing in obs.")
            self.obs = self.obs.join(qc, how="left")
        else:
            self.obs = qc.copy()

        logger.info(f"add_umi_stats: merged UMI stats into obs ({len(qc)} samples).")

class _TCRSeqPreprocessing:
    """Filtering and subsetting operations; produce new TCRSeq with updated history."""
    def __init__(self, adata: TCRSeq):
        self.adata = adata

    def filter_by_min_count(self, min_count: int) -> TCRSeq:
        # Log input TRA/TRB counts
        input_df = self.adata.clonotypes_df
        input_tra = input_df[input_df['TRA_custom_id'] != ''].shape[0]
        input_trb = input_df[input_df['TRB_custom_id'] != ''].shape[0]

        counts = input_df.groupby('combined_custom_id').size()
        keep = counts[counts >= min_count].index
        new_df = input_df[input_df['combined_custom_id'].isin(keep)].copy()

        # Log output TRA/TRB counts
        output_tra = new_df[new_df['TRA_custom_id'] != ''].shape[0]
        output_trb = new_df[new_df['TRB_custom_id'] != ''].shape[0]
        logger.info(f"filter_by_min_count (min_count ≥ {min_count}): "
                    f"Input counts: TRA={input_tra}, TRB={input_trb}; "
                    f"Output counts: TRA={output_tra}, TRB={output_trb}")

        new_obs = self.adata.obs.loc[new_df['sample_id'].unique()] if self.adata.obs is not None else None
        new = TCRSeq(new_df, new_obs,
                     data_subset=self.adata.data_subset,
                     project_name=self.adata.project_name)
        new._record_step(f"min_count≥{min_count}")
        return new

    def filter_shared_clones(self, min_samples: int, group: Optional[str] = None) -> TCRSeq:
        input_df = self.adata.clonotypes_df
        input_tra = input_df[input_df['TRA_custom_id'] != ''].shape[0]
        input_trb = input_df[input_df['TRB_custom_id'] != ''].shape[0]

        if group and self.adata.obs is not None and 'group' in self.adata.obs:
            samples = self.adata.obs[self.adata.obs['group'] == group].index.tolist()
        else:
            samples = self.adata.samples

        counts = input_df[input_df['sample_id'].isin(samples)]\
            .groupby('combined_custom_id')['sample_id']\
            .nunique()
        keep = counts[counts >= min_samples].index
        new_df = input_df[input_df['combined_custom_id'].isin(keep)].copy()

        output_tra = new_df[new_df['TRA_custom_id'] != ''].shape[0]
        output_trb = new_df[new_df['TRB_custom_id'] != ''].shape[0]
        logger.info(f"filter_shared_clones (min_samples ≥ {min_samples}, group={group}): "
                    f"Input counts: TRA={input_tra}, TRB={input_trb}; "
                    f"Output counts: TRA={output_tra}, TRB={output_trb}")

        new_obs = self.adata.obs.loc[new_df['sample_id'].unique()] if self.adata.obs is not None else None
        new = TCRSeq(new_df, new_obs,
                     data_subset=self.adata.data_subset,
                     project_name=self.adata.project_name)
        new._record_step(f"shared≥{min_samples}samples")
        return new

    def subset_by_group(self, group_value: str) -> TCRSeq:
        if self.adata.obs is None or 'group' not in self.adata.obs:
            raise ValueError("Missing 'group' in obs")

        input_df = self.adata.clonotypes_df
        input_tra = input_df[input_df['TRA_custom_id'] != ''].shape[0]
        input_trb = input_df[input_df['TRB_custom_id'] != ''].shape[0]

        samples = self.adata.obs[self.adata.obs['group'] == group_value].index.tolist()
        new_df = input_df[input_df['sample_id'].isin(samples)].copy()

        output_tra = new_df[new_df['TRA_custom_id'] != ''].shape[0]
        output_trb = new_df[new_df['TRB_custom_id'] != ''].shape[0]
        logger.info(f"subset_by_group (group={group_value}): "
                    f"Input counts: TRA={input_tra}, TRB={input_trb}; "
                    f"Output counts: TRA={output_tra}, TRB={output_trb}")

        new_obs = self.adata.obs.loc[samples].copy()
        new = TCRSeq(new_df, new_obs,
                     data_subset=f"{self.adata.data_subset}_group:{group_value}",
                     project_name=self.adata.project_name)
        new._record_step(f"group={group_value}")
        return new

    def subset_by_chain(self, chain: str) -> TCRSeq:
        valid = {"TRA": ["TRA", "TRAC"], "TRB": ["TRB", "TRBC"]}
        if chain not in valid:
            raise ValueError("chain must be 'TRA' or 'TRB'")

        input_df = self.adata.clonotypes_df
        input_tra = input_df[input_df['TRA_custom_id'] != ''].shape[0]
        input_trb = input_df[input_df['TRB_custom_id'] != ''].shape[0]

        new_df = input_df[input_df['locus'].isin(valid[chain])].copy()

        output_tra = new_df[new_df['TRA_custom_id'] != ''].shape[0]
        output_trb = new_df[new_df['TRB_custom_id'] != ''].shape[0]
        logger.info(f"subset_by_chain (chain={chain}): "
                    f"Input counts: TRA={input_tra}, TRB={input_trb}; "
                    f"Output counts: TRA={output_tra}, TRB={output_trb}")

        new_obs = self.adata.obs.loc[new_df['sample_id'].unique()] if self.adata.obs is not None else None
        new = TCRSeq(new_df, new_obs,
                     data_subset=f"{self.adata.data_subset}_chain:{chain}",
                     project_name=self.adata.project_name)
        new._record_step(f"chain={chain}")
        return new


class _TCRSeqTools:
    """Core calculations: counts matrix, diversity measures, shared clones."""
    def __init__(self, adata: TCRSeq):
        self.adata = adata

    def get_counts_matrix(self) -> pd.DataFrame:
        if self.adata._cm is None:
            df = self.adata.clonotypes_df
            if 'consensus_count' in df.columns:
                cm = df.pivot_table(
                    index='combined_custom_id',
                    columns='sample_id',
                    values='consensus_count',
                    aggfunc='sum',
                    fill_value=0
                )
            else:
                cm = df.groupby(['combined_custom_id','sample_id'])\
                       .size().unstack(fill_value=0)
            self.adata._cm = cm
        return self.adata._cm

    def compute_entropy(self) -> pd.Series:
        cm = self.get_counts_matrix()
        return pd.Series({
            s: entropy(cm[s] / cm[s].sum()) if cm[s].sum() else np.nan
            for s in cm.columns
        })

    def compute_gini(self) -> pd.Series:
        def gini(x: np.ndarray) -> float:
            if np.all(x == 0):
                return 0.0
            sx = np.sort(x)
            n = len(sx)
            cum = np.cumsum(sx)
            return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
        cm = self.get_counts_matrix()
        return pd.Series({s: gini(cm[s].values) for s in cm.columns})

    def compute_simpson(self) -> pd.Series:
        cm = self.get_counts_matrix().astype(float)
        return pd.Series({
            s: 1 - np.sum((cm[s] / cm[s].sum())**2) if cm[s].sum() else np.nan
            for s in cm.columns
        })

    def get_sample_stats(self) -> pd.DataFrame:
        stats = []
        for sid, sub in self.adata.clonotypes_df.groupby('sample_id'):
            temp = TCRSeq(sub,
                          self.adata.obs.loc[[sid]] if self.adata.obs is not None and sid in self.adata.obs.index else None,
                          data_subset=self.adata.data_subset,
                          project_name=self.adata.project_name)
            stats.append(asdict(temp._get_reading_stats_obj()))
        return pd.DataFrame(stats)
    
    def get_shared_clones(self, n_top: int = 10) -> pd.DataFrame:
        """
        Return the top N clones with the highest average abundance across samples.
        
        The function calculates the mean count of each clone across all samples (from the counts matrix),
        sorts them in descending order (so that clones with the highest average counts come first), and
        returns the top n clones.

        Args:
            n_top: Number of top clones to return.

        Returns:
            A pandas DataFrame with columns:
                - The clone identifier (from the index, reset as a column)
                - 'mean_count': the average clone count across samples (sorted in descending order)
        """
        cm = self.get_counts_matrix()
        df_mean = cm.mean(axis=1).reset_index(name='mean_count')
        return df_mean.sort_values('mean_count', ascending=False).head(n_top)

class _TCRSeqPlotting:
    """All ggplot2‐style plotting via plotnine and matplotlib."""
    def __init__(self, adata: TCRSeq):
        self.adata = adata

    def plot_scatter_from_obs(
        self,
        x_col: str,
        y_col: str,
        color_by: Optional[str] = None,
        log_scale: bool = True,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None
    ) -> p9.ggplot:
        """
        Scatter plot constructed from two columns in the sample metadata (.obs).
        The specified x_col and y_col can come either from the original sample metadata
        or from computed sequencing stats (via tl.get_sample_stats()). In the latter case,
        if the requested columns are missing from obs (or all NaN), this function merges in
        computed stats on "sample_id" to fill those values.
        
        A best-fit (linear) line is overlaid and the Spearman correlation (ρ) and its p-value
        are shown as an annotation on the plot. The annotation location is computed from the
        data’s bounding box so that it is automatically placed (the project label appears in 
        the top-left, and the correlation is now at the bottom-left).
        Optionally, lower limits on the x and y axes can be enforced following log10-transformation.
        
        Parameters:
          x_col : The name of the column for the x-axis.
          y_col : The name of the column for the y-axis.
          color_by : (Optional) Column name to assign colors by.
          log_scale : If True, both axes are log10-transformed. (Default: True)
          x_min : Optional lower limit for the x-axis (applied after log10 transformation if log_scale=True).
          y_min : Optional lower limit for the y-axis.
        
        Returns:
          A plotnine ggplot scatter plot object.
        """
        import pandas as pd
        from scipy.stats import spearmanr

        # Start with self.adata.obs; ensure that 'sample_id' is present by resetting the index.
        if self.adata.obs is not None and not self.adata.obs.empty:
            df = self.adata.obs.copy().reset_index()
        else:
            df = pd.DataFrame()

        # Check whether x_col and y_col exist (or if all values in those columns are NaN).
        missing = []
        for col in (x_col, y_col):
            if col not in df.columns or df[col].isna().all():
                missing.append(col)

        # If missing, merge in computed sample stats (via tl.get_sample_stats)
        if missing:
            stats = self.adata.tl.get_sample_stats().reset_index()  # ensures "sample_id" exists
            df = pd.merge(df, stats, on="sample_id", how="outer", suffixes=("", "_computed"))
            # For each originally missing column, fill it in from the computed version if available.
            for col in missing:
                computed_col = col + "_computed"
                if computed_col in df.columns:
                    df[col] = df[col].combine_first(df[computed_col])

        # Verify that both columns are now present.
        for col in (x_col, y_col):
            if col not in df.columns:
                raise KeyError(
                    f"plot_scatter_from_obs: Column '{col}' not found in the sample metadata or computed stats. "
                    "Please ensure that the provided column name is correct."
                )

        # Convert the requested columns to numeric and drop rows with missing values.
        for col in (x_col, y_col):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[x_col, y_col])

        # Compute Spearman correlation between x_col and y_col.
        rho, pval = spearmanr(df[x_col], df[y_col])
        title = f"Spearman (rho={rho:.2f}, p={pval:.3g})"

        # Compute bounding box information from the data.
        x_min_val, x_max_val = df[x_col].min(), df[x_col].max()
        y_min_val, y_max_val = df[y_col].min(), df[y_col].max()
        x_range = x_max_val - x_min_val
        y_range = y_max_val - y_min_val

        # Place annotation for project info at the top-left.
        project_label = f"[{self.adata.project_name}::{self.adata.data_subset}]"
        project_annot_x = x_min_val + 0.05 * x_range
        project_annot_y = y_max_val - 0.05 * y_range

        # Build aesthetic mapping.
        aes_kwargs = {"x": x_col, "y": y_col}
        if color_by is not None and color_by in df.columns:
            aes_kwargs["color"] = color_by

        # Create the scatter plot with best-fit (linear) line.
        plot = (
            p9.ggplot(df, p9.aes(**aes_kwargs))
            + p9.geom_point()
            + p9.geom_smooth(method="lm", se=False, color="black")
            + p9.labs(x=x_col, y=y_col, title=title)
            + economist_theme()
            + p9.annotate(
                "text",
                x=project_annot_x, y=project_annot_y,
                label=project_label, size=8, ha="left", va="top",color="gray"
            )
        )

        # Compute coordinates for the correlation annotation at the bottom-left.
        annot_x_bot = x_min_val + 0.05 * x_range
        annot_y_bot = y_min_val + 0.05 * y_range

        plot += p9.annotate(
            "text",
            x=annot_x_bot,
            y=annot_y_bot,
            label=f"rho={rho:.2f}\np={pval:.3g}",
            ha="left",
            va="bottom",
            size=10
        )

        if log_scale:
            if x_min is not None:
                plot += p9.scale_x_log10(limits=(x_min, None), labels=lambda l: [f"{v:.1e}" for v in l])
            else:
                plot += p9.scale_x_log10(labels=lambda l: [f"{v:.1e}" for v in l])
            if y_min is not None:
                plot += p9.scale_y_log10(limits=(y_min, None), labels=lambda l: [f"{v:.1e}" for v in l])
            else:
                plot += p9.scale_y_log10(labels=lambda l: [f"{v:.1e}" for v in l])
        return plot
    def plot_sample_stats(self, split_by: Optional[str] = None,
                          x_min: Optional[float] = None,
                          y_min: Optional[float] = None,
                          label_order: Optional[List[str]] = None) -> Union[p9.ggplot, tuple]:
        """Per-sample barplots of basic metrics and TRA/TRB clone counts.
        
        This updated version shows two measures for TRA/TRB:
          - "unique": the number of unique clones (duplicates dropped)
          - "total": the total count (e.g. using consensus_count if available)
          
        The first plot (p1) shows additional per-sample metrics.
        The second plot (p2) is faceted by measure so that the top facet shows unique counts
        and the bottom facet shows consensus (total) counts. In each facet each sample is shown
        as a single stacked bar (stacked by chain: TRA and TRB). The bar outline color is set to 
        black for unique counts and grey for total counts.
        """
        # ---- First plot: Per-sample metrics ----
        stats = self.adata.tl.get_sample_stats()
        if split_by == 'group':
            id_vars_ = ['sample_id', 'group']
        else:
            id_vars_ = ['sample_id']
        melt = stats.drop(columns=['n_unique_clones'], errors='ignore').melt(
            id_vars=id_vars_,
            var_name='metric', value_name='value'
        )
        
        if split_by and split_by in melt.columns and split_by != 'sample_id':
            aes1 = p9.aes(x='sample_id', y='value', fill=split_by)
        else:
            aes1 = p9.aes(x='sample_id', y='value')
        
        p1 = (
            p9.ggplot(melt, aes1)
            + p9.geom_col()
            + p9.facet_wrap('~metric', scales='free_y')
            + p9.labs(title=f"[{self.adata.project_name}:{self.adata.data_subset}] Per-sample metrics")
            + economist_theme()
            + p9.theme(panel_spacing=0.4)
        )
        
        if label_order is not None and len(label_order) > 0:
            p1 += p9.scale_x_discrete(limits=list(label_order))
        
        # ---- Second plot: TRA/TRB counts as two stacked bars (unique and total) ----
        df = self.adata.clonotypes_df
        
        # TRA unique counts: number of distinct TRA clones per sample
        tra_unique = (
            df[df['TRA_custom_id'] != '']
            .drop_duplicates(['sample_id', 'TRA_custom_id'])
            .groupby('sample_id').size().reset_index(name='count')
        )
        tra_unique['chain'] = 'TRA'
        tra_unique['measure'] = 'unique'
        
        # TRA total counts: sum consensus_count if available; otherwise count rows
        if 'consensus_count' in df.columns:
            tra_total = (
                df[df['TRA_custom_id'] != '']
                .drop_duplicates(['sample_id', 'TRA_custom_id'])
                .groupby('sample_id')['consensus_count']
                .sum().reset_index(name='count')
            )
        else:
            tra_total = (
                df[df['TRA_custom_id'] != '']
                .groupby('sample_id').size().reset_index(name='count')
            )
        tra_total['chain'] = 'TRA'
        tra_total['measure'] = 'total'
        
        # TRB unique counts: number of distinct TRB clones per sample
        trb_unique = (
            df[df['TRB_custom_id'] != '']
            .drop_duplicates(['sample_id', 'TRB_custom_id'])
            .groupby('sample_id').size().reset_index(name='count')
        )
        trb_unique['chain'] = 'TRB'
        trb_unique['measure'] = 'unique'
        
        # TRB total counts: sum consensus_count if available; otherwise count rows
        if 'consensus_count' in df.columns:
            trb_total = (
                df[df['TRB_custom_id'] != '']
                .drop_duplicates(['sample_id', 'TRB_custom_id'])
                .groupby('sample_id')['consensus_count']
                .sum().reset_index(name='count')
            )
        else:
            trb_total = (
                df[df['TRB_custom_id'] != '']
                .groupby('sample_id').size().reset_index(name='count')
            )
        trb_total['chain'] = 'TRB'
        trb_total['measure'] = 'total'
        
        # Combine counts for both measures
        counts = pd.concat([tra_unique, tra_total, trb_unique, trb_total], ignore_index=True)
        
        p2 = (
            p9.ggplot(counts, p9.aes(x='sample_id', y='count', fill='chain', color='measure'))
            + p9.geom_col(width=0.7, position="stack")
            + p9.facet_wrap("~measure", ncol=1, scales='free_y')
            + p9.labs(
                  title=f"[{self.adata.project_name}:{self.adata.data_subset}] TRA/TRB Counts: Unique and Total",
                  x="Sample", y="Clone Count", fill="Chain"
              )
            + p9.scale_color_manual(values={'unique': 'black', 'total': 'grey'})
            + economist_theme()
            + p9.theme(panel_spacing=0.4)
        )
        
        if label_order is not None and len(label_order) > 0:
            p2 += p9.scale_x_discrete(limits=list(label_order))
        
        return p1, p2

    def plot_sample_stats_old(
        self,
        split_by: Optional[str] = None,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None,
        label_order: Optional[List[str]] = None
    ) -> Union[p9.ggplot, tuple]:
        """Per-sample barplots of basic metrics and TRA/TRB clone counts."""
        stats = self.adata.tl.get_sample_stats()
        if split_by == 'group':
            id_vars_ = ['sample_id', 'group']
        else:
            id_vars_ = ['sample_id']
        melt = stats.drop(columns=['n_unique_clones'], errors='ignore').melt(
            id_vars=id_vars_,
            var_name='metric', value_name='value'
        )
        
        if split_by and split_by in melt.columns and split_by != 'sample_id':
            aes1 = p9.aes(x='sample_id', y='value', fill=split_by)
        else:
            aes1 = p9.aes(x='sample_id', y='value')
        
        p1 = (
            p9.ggplot(melt, aes1)
            + p9.geom_col()
            + p9.facet_wrap('~metric', scales='free_y')
            + p9.labs(title=f"[{self.adata.project_name}:{self.adata.data_subset}] Per-sample metrics")
            + economist_theme()
            + p9.theme(panel_spacing=0.4)
        )
        
        if label_order is not None and len(label_order) > 0:
            p1 += p9.scale_x_discrete(limits=list(label_order))
        
        # Second plot section remains (using fill="chain"):
        df = self.adata.clonotypes_df
        tra = (
            df[df['TRA_custom_id']!='']
            .drop_duplicates(['sample_id', 'TRA_custom_id'])
            .groupby('sample_id').size()
            .reset_index(name='clone_count')
        )
        tra['chain'] = 'TRA'
        trb = (
            df[df['TRB_custom_id']!='']
            .drop_duplicates(['sample_id', 'TRB_custom_id'])
            .groupby('sample_id').size()
            .reset_index(name='clone_count')
        )
        trb['chain'] = 'TRB'
        counts = pd.concat([tra, trb], ignore_index=True)
        aes2 = p9.aes(x='sample_id', y='clone_count', fill='chain')
        p2 = (
            p9.ggplot(counts, aes2)
            + p9.geom_col()
            + p9.labs(title=f"[{self.adata.project_name}:{self.adata.data_subset}] Unique TRA/TRB counts")
            + economist_theme()
            + p9.theme(panel_spacing=0.4)
        )
        if label_order is not None and len(label_order) > 0:
            p2 += p9.scale_x_discrete(limits=list(label_order))
        
        #if cp:
        #    import numpy as np
        #    patch = cp.patch(p1, p2)
        #    patch += cp.layout(design=np.array([0, 1]).reshape(1,2))
        #    return patch
        return p1, p2

    def plot_shared_clone(self, split_by: Optional[str] = None, n_top: int = 10) -> p9.ggplot:
        """Bubble-plot of the top N clones shared across samples."""
        # Get the top shared clones
        shared = self.adata.tl.get_shared_clones(n_top)
        # Melt the counts matrix for the shared clones
        cm = (
            self.adata.X
            .loc[shared['combined_custom_id']]
            .reset_index()
            .melt(
                id_vars='combined_custom_id',
                var_name='sample_id',
                value_name='clone_count'
            )
        )
        # Determine the coloring variable
        color = split_by if (split_by and self.adata.obs is not None and split_by in self.adata.obs) else 'sample_id'
        # Define the aesthetic mapping: mapping clone_count to size
        aes_mapping = p9.aes(
            x='sample_id',
            y='combined_custom_id',
            size='clone_count',
            color=color
        )
        # Build the ggplot object
        p = (
            p9.ggplot(cm, aes_mapping)
            + p9.geom_point()
            + p9.labs(
                  title=f"[{self.adata.project_name}:{self.adata.data_subset}] Top {n_top} shared clones"
              )
            # Here we add a custom size scale with fixed breaks
            + p9.scale_size_continuous(breaks=[0, 100, 200, 300, 400])
            + economist_theme()
            + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
        )
        return p



    def plot_venn(self, group: Optional[str] = None) -> plt.Figure:
        """Venn diagram of overlaps for 2 or 3 samples (or plain text otherwise), with percentages."""
        if group and self.adata.obs is not None and 'group' in self.adata.obs.columns:
            samples = self.adata.obs[self.adata.obs['group'] == group].index.tolist()
        else:
            samples = self.adata.samples

        cm = self.adata.tl.get_counts_matrix()

        # 2-way Venn
        if len(samples) == 2 and venn2:
            s1, s2 = samples
            set1 = set(cm.index[cm[s1] > 0])
            set2 = set(cm.index[cm[s2] > 0])
            total = len(set1 | set2)

            plt.figure(figsize=(8, 6))
            # Create the venn diagram and get the patch object.
            v = venn2(
                (len(set1 - set2), len(set2 - set1), len(set1 & set2)),
                set_labels=(s1, s2)
            )
            venn2_circles(
                (len(set1 - set2), len(set2 - set1), len(set1 & set2)),
                linestyle='dashed', linewidth=1
            )
            # Annotate each region with count and percentage
            for region, count in (('10', len(set1 - set2)),
                                  ('01', len(set2 - set1)),
                                  ('11', len(set1 & set2))):
                label = v.get_label_by_id(region)
                if label is not None:
                    label.set_text(f"{count}\n({count/total*100:.1f}%)")

            plt.title(f"[{self.adata.project_name}:{self.adata.data_subset}]", fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

        # 3-way Venn
        elif len(samples) == 3 and venn3:
            s1, s2, s3 = samples
            set1 = set(cm.index[cm[s1] > 0])
            set2 = set(cm.index[cm[s2] > 0])
            set3 = set(cm.index[cm[s3] > 0])
            total = len(set1 | set2 | set3)

            A = len(set1 - set2 - set3)
            B = len(set2 - set1 - set3)
            AB = len((set1 & set2) - set3)
            C = len(set3 - set1 - set2)
            AC = len((set1 & set3) - set2)
            BC = len((set2 & set3) - set1)
            ABC = len(set1 & set2 & set3)

            plt.figure(figsize=(8, 6))
            v = venn3((A, B, AB, C, AC, BC, ABC), set_labels=(s1, s2, s3))
            venn3_circles((A, B, AB, C, AC, BC, ABC), linestyle='dashed', linewidth=1)
            
            # Mapping of venn3 regions to their counts.
            annotation_mapping = {
                '100': A,
                '010': B,
                '110': AB,
                '001': C,
                '101': AC,
                '011': BC,
                '111': ABC
            }
            for region, count in annotation_mapping.items():
                label = v.get_label_by_id(region)
                if label is not None:
                    label.set_text(f"{count}\n({count/total*100:.1f}%)")

            plt.title(f"[{self.adata.project_name}:{self.adata.data_subset}]", fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

        # Fallback for unsupported numbers of sets
        else:
            plt.figure(figsize=(8, 6))
            plt.text(
                0.5, 0.5,
                f"Venn diagram supports only 2 or 3 sets; got {len(samples)}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12, fontweight="bold"
            )
            plt.title("Venn Diagram", fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

    def plot_upset(self, label_order: Optional[List[str]] = None) -> plt.Figure:
        """Upset plot of clone presence/absence across samples with optional label ordering."""
        cm = self.adata.tl.get_counts_matrix()
        pres = cm > 0
        # Reorder columns if label_order is provided.
        if label_order is not None:
            # Ensure that all labels in label_order exist in pres.columns:
            missing = set(label_order) - set(pres.columns)
            if missing:
                raise KeyError(f"The following labels in label_order were not found in the data: {missing}")
            pres = pres[label_order]
        upset_series = from_indicators(pres.columns.tolist(), pres)
        upset_series.name = "count"
        upset_plot(upset_series)
        plt.suptitle(
            f"[{self.adata.project_name}:{self.adata.data_subset}] Upset Plot",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        return plt.gcf()



    def plot_sharing_heatmap(self) -> plt.Figure:
        """Heatmap of pairwise Jaccard indices between samples."""
        cm = (self.adata.X > 0).astype(int)
        samples = cm.columns.tolist()
        mat = pd.DataFrame(index=samples, columns=samples, dtype=float)
        for i in samples:
            for j in samples:
                a = set(cm.index[cm[i] > 0])
                b = set(cm.index[cm[j] > 0])
                mat.loc[i, j] = len(a & b) / len(a | b) if a | b else 0
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels(samples, rotation=90)
        ax.set_yticks(range(len(samples)))
        ax.set_yticklabels(samples)
        fig.colorbar(im, ax=ax, label='Jaccard index')
        fig.tight_layout()
        return fig


    def plot_cdr3_length(
        self,
        chain: Optional[str] = None,
        split_by: Optional[str] = None
    ) -> p9.ggplot:
        """Histogram of CDR3 amino-acid lengths, optionally colored by obs[col]."""
        df = self.adata.clonotypes_df.copy()
        if chain:
            valid = {"TRA": ["TRA","TRAC"], "TRB": ["TRB","TRBC"]}
            df = df[df['locus'].isin(valid[chain])]
        df = df.assign(cdr3_len=df['junction_aa'].str.len())
        max_len = int(df['cdr3_len'].max() or 0)

        aes = p9.aes(x='cdr3_len')
        if split_by and self.adata.obs is not None and split_by in self.adata.obs.columns:
            df = df.merge(self.adata.obs.reset_index()[['sample_id', split_by]],
                          on='sample_id', how='left')
            aes = p9.aes(x='cdr3_len', fill=split_by)

        p = (
            p9.ggplot(df, aes)
            + p9.geom_histogram(binwidth=1, color="white", position="dodge")
            + p9.labs(
                title=f"[{self.adata.project_name}:{self.adata.data_subset}] CDR3 length distribution{(' '+chain) if chain else ''}",
                x="CDR3 length", y="Count"
            )
            + economist_theme()
            + p9.scale_x_continuous(limits=(0, max_len + 1))
        )
        return p

    def plot_gene_usage(
        self,
        gene: str = 'v_call',
        label_order: Optional[List[str]] = None,
        filter_constant: bool = False,
        shared: bool = False,
        unique: bool = False,
        normalized: bool = True
    ) -> p9.ggplot:
        """
        Enhanced Heatmap of gene usage across samples.

        For each sample on the x‐axis, this function calculates the frequency of each gene 
        (by default, 'v_call') relative to all genes in that sample. You can choose whether to 
        base the counts on all clones (using consensus_count if available) or only on unique clones 
        (by dropping duplicate calls per sample/gene). In addition, if normalized=True then the value 
        for each gene is computed as (gene count/total counts for that sample)*100.

        New parameters:
          gene:            Column name to use for gene (default 'v_call').
          label_order:     Optional list specifying the order of sample IDs for the x-axis.
          filter_constant: If True, remove genes for which the (normalized or raw) value is the same across all samples.
          shared:          If True, only include genes that appear (nonzero) in every sample.
          unique:          If True, count only a single occurrence per sample for each gene (i.e. consider unique clones only).
          normalized:      If True, the gene count is divided by the sample’s total so that final values are percentages.
                           If False, the raw counts are presented.

        Returns:
          A plotnine ggplot object for the heatmap.
        """
        import pandas as pd
        # Start from a copy of the clonotypes DataFrame.
        df = self.adata.clonotypes_df.copy()

        if gene not in df.columns:
            raise ValueError(f"plot_gene_usage: Column '{gene}' not found in clonotypes_df.")

        # If using unique clones only, drop duplicate gene calls per sample.
        if unique:
            df = df.drop_duplicates(subset=["sample_id", gene])
            df["gene_count"] = 1
        else:
            # Use consensus_count if available, otherwise count each record as 1.
            if "consensus_count" in df.columns:
                df["gene_count"] = df["consensus_count"]
            else:
                df["gene_count"] = 1

        # Aggregate counts per sample and gene.
        agg = (
            df.groupby(["sample_id", gene])["gene_count"]
              .sum()
              .reset_index()
              .rename(columns={gene: "gene"})
        )

        # Compute the total counts per sample.
        totals = (
            agg.groupby("sample_id")["gene_count"]
               .sum()
               .reset_index()
               .rename(columns={"gene_count": "total_count"})
        )
        agg = agg.merge(totals, on="sample_id", how="left")

        # Compute usage value: normalized percentage or raw count.
        if normalized:
            agg["value"] = agg["gene_count"] / agg["total_count"] * 100
        else:
            agg["value"] = agg["gene_count"]

        # Determine samples to be used for the x-axis. If label_order is provided, use those;
        # otherwise, use the samples present in the aggregated data.
        samples = label_order if label_order is not None else sorted(agg["sample_id"].unique())

        # Pivot the data so that we can filter out genes based on variation or shared-ness.
        pivot_df = agg.pivot(index="gene", columns="sample_id", values="value").reindex(columns=samples)
        pivot_df = pivot_df.fillna(0)  # treat missing calls as 0

        # Optionally drop genes that are constant across samples.
        if filter_constant:
            constant_genes = pivot_df.index[pivot_df.nunique(axis=1) <= 1]
            pivot_df = pivot_df.drop(index=constant_genes)

        # Optionally, only keep genes that are observed in every sample (nonzero in all samples).
        if shared:
            non_shared_genes = pivot_df.index[(pivot_df == 0).any(axis=1)]
            pivot_df = pivot_df.drop(index=non_shared_genes)

        # Get the list of genes to keep.
        genes_to_keep = pivot_df.index.tolist()

        # Filter the aggregated dataframe to only include those genes.
        agg = agg[agg["gene"].isin(genes_to_keep)]

        # Build the heatmap.
        p = (
            p9.ggplot(agg, p9.aes(x="sample_id", y="gene", fill="value"))
            + p9.geom_tile()
            + p9.labs(
                title=f"[{self.adata.project_name}:{self.adata.data_subset}] {gene.upper()} Usage",
                x="Sample",
                y=gene.upper(),
                fill="Percent (%)" if normalized else "Count"
            )
            + economist_theme()
            + p9.scale_fill_gradient(low="white", high="steelblue")
        )

        # If a label_order is provided, enforce that order on the x-axis.
        if label_order is not None and len(label_order) > 0:
            p += p9.scale_x_discrete(limits=label_order)

        # Adjust the figure height based on the number of unique genes.
        n_genes = agg["gene"].nunique()
        base_height = 0.3  # inches per gene (adjust as needed)
        min_height = 5     # minimum height in inches
        dynamic_height = max(min_height, n_genes * base_height)
        p += p9.theme(figure_size=(8, dynamic_height))

        return p
    
    def plot_clone_rank(
        self,
        color_by: Optional[str] = None,
        split_by: Optional[str] = None,
        normalize: bool = False,
        sample_subset: Optional[List[str]] = None,
        log_scale: bool = True
    ) -> p9.ggplot:
        """
        Overlay rank–abundance curves for all samples.

        Each sample's clone count distribution is converted into a rank–abundance curve.
        Parameters:
          color_by:     (Optional) A column name from .obs to use for coloring the curves.
          split_by:     (Optional) A column name from .obs used to facet the plot.
                        Each unique value gets its own panel (all panels arranged in one row).
          normalize:    (Optional) If True, each clone's count in a sample is divided by the total number
                        of unique clones in that sample, so that the y-axis shows a value relative to the 
                        unique clone count.
          sample_subset:(Optional) A list of sample IDs to include. If None, all samples (columns in self.adata.X) are used.
          log_scale:    (Optional) If True, the x-axis is log10-transformed and, when not normalized, the y-axis as well.

        Returns:
          A plotnine ggplot object with overlaid rank–abundance curves.
        """
        import pandas as pd
        import numpy as np

        # Retrieve the counts matrix (rows = clones, columns = sample IDs)
        X = self.adata.X.copy()

        # Determine the samples to plot.
        if sample_subset is not None:
            valid_samples = [s for s in sample_subset if s in X.columns]
            if not valid_samples:
                raise ValueError("No provided sample IDs were found in the counts matrix.")
            X = X[valid_samples]
        else:
            valid_samples = list(X.columns)

        # For each sample, compute its rank–abundance curve.
        rank_list = []
        for s in valid_samples:
            ser = X[s].sort_values(ascending=False)
            if ser.sum() == 0:
                continue  # skip samples with zero counts
            df_sample = pd.DataFrame({
                "rank": np.arange(1, len(ser) + 1),
                "count": ser.values,
                "sample_id": s
            })
            if normalize:
                # Normalize by the total number of unique clones (i.e. the length of the series)
                total_unique = len(ser)
                df_sample["value"] = df_sample["count"] / total_unique
            else:
                df_sample["value"] = df_sample["count"]
            rank_list.append(df_sample)
        if not rank_list:
            raise ValueError("No data available from the provided samples.")
        rank_df = pd.concat(rank_list, ignore_index=True)

        # Merge sample-level metadata if either color_by or split_by is provided.
        if color_by is not None or split_by is not None:
            obs = self.adata.obs.copy()
            # Ensure "sample_id" exists as a column.
            if "sample_id" not in obs.columns:
                obs = obs.reset_index()
                if "sample_id" not in obs.columns:
                    obs = obs.rename(columns={obs.columns[0]: "sample_id"})
            meta_cols = ["sample_id"]
            if color_by is not None:
                meta_cols.append(color_by)
            if split_by is not None:
                meta_cols.append(split_by)
            obs = obs[meta_cols]
            rank_df = pd.merge(rank_df, obs, on="sample_id", how="left")

        # Define the basic aesthetics.
        if color_by is not None:
            mapping = p9.aes(x="rank", y="value", group="sample_id", color=color_by)
        else:
            mapping = p9.aes(x="rank", y="value", group="sample_id")

        # Create the base plot.
        p = (
            p9.ggplot(rank_df, mapping)
            + p9.geom_line()
            + p9.labs(
                title=f"[{self.adata.project_name}:{self.adata.data_subset}] Overlayed Rank–Abundance Curves",
                x="Rank",
                y="Relative Abundance" if normalize else "Clone Count"
            )
            + economist_theme()
        )

        # Apply log-scale transformation if requested.
        if log_scale:
            p += p9.scale_x_log10()
            if not normalize:
                p += p9.scale_y_log10()
                p += p9.scale_y_log10(
                    limits=(1, 10000),  # Set y-axis limits (must be >0 for log scale)
                    labels=lambda l: [f"{x:.0e}".replace("e+0", "e").replace("e+", "e") for x in l]
                    #labels=lambda l: [f'{x:.1e}' for x in l]  # Show y-axis labels in scientific notation
                )

        # Facet the plot if split_by is provided; all panels arranged in one row.
        if split_by is not None:
            p += p9.facet_wrap("~" + split_by, nrow=1)           
            n_splits = obs[split_by].nunique() 
            base_height = 0.3  # inches per gene (adjust as needed)
            min_height = 5     # minimum height in inches
            dynamic_height = max(min_height, n_splits * base_height)
            p += p9.theme(figure_size=(8, dynamic_height))

        return p


    
    def plot_clonal_expansion(
        self,
        bins: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        label_order: Optional[List[str]] = None
    ) -> p9.ggplot:
        """
        Create a stacked barplot of clone-size bins per sample.

        This function bins the clone sizes (using the 'consensus_count' column if available)
        and then plots, per sample, either the raw clone counts in each bin or the normalized
        proportions.

        Args:
            bins: Optional list of bin edges for clone-size (defaults to [0.5, 1.5, 2.5, 5.5, 20.5, np.inf]).
            labels: Optional labels for each bin (defaults to ["1", "2", "3-5", "6-20", ">20"]).
            normalize: If True, plot proportions (normalized counts) instead of raw counts.
            label_order: Optional list specifying the order for the sample_id on the x-axis.

        Returns:
            A plotnine ggplot object with the clonal expansion stacked barplot.
        """
        import numpy as np
        import pandas as pd

        # Access the clonotypes DataFrame from the TCRSeq object.
        df = self.adata.clonotypes_df

        # Use default bins and labels if not provided.
        if bins is None:
            bins = [0.5, 1.5, 2.5, 5.5, 20.5, np.inf]
        if labels is None:
            labels = ["1", "2", "3-5", "6-20", ">20"]

        # Group by sample_id and combined_custom_id.
        if 'consensus_count' in df.columns:
            cc = df.groupby(['sample_id', 'combined_custom_id'])['consensus_count'].sum().reset_index()
        else:
            cc = df.groupby(['sample_id', 'combined_custom_id']).size().reset_index(name='consensus_count')

        # Bin the consensus_count values into expansion bins.
        cc['expansion_bin'] = pd.cut(
            cc['consensus_count'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Count clones per sample and expansion_bin.
        exp_df = cc.groupby(['sample_id', 'expansion_bin']).size().reset_index(name='n_clones')

        # Optionally normalize the counts to proportions.
        if normalize:
            total = exp_df.groupby('sample_id')['n_clones'].transform('sum')
            exp_df['proportion'] = exp_df['n_clones'] / total
            y_val, ylab = 'proportion', "Proportion of Clones"
        else:
            y_val, ylab = 'n_clones', "Number of Clones"

        # Build the ggplot object using plotnine.
        aes_mapping = p9.aes(x='sample_id', y=y_val, fill='expansion_bin')
        plot = (
            p9.ggplot(exp_df, aes_mapping)
            + p9.geom_col()
            + p9.labs(
                x="Sample",
                y=ylab,
                fill="Clone Count",
                title=f"[{self.adata.project_name}:{self.adata.data_subset}] Clonal Expansion"
            )
            + economist_theme()
            + p9.theme(
                panel_spacing=0.4,
                axis_text_x=p9.element_text(rotation=45, ha="right")
            )
        )

        # Apply a custom x-axis order if provided.
        if label_order is not None and len(label_order):
            plot += p9.scale_x_discrete(limits=list(label_order))

        return plot

    def plot_cumulative_clone_discovery(self, sample_order: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot the cumulative discovery of new clones across technical replicates.

        For each sample (ordered by sample_order), the function calculates:
          - The number of “new” clones discovered in that sample (i.e. clones not seen in any previous sample)
          - The cumulative number of unique clones discovered up to that sample

        These values are plotted as a barplot (new clone counts per sample) with an overlaid
        line plot (cumulative unique clone count).

        Args:
            sample_order: Optional list specifying the order of sample IDs to use. If not provided,
                          the TCRSeq object's samples (sorted unique sample_ids) are used.

        Returns:
            The matplotlib Figure object.
        """
        # If not provided, use the samples from the TCRSeq object.
        if sample_order is None:
            sample_order = self.adata.samples

        cumulative_clones = set()
        new_counts = []
        cumulative_counts = []
        df = self.adata.clonotypes_df

        # Loop through the selected samples in order.
        for sample in sample_order:
            sample_clones = set(df.loc[df['sample_id'] == sample, 'combined_custom_id'])
            new = len(sample_clones - cumulative_clones)
            new_counts.append(new)
            cumulative_clones.update(sample_clones)
            cumulative_counts.append(len(cumulative_clones))

        # Create the plot.
        fig, ax1 = plt.subplots(figsize=(8, 6))
        x = range(len(sample_order))
        ax1.bar(x, new_counts, color='lightblue', edgecolor='black', label='New Clones Discovered')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('New Clones Discovered', color='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sample_order)

        ax2 = ax1.twinx()
        ax2.plot(x, cumulative_counts, color='red', marker='o', label='Cumulative Unique Clones')
        ax2.set_ylabel('Cumulative Unique Clones', color='red')

        # Merge the legends from both axes.
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.title("Cumulative New Clone Discovery Across Samples")
        plt.tight_layout()
        plt.show()

        return fig


    def plot_diversity(self, label_order: Optional[List[str]] = None, nrow: Optional[int] = None, fill: Optional[str] = None) -> p9.ggplot:
        import numpy as np
        import pandas as pd

        entropy_series = self.adata.entropy
        gini_series    = self.adata.gini
        simpson_series = self.adata.simpson

        if entropy_series.index.name is None:
            entropy_series.index.name = "sample_id"
        if gini_series.index.name is None:
            gini_series.index.name = "sample_id"
        if simpson_series.index.name is None:
            simpson_series.index.name = "sample_id"

        entropy_df = entropy_series.reset_index(name="value")
        entropy_df["measure"] = "Entropy"

        gini_df = gini_series.reset_index(name="value")
        gini_df["measure"] = "Gini"

        simpson_df = simpson_series.reset_index(name="value")
        simpson_df["measure"] = "Simpson"

        n_unique = self.adata.clonotypes_df.groupby("sample_id")["combined_custom_id"].nunique()
        pielou_series = entropy_series.copy()
        condition = n_unique > 1
        pielou_series.loc[condition] = pielou_series.loc[condition] / np.log(n_unique[condition])
        pielou_series.loc[~condition] = 1.0

        if pielou_series.index.name is None:
            pielou_series.index.name = "sample_id"
        pielou_df = pielou_series.reset_index(name="value")
        pielou_df["measure"] = "Pielou Evenness"

        diversity_df = pd.concat([entropy_df, gini_df, simpson_df, pielou_df], ignore_index=True)

        if fill is not None:
            obs_df = self.adata.obs.reset_index()[["sample_id", fill]]
            diversity_df = pd.merge(diversity_df, obs_df, on="sample_id", how="left")
            mapping = p9.aes(x="sample_id", y="value", fill=fill)
        else:
            mapping = p9.aes(x="sample_id", y="value")

        if nrow is not None:
            facet = p9.facet_wrap("~ measure", scales="free_y", nrow=nrow)
        else:
            facet = p9.facet_wrap("~ measure", scales="free_y")

        plot_obj = (
            p9.ggplot(diversity_df, mapping)
            + p9.geom_point(size=5)
            + p9.geom_line(p9.aes(group=1))
            + facet
            + p9.labs(
                title=f"[{self.adata.project_name}:{self.adata.data_subset}] Diversity Metrics",
                x="Sample",
                y="Diversity Value"
            )
            + economist_theme()
            + p9.theme(
                  axis_text_x=p9.element_text(rotation=45, ha="right"),
                  panel_spacing=0.8
              )
        )

        if nrow == 1:
            plot_obj = plot_obj + p9.theme(figure_size=(16, 3))
        else:
            plot_obj = plot_obj + p9.theme(figure_size=(8, 6))

        if label_order is not None and len(label_order) > 0:
            plot_obj += p9.scale_x_discrete(limits=label_order)

        if fill is not None:
            # Define a sorting key so that numeric‐like levels are sorted numerically
            def sort_key(x):
                try:
                    return int(x)
                except:
                    return x
            levels = sorted(diversity_df[fill].dropna().unique(), key=sort_key)
            # Define a default palette for up to 10 levels
            default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            manual_colors = {level: default_colors[i % len(default_colors)] for i, level in enumerate(levels)}
            plot_obj += p9.scale_fill_manual(values=manual_colors)

        return plot_obj

    def plot_diversity_v1(
        self,
        label_order: Optional[List[str]] = None,
        nrow: Optional[int] = None,
        fill: Optional[str] = None
    ) -> p9.ggplot:
        """
        Generate a faceted ggplot that displays diversity metrics for each sample.
        
        The plot displays distributions of:
          - Entropy
          - Gini
          - Simpson
          - Pielou Evenness (computed as Entropy divided by the log of the number
            of unique clones; defined as 1.0 when only one clone exists)
        
        Each measure is shown in its own facet with its own y-axis scale.
        
        Args:
            label_order: Optional list of sample IDs to order the x-axis.
            nrow: Optional integer for the number of facet rows.
                  For example, nrow=1 plots all 4 metrics in one row.
            fill: Optional column name (as a string) from self.adata.obs to map to the fill
                  aesthetic so that points are colored by a grouping variable.
        
        Returns:
            A plotnine ggplot object.
        """
        import numpy as np
        import pandas as pd

        # Retrieve the diversity measures (each expected to be a pd.Series indexed by sample_id)
        entropy_series = self.adata.entropy
        gini_series    = self.adata.gini
        simpson_series = self.adata.simpson

        # Ensure that the series' index names are set to "sample_id"
        if entropy_series.index.name is None:
            entropy_series.index.name = "sample_id"
        if gini_series.index.name is None:
            gini_series.index.name = "sample_id"
        if simpson_series.index.name is None:
            simpson_series.index.name = "sample_id"

        # Create DataFrames (long-format) for each measure.
        entropy_df = entropy_series.reset_index(name="value")
        entropy_df["measure"] = "Entropy"

        gini_df = gini_series.reset_index(name="value")
        gini_df["measure"] = "Gini"

        simpson_df = simpson_series.reset_index(name="value")
        simpson_df["measure"] = "Simpson"

        # Compute Pielou Evenness: for each sample, it equals Entropy / log(n_unique_clones),
        # or 1 if only one clone exists.
        n_unique = self.adata.clonotypes_df.groupby("sample_id")["combined_custom_id"].nunique()
        pielou_series = entropy_series.copy()
        condition = n_unique > 1
        pielou_series.loc[condition] = pielou_series.loc[condition] / np.log(n_unique[condition])
        pielou_series.loc[~condition] = 1.0

        if pielou_series.index.name is None:
            pielou_series.index.name = "sample_id"
        pielou_df = pielou_series.reset_index(name="value")
        pielou_df["measure"] = "Pielou Evenness"

        # Combine all metrics into one long-format DataFrame.
        diversity_df = pd.concat([entropy_df, gini_df, simpson_df, pielou_df], ignore_index=True)

        # If a fill mapping is requested, merge in the corresponding column from .obs.
        # Assumes that self.adata.obs has sample_id as its index.
        if fill is not None:
            obs_df = self.adata.obs.reset_index()[["sample_id", fill]]
            diversity_df = pd.merge(diversity_df, obs_df, on="sample_id", how="left")
            mapping = p9.aes(x="sample_id", y="value", fill=fill)
        else:
            mapping = p9.aes(x="sample_id", y="value")

        # Define the facet layout.
        if nrow is not None:
            facet = p9.facet_wrap("~ measure", scales="free_y", nrow=nrow)
        else:
            facet = p9.facet_wrap("~ measure", scales="free_y")

        # Build the ggplot object.
        plot_obj = (
            p9.ggplot(diversity_df, mapping)
            + p9.geom_point(size=5)
            + p9.geom_line(p9.aes(group=1))  # Connect the points – group=1 ensures a single line per facet.
            + facet
            + p9.labs(
                title=f"[{self.adata.project_name}:{self.adata.data_subset}] Diversity Metrics",
                x="Sample",
                y="Diversity Value"
            )
            + economist_theme()
            + p9.theme(
                  axis_text_x=p9.element_text(rotation=45, ha="right"),
                  panel_spacing=0.8  # Increase distance between subplots.
              )
        )

        # Adjust the overall figure size based on the layout.
        if nrow == 1:
            plot_obj = plot_obj + p9.theme(figure_size=(16, 3))
        else:
            plot_obj = plot_obj + p9.theme(figure_size=(8, 6))

        if label_order is not None and len(label_order) > 0:
            plot_obj += p9.scale_x_discrete(limits=label_order)

        return plot_obj
    

    
__all__ = ["TCRSeq", "economist_theme", "unit", "SampleStats"]
