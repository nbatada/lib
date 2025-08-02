import logging
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

try:
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
except ImportError:
    venn2 = venn3 = venn2_circles = venn3_circles = None

try:
    import cowpatch as cp
except ImportError:
    cp = None

from plotnine import theme_minimal, element_rect, element_line, element_blank, element_text

# configure module‐level logger
logger = logging.getLogger(__name__)


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


def economist_theme() -> p9.theme:
    """A clean, gray-background theme for plotnine."""
    return (
        theme_minimal(base_family="serif", base_size=12)
        + p9.theme(
            panel_background=element_rect(fill="#f5f5f5", color=None),
            panel_grid_major=element_line(color="#dddddd", size=0.5),
            panel_grid_minor=element_blank(),
            panel_border=element_blank(),
            axis_title=element_text(size=10, weight="bold", family="serif"),
            axis_text_x=element_text(size=9, family="serif"),
            axis_text_y=element_text(size=9, family="serif"),
            plot_title=element_text(size=14, weight="bold", family="serif"),
            legend_position="none"
        )
    )


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
            "-" * 46,
            f"\tshape: {self.clonotypes_df.shape}",
            f"\tUnique samples: {len(self.samples)}",
            f"\tUnique clones: {len(self.clones)}",
            f"\tcolumns: {', '.join(self.clonotypes_df.columns)}",
            "",
            "[.X] -- counts matrix (clones × samples)",
            "-" * 46,
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
            return stats.merge(self.obs.reset_index(), on='sample_id', how='right')
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
            logger.info(f"After filtering loci: {df.shape[0]} rows remain in {filename}")

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
        proc = df[df['productive']] if productive_only else df
        proc['TRA_custom_id'] = proc['chain_clonotype_id'].where(
            proc['locus'].isin(['TRA','TRAC']), ''
        )
        proc['TRB_custom_id'] = proc['chain_clonotype_id'].where(
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
        """
        dfs = []
        stats_list: List[SampleStats] = []

        for _, row in sample_info.iterrows():
            tcr = cls.from_file(row[filename_col], project_name=project_name, **kwargs)
            # capture per-sample stats
            stats_list.append(tcr._get_reading_stats_obj())
            # collect clonotype rows
            df = tcr.clonotypes_df.copy()
            df['sample_id'] = row[sample_id_col]
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        sample_stats_df = pd.DataFrame([asdict(s) for s in stats_list]).set_index('sample_id')

        obs = sample_info.set_index(sample_id_col)
        # join QC metrics
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
        filepath: str,
        index_col: str = "sample_id",
        qc_cols: Optional[List[str]] = None,
        **read_csv_kwargs
    ) -> None:
        """
        Read a per-sample UMI/QC table (e.g. your fastqtools output),
        index by `index_col`, optionally select `qc_cols`,
        and left-join into self.obs.

        USAGE:
            tcr = TCRSeq.from_sample_info(sample_info, …)
            tcr.add_umi_stats("all_samples_umi_count.tsv", qc_cols=["reads_total","reads_left_pct"])
            p = tcr.pl.plot_sequence_stats(
                x_col="reads_total",
                y_col="n_unique_clones",
                color_by="group"
            )
        """
        qc = pd.read_csv(filepath, **read_csv_kwargs)
        if index_col not in qc.columns:
            raise KeyError(f"add_umi_stats: '{index_col}' not found in {filepath}")
        qc = qc.set_index(index_col)
        if qc_cols is not None:
            missing = set(qc_cols) - set(qc.columns)
            if missing:
                raise KeyError(f"add_umi_stats: requested qc_cols {missing} not in file")
            qc = qc[qc_cols]
        if self.obs is None:
            self.obs = qc.copy()
        else:
            self.obs = self.obs.join(qc, how="left")
        logger.info(f"add_umi_stats: merged {filepath} into obs ({len(qc)} samples)")

class _TCRSeqPreprocessing:
    """Filtering and subsetting operations; produce new TCRSeq with updated history."""
    def __init__(self, adata: TCRSeq):
        self.adata = adata

    def filter_by_min_count(self, min_count: int) -> TCRSeq:
        counts = self.adata.clonotypes_df.groupby('combined_custom_id').size()
        keep = counts[counts >= min_count].index
        new_df = self.adata.clonotypes_df[self.adata.clonotypes_df['combined_custom_id']
                                           .isin(keep)].copy()
        new_obs = (self.adata.obs.loc[new_df['sample_id'].unique()]
                   if self.adata.obs is not None else None)
        new = TCRSeq(new_df, new_obs,
                     data_subset=self.adata.data_subset,
                     project_name=self.adata.project_name)
        new._record_step(f"min_count≥{min_count}")
        return new

    def filter_shared_clones(self, min_samples: int, group: Optional[str] = None) -> TCRSeq:
        if group and self.adata.obs is not None and 'group' in self.adata.obs:
            samples = self.adata.obs[self.adata.obs['group']==group].index.tolist()
        else:
            samples = self.adata.samples
        counts = (self.adata.clonotypes_df[self.adata.clonotypes_df['sample_id']
                  .isin(samples)]
                  .groupby('combined_custom_id')['sample_id']
                  .nunique())
        keep = counts[counts >= min_samples].index
        new_df = self.adata.clonotypes_df[self.adata.clonotypes_df['combined_custom_id']
                                           .isin(keep)].copy()
        new_obs = (self.adata.obs.loc[new_df['sample_id'].unique()]
                   if self.adata.obs is not None else None)
        new = TCRSeq(new_df, new_obs,
                     data_subset=self.adata.data_subset,
                     project_name=self.adata.project_name)
        new._record_step(f"shared≥{min_samples}samples")
        return new

    def subset_by_group(self, group_value: str) -> TCRSeq:
        if self.adata.obs is None or 'group' not in self.adata.obs:
            raise ValueError("Missing 'group' in obs")
        samples = self.adata.obs[self.adata.obs['group']==group_value].index.tolist()
        new_df = self.adata.clonotypes_df[self.adata.clonotypes_df['sample_id']
                                           .isin(samples)].copy()
        new_obs = self.adata.obs.loc[samples].copy()
        new = TCRSeq(new_df, new_obs,
                     data_subset=f"{self.adata.data_subset}_group:{group_value}",
                     project_name=self.adata.project_name)
        new._record_step(f"group={group_value}")
        return new

    def subset_by_chain(self, chain: str) -> TCRSeq:
        valid = {"TRA":["TRA","TRAC"], "TRB":["TRB","TRBC"]}
        if chain not in valid:
            raise ValueError("chain must be 'TRA' or 'TRB'")
        new_df = self.adata.clonotypes_df[self.adata.clonotypes_df['locus']
                                           .isin(valid[chain])].copy()
        new_obs = (self.adata.obs.loc[new_df['sample_id'].unique()]
                   if self.adata.obs is not None else None)
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
        cm = self.get_counts_matrix()
        df_mean = cm.mean(axis=1).reset_index(name='mean_count')
        return df_mean.sort_values('mean_count', ascending=False).head(n_top)


class _TCRSeqPlotting:
    """All ggplot2‐style plotting via plotnine and matplotlib."""
    def __init__(self, adata: TCRSeq):
        self.adata = adata


    def plot_sample_stats(
        self,
        split_by: Optional[str] = None,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None,
        label_order: Optional[List[str]] = None
    ) -> Union[p9.ggplot, tuple]:
        """Per-sample barplots of basic metrics and TRA/TRB clone counts."""
        stats = self.adata.tl.get_sample_stats()
        melt = stats.drop(columns=['n_unique_clones'], errors='ignore').melt(
            id_vars=['sample_id','group'] if split_by=='group' else ['sample_id'],
            var_name='metric', value_name='value'
        )
        color = split_by if split_by in melt.columns and split_by!='sample_id' else None
        aes1 = p9.aes(x='sample_id', y='value', fill=color)
        p1 = (
            p9.ggplot(melt, aes1)
            + p9.geom_col()
            + p9.facet_wrap('~metric', scales='free_y')
            + p9.labs(title="Per-sample metrics")
            + economist_theme()
            + p9.theme(panel_spacing=0.4)
        )
        if label_order:
            p1 += p9.scale_x_discrete(limits=label_order)

        df = self.adata.clonotypes_df
        tra = (
            df[df['TRA_custom_id']!='']
            .drop_duplicates(['sample_id','TRA_custom_id'])
            .groupby('sample_id').size()
            .reset_index(name='clone_count')
        )
        tra['chain'] = 'TRA'
        trb = (
            df[df['TRB_custom_id']!='']
            .drop_duplicates(['sample_id','TRB_custom_id'])
            .groupby('sample_id').size()
            .reset_index(name='clone_count')
        )
        trb['chain'] = 'TRB'
        counts = pd.concat([tra, trb], ignore_index=True)
        aes2 = p9.aes(x='sample_id', y='clone_count', fill='chain')
        p2 = (
            p9.ggplot(counts, aes2)
            + p9.geom_col()
            + p9.labs(title="Unique TRA/TRB counts")
            + economist_theme()
            + p9.theme(panel_spacing=0.4)
        )
        if label_order:
            p2 += p9.scale_x_discrete(limits=label_order)

        if cp:
            import numpy as np
            patch = cp.patch(p1, p2)
            patch += cp.layout(design=np.array([0,1]).reshape(1,2))
            return patch
        return p1, p2


    def plot_shared_clone(self, split_by: Optional[str] = None, n_top: int = 10) -> p9.ggplot:
        """Bubble-plot of the top N clones shared across samples."""
        shared = self.adata.tl.get_shared_clones(n_top)
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
        color = split_by if (split_by and self.adata.obs is not None and split_by in self.adata.obs) else 'sample_id'
        aes = p9.aes(
            x='sample_id',
            y='combined_custom_id',
            size='clone_count',
            color=color
        )
        p = (
            p9.ggplot(cm, aes)
            + p9.geom_point()
            + p9.labs(title=f"Top {n_top} shared clones")
            + economist_theme()
            + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
        )
        return p


    def plot_clonal_expansion(
        self,
        bins: List[float] = None,
        labels: List[str] = None,
        normalize: bool = False,
        label_order: Optional[List[str]] = None
    ) -> p9.ggplot:
        """Stacked barplot of clone-size bins (optionally normalized)."""
        df = self.adata.clonotypes_df
        if bins is None:
            bins = [0.5, 1.5, 2.5, 5.5, 20.5, np.inf]
        if labels is None:
            labels = ["1", "2", "3-5", "6-20", ">20"]

        if 'consensus_count' in df.columns:
            cc = (
                df.groupby(['sample_id','combined_custom_id'])['consensus_count']
                .sum().reset_index()
            )
        else:
            cc = (
                df.groupby(['sample_id','combined_custom_id'])
                .size().reset_index(name='consensus_count')
            )

        cc['expansion_bin'] = pd.cut(
            cc['consensus_count'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        exp_df = (
            cc.groupby(['sample_id','expansion_bin'])
            .size().reset_index(name='n_clones')
        )

        if normalize:
            total = exp_df.groupby('sample_id')['n_clones'].transform('sum')
            exp_df['proportion'] = exp_df['n_clones'] / total
            y_val, ylab = 'proportion', "Proportion of Clones"
        else:
            y_val, ylab = 'n_clones', "Number of Clones"

        aes = p9.aes(x='sample_id', y=y_val, fill='expansion_bin')
        p = (
            p9.ggplot(exp_df, aes)
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
        if label_order:
            p += p9.scale_x_discrete(limits=label_order)
        return p


    def plot_venn(self, group: Optional[str] = None) -> plt.Figure:
        """Venn diagram of overlaps for 2 or 3 samples (or plain text otherwise)."""
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
            plt.figure(figsize=(8, 6))
            venn2(
                (len(set1 - set2), len(set2 - set1), len(set1 & set2)),
                set_labels=(s1, s2)
            )
            venn2_circles(
                (len(set1 - set2), len(set2 - set1), len(set1 & set2)),
                linestyle='dashed', linewidth=1
            )
            plt.title(f"[{self.adata.project_name}:{self.adata.data_subset}]", fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

        # 3-way Venn
        elif len(samples) == 3 and venn3:
            s1, s2, s3 = samples
            set1 = set(cm.index[cm[s1] > 0])
            set2 = set(cm.index[cm[s2] > 0])
            set3 = set(cm.index[cm[s3] > 0])
            A = len(set1 - set2 - set3)
            B = len(set2 - set1 - set3)
            AB = len((set1 & set2) - set3)
            C = len(set3 - set1 - set2)
            AC = len((set1 & set3) - set2)
            BC = len((set2 & set3) - set1)
            ABC = len(set1 & set2 & set3)
            plt.figure(figsize=(8, 6))
            venn3((A, B, AB, C, AC, BC, ABC), set_labels=(s1, s2, s3))
            venn3_circles((A, B, AB, C, AC, BC, ABC), linestyle='dashed', linewidth=1)
            plt.title(f"[{self.adata.project_name}:{self.adata.data_subset}]", fontsize=14, fontweight="bold")
            plt.tight_layout()
            return plt.gcf()

        # fallback
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


    def plot_upset(self) -> plt.Figure:
        """Upset plot of clone presence/absence across samples."""
        cm = self.adata.tl.get_counts_matrix()
        pres = cm > 0
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
                title=f"CDR3 length distribution{(' '+chain) if chain else ''}",
                x="CDR3 length", y="Count"
            )
            + economist_theme()
            + p9.scale_x_continuous(limits=(0, max_len + 1))
        )
        return p


    def plot_gene_usage(
        self,
        gene: str = 'v_call',
        top_n: int = 10,
        split_by: Optional[str] = None
    ) -> p9.ggplot:
        """Barplot of the top N V or J genes, optionally split by obs[group]."""
        df = self.adata.clonotypes_df.copy()
        if split_by and self.adata.obs is not None and split_by in self.adata.obs.columns:
            df = df.merge(
                self.adata.obs.reset_index()[['sample_id', split_by]],
                on='sample_id', how='left'
            )
            grouping = [split_by, gene]
        else:
            grouping = [gene]

        top = df.groupby(grouping).size().reset_index(name='count')
        if split_by:
            top = (
                top
                .groupby(split_by)
                .apply(lambda d: d.nlargest(top_n, 'count'))
                .reset_index(drop=True)
            )
        else:
            top = top.nlargest(top_n, 'count')

        max_count = int(top['count'].max() or 0)
        aes = p9.aes(x=gene, y='count', fill=split_by) if split_by else p9.aes(x=gene, y='count')
        p = (
            p9.ggplot(top, aes)
            + p9.geom_col(position="dodge")
            + p9.labs(title=f"Top {top_n} {gene} usage", x=gene, y="Count")
            + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
            + economist_theme()
            + p9.scale_y_continuous(limits=(0, max_count * 1.05))
        )
        return p


    def plot_rank_abundance(
        self,
        sample_id: str,
        split_by: Optional[str] = None
    ) -> p9.ggplot:
        """Rank–abundance curve (clone rank vs count) on log–log scales."""
        cm = self.adata.X[sample_id].sort_values(ascending=False).reset_index(name='count')
        cm['rank'] = np.arange(1, len(cm) + 1)
        max_rank = int(cm['rank'].max() or 1)
        max_count = float(cm['count'].max() or 1)
        min_count = float(cm['count'][cm['count']>0].min() or 1)

        aes = p9.aes(x='rank', y='count')
        if split_by and self.adata.obs is not None and split_by in self.adata.obs.columns:
            cm = cm.merge(
                self.adata.obs.reset_index()[['sample_id', split_by]],
                on='sample_id', how='left'
            )
            aes = p9.aes(x='rank', y='count', color=split_by)

        p = (
            p9.ggplot(cm, aes)
            + p9.geom_line()
            + p9.labs(title=f"Rank–abundance: {sample_id}", x="Rank", y="Count")
            + economist_theme()
            + p9.scale_x_log10(limits=(1, max_rank))
            + p9.scale_y_log10(limits=(min_count * 0.9, max_count * 1.1))
        )
        return p

    def plot_sequence_stats(
        self,
        x_col: str = "reads_total",
        y_col: str = "umi_total",
        color_by: Optional[str] = None,
        log_scale: bool = True
    ) -> Optional[p9.ggplot]:
        """
        Scatter x_col vs. y_col from self.adata.obs (must exist!),
        color by an optional obs column, and (optionally) log-scale both axes.
        """
        df = self.adata.obs.copy()
        for c in (x_col, y_col):
            if c not in df.columns:
                logger.warning(f"plot_sequence_stats: '{c}' not in obs; skipping plot")
                return None
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[x_col, y_col])

        aes_kwargs = {"x": x_col, "y": y_col}
        if color_by and color_by in df.columns:
            aes_kwargs["color"] = color_by

        p = (
            p9.ggplot(df, p9.aes(**aes_kwargs))
            + p9.geom_point()
            + p9.labs(x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            + economist_theme()
        )
        if log_scale:
            p += p9.scale_x_log10() + p9.scale_y_log10()
        return p

def help():
    """
    import tcrseq
    tcrseq.help()

    Quickstart guide for the TCRSeq module.

    1) Create a sample_info DataFrame with at least:
         - sample_id
         - a column of AIRR/CellRanger filenames
         - (optional) metadata like 'group' for coloring.

    2) Load:
         tcr = TCRSeq.from_sample_info(
             sample_info,
             filename_col="your_filename_col",
             sample_id_col="sample_id",
             project_name="MyProject"
         )

    3) (Optional) Merge in FASTQ‐derived QC:
         tcr.add_umi_stats(
             filepath="all_samples_umi_stats.tsv",
             index_col="sample_id",
             qc_cols=["reads_total","reads_left_pct","umi_total"]
         )

    4) Scatter QC vs clonotypes:
         p = tcr.pl.plot_sequence_stats(
             x_col="reads_total",
             y_col="n_unique_clones",
             color_by="group",
             log_scale=True
         )
         p.save("qc_vs_clones.png", dpi=150)

    5) Per-group diversity & gene-usage:
         plots = tcr.compare_groups("group")
         # plots["A"][0] is diversity for group A, etc.

    6) Summaries & exports:
         print(tcr.summary())
         adata = tcr.to_anndata()

    All further plotting is under tcr.pl (e.g. plot_gene_usage, plot_cdr3_length, …).
    """
    print(help.__doc__)

__all__ = ["TCRSeq", "economist_theme", "unit", "SampleStats"]
