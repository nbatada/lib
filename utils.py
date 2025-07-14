# Updated on 14 July 2025
import re
import pandas as pd
from collections.abc import Iterable
import plotnine as p9
import numpy as np
from datetime import datetime
import subprocess
import sys
import matplotlib.pyplot as plt
import itertools
from graphviz import Digraph # Make sure graphviz is imported




def sort_natural(alist):
    def natural_sort_key(s):
        def convert_part(text):
            try:
                return int(text)
            except ValueError:
                try:
                    return float(text)  # Convert to float for decimal numbers
                except ValueError:
                    return text.lower()
        return [convert_part(text) for text in re.split('([0-9]+\.?[0-9]*)', s)] # Improved regex
    return sorted(alist, key=natural_sort_key)


def parse_sample_id(s):
    return re.findall(r'[pP][0-9][0-9][eE][0-9][0-9][sS][0-9][0-9]', s, re.IGNORECASE)[0]

def theme_nizar():
    # https://plotnine.org/reference/theme.html
    # https://plotnine.org/reference/
    #return ( p9.theme( axis_text_x=p9.element_text(rotation=25, hjust=1), figure_size=(4,4), legend_position='top') + p9.guides(fill = p9.guide_legend(ncol = 1))) #+ p9.theme_clean()
    return ( p9.theme(panel_background=p9.element_rect(fill="white"),
                      panel_grid_major=p9.element_line(linetype='dotted',color='grey', size=0.2),
                      panel_grid_minor=p9.element_line(linetype='dotted',color='grey', size=0.5),
                      panel_border=p9.element_rect(color='grey', fill=None),
                      legend_title=p9.element_text(weight="bold"),
                      legend_direction="horizontal",
                      legend_text=p9.element_text(size=6),
                      axis_text_x=p9.element_text(rotation=25, hjust=1, size=8),
                      axis_text_y=p9.element_text(size=8),
                      legend_position='top',
                      strip_text=p9.element_text(size=6))
             )


def outlier_rows(df, threshold=2, min_mad_ratio=0.05, diff_multiplier=2):
    """
    One robust unsupervised method for outlier detection in small datasets is to compute the 
    modified z-scores using the median absolute deviation (MAD). Instead of using the mean and 
    standard deviation (which can be skewed by extreme values), you calculate each data point’s 
    deviation from the median, scale it by the MAD, and then flag any point as an outlier if its 
    modified z-score (0.6745 × (x – median)⁄MAD) exceeds a chosen threshold (often around 3.5 or 5).

    This approach solves the weakness of MAD when sampels are very homogenous (MAD is very small) and then
    small differences are amplified and flagged as outliers falsely.
    
    """
    def helper_func_row_outliers(row):
        m = row.median()
        mad = np.median(np.abs(row - m))
        # Ensure the MAD is not too small by enforcing a minimum floor based on the median
        effective_mad = mad if mad >= min_mad_ratio * m else min_mad_ratio * m
        # Compute the modified z-score for each value in the row
        mod_z = 0.6745 * (row - m) / effective_mad
        # Calculate the coefficient of variation (CV)
        cv = row.std() / row.mean() if row.mean() != 0 else 0
        # Only flag values that are both statistically extreme and have a meaningful absolute difference
        diff_filter = np.abs(row - m) > diff_multiplier * (cv * m)
        outlier_mask = (np.abs(mod_z) > threshold) & diff_filter
        return row.index[outlier_mask].tolist()

    return df.apply(helper_func_row_outliers, axis=1)


def sampleinfo2graph(df, variables, output_path='experimental_design_graph'):
    dot = Digraph(format='pdf', name='Sample_Info_Graph', graph_attr={'rankdir': 'LR'})
    
    # Map colors to the variables (only used for the *values* now, not separate variable nodes)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrod', 'lightpink', 'lightgray', 'lightcyan', 'lightsalmon']
    
    if len(variables) > len(colors):
        raise ValueError("Not enough colors to represent variables uniquely.")
    
    previous_value_nodes = None # Stores the set of nodes from the previous variable's values

    # Iterate through each variable
    for idx, var in enumerate(variables):
        unique_values = df[var].unique()
        fill_color = colors[idx] # Color for the value nodes of the current variable
        
        current_value_nodes = [] # List to store the new prefixed node names for this variable
        
        for value in unique_values:
            # Create the prefixed node name and label
            node_id = f"{var}_{value}" # Internal ID remains 'variable_value'
            #node_label = f"[{var}]\n{str(value)}" # Display label is 'variable:value'
            #node_label = f'<<B>{var}</B><BR/><I>{str(value)}</I>>'
            node_label = f'<<FONT POINT-SIZE="10"><B>{var}</B></FONT><BR/><FONT POINT-SIZE="12"><I>{str(value)}</I></FONT>>'
            dot.node(node_id, node_label, shape='ellipse', style='filled', fillcolor=fill_color)
            current_value_nodes.append(node_id)
        
        # Connect nodes from the previous variable to the current variable's nodes
        if previous_value_nodes:
            # For each unique value from the *previous* variable
            for prev_node_id in previous_value_nodes:
                # Find the actual unique values for this specific 'prev_node_id'
                # This logic assumes a full connection from *all* previous level nodes to *all* current level nodes.
                # If specific connections are needed based on df_info rows, that would require more complex grouping.
                
                # Given the previous function's behavior, it connected every node from the previous
                # layer to every node in the current layer. We will replicate that.
                for current_node_id in current_value_nodes:
                    dot.edge(prev_node_id, current_node_id)
        
        # Update previous_value_nodes for the next iteration
        previous_value_nodes = current_value_nodes
    
    pdf_path = output_path.removesuffix('.pdf')
    dot.render(pdf_path, cleanup=True)
    
    return pdf_path+'.pdf'
#_
def whos():
    variables = globals() # or locals() for local variables within the function
    #print("Variables in current buffer larger than 1Gb (and their Sizes):\n")
    total_size_gb = 0
    large_variables_found = False # Flag to check if any large variables were found

    variable_sizes = {} # Store variable sizes for total calculation

    for name, value in variables.items():
        if not name.startswith('_') and name not in ['__builtin__', '__builtins__', '__name__', '__doc__', '__package__', 'variables', 'name', 'value', 'size_bytes', 'size_gb', 'large_variables_found']: # Exclude special and internal variables
            size_bytes = sys.getsizeof(value)
            size_gb = size_bytes / (1024 ** 3) # Convert bytes to Gb
            variable_sizes[name] = size_gb # Store size for total calculation
            if size_gb > 1.0: # Check if size is greater than 1Gb
                print(f"Variable: '{name}', Size: {size_gb:.1f} Gb")
                large_variables_found = True # Set flag if at least one large variable is found

    total_size_gb = sum(variable_sizes.values()) # Calculate total size from stored sizes
    print(f"\nTotal memory in use by listed variables: {total_size_gb:.1f} Gb") # Updated line


def filter_rows_by_pct_zeros(df, MIN_VAL=1, threshold_fraction=0.75):
    min_nonzero_columns = int(threshold_fraction * df.shape[1])
    return df[df.ge(MIN_VAL).sum(axis=1) >= min_nonzero_columns] # ge is >=


def run_cmd_capture_output(command):
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
        return result.stdout  # Capture and return stdout as a string
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)  # Print stderr if an error occurs
        return None


def current_time():
    now = datetime.now()
    #current_time_str = now.strftime("%H:%M:%S (%Y-%m-%d)")
    current_time_str = now.strftime("%H:%M (%m/%d)")
    return(current_time_str)

def tabulate(df, r, c, normalize_by=None):
    '''
    r and c are column names in df
    in the output: r-values will be in rownames/index and c-values will be colnames
    
    USAGE: tabulate(df_tregs, 'group', 'authors_cell_type_level_3__treg_subclustering')

    normalize_by=None, rowsum or colsum

    '''
    u = df.groupby([r, c]).size().reset_index(name='count') # long
    
    T = u.pivot(index=r, columns=c, values='count') # compact r x c 
    
    #T['rowsum'] = T.sum(axis=1).astype('int')
    rowsum = T.sum(axis=1) #.astype('int')
    
    #T.loc['colsum'] = T.sum(axis=0).astype('int')
    colsum = T.sum(axis=0) #.astype('int')


    T.index.name = None
    T.columns.name = None
    T=T.fillna(0)
    if (normalize_by=='rowsum') | (normalize_by=='row'):
        #T=x.div(T[normalize_by], axis=0)*100
        T=T.div(rowsum, axis=0)*100
    elif (normalize_by=='colsum')| (normalize_by=='column'):
        #T=x.div(T.loc[normalize_by], axis=1)*100
        T=T.div(colsum, axis=1)*100
    elif (normalize_by=='total'):
        T = T / T.to_numpy().sum()
    return T.astype('int')



from collections import Counter

def clean_str_values(x):
    if isinstance(x, pd.Series) or isinstance(x, pd.Index):
        # Convert to a list for mutable operations
        values_list = list(x.str.lower().str.replace(' ', '_').str.replace("'", '').str.replace('-', '').str.replace('(', '').str.replace(')', '').str.replace('.', '_').str.replace(':', '_'))
        
        counts = Counter(values_list)
        duplicates = [item for item, count in counts.items() if count > 1]
        
        # Resolve duplicates by appending a suffix
        for dup in duplicates:
            suffix_count = 1
            for i, val in enumerate(values_list):
                if val == dup:
                    if suffix_count > 1:
                        values_list[i] = f"{val}_{suffix_count}"
                    suffix_count += 1
        
        # Return as the original type if it was an Index, otherwise as Series
        if isinstance(x, pd.Index):
            return pd.Index(values_list)
        else:
            return pd.Series(values_list, index=x.index)

    elif isinstance(x, list):
        cleaned_values = [str(item).lower().replace(' ', '_').replace("'", '').replace('-', '').replace('(', '').replace(')', '').replace('.', '_').replace(':', '_') for item in x]
        
        counts = Counter(cleaned_values)
        duplicates = [item for item, count in counts.items() if count > 1]
        
        for dup in duplicates:
            suffix_count = 1
            for i in range(len(cleaned_values)):
                if cleaned_values[i] == dup:
                    if suffix_count > 1:
                        cleaned_values[i] = f"{cleaned_values[i]}_{suffix_count}"
                    suffix_count += 1
        return cleaned_values
    else:
        raise TypeError("Input must be a Pandas Series, Index, or a list.")
    


def jupyter_header():
    return '''
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
%matplotlib inline
rcParams['figure.figsize']=(7,7)


import pandas as pd
pd.set_option('display.max_columns', 120)
pd.set_option('max_colwidth', 400)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = "{:.2f}".format

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

   
import scanpy as sc
sc.settings.set_figure_params(dpi=100, facecolor="white")
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.frameon = True
sc.settings.autoshow = True # needs inline
sc.settings.n_jobs = 4
import scanpy.external as sce # harmony

import numpy as np
import plotnine as p9
from collections import Counter
from importlib import reload

#---
import sys
import os
HOME=f'{os.path.expanduser("~")}'
sys.path.insert(0,f'{HOME}/lib')    
import utils
import utils_scrnaseq
import utils_tcrseq
    '''


def top5(df):
    results = {}
    
    # Iterate through each column of the DataFrame
    for col_name in df.columns:
        # Access the column data. This can be a Series or, in specific MultiIndex cases, a DataFrame.
        column_data_series_or_df = df[col_name]

        # Initialize value_counts and total_count for current column
        current_value_counts = None
        current_total_count = None
        
        is_processable = True
        
        if isinstance(column_data_series_or_df, pd.Series):
            current_value_counts = column_data_series_or_df.value_counts(dropna=False)
            current_total_count = len(column_data_series_or_df)
            has_nan = column_data_series_or_df.isnull().any()
        elif isinstance(column_data_series_or_df, pd.DataFrame):
            # This happens if `col_name` is a top-level in a MultiIndex column,
            # and `df[col_name]` returns a DataFrame with multiple sub-columns.
            if column_data_series_or_df.shape[1] == 1:
                # If it's a DataFrame with a single column, convert it to a Series
                series_to_process = column_data_series_or_df.iloc[:, 0]
                current_value_counts = series_to_process.value_counts(dropna=False)
                current_total_count = len(series_to_process)
                has_nan = series_to_process.isnull().any()
            else:
                # It's a DataFrame with multiple columns, cannot process as a single column
                column_output = ["N/A (Multi-column)" for _ in range(5)]
                # Check for NaNs across all columns in this multi-column DataFrame slice
                has_nan = column_data_series_or_df.isnull().any().any()
                column_output.append(f"does_it_have_nan_values: {'Yes' if has_nan else 'No'}")
                results[col_name] = column_output
                is_processable = False
        else:
            # Handle other unexpected types if necessary
            column_output = ["N/A (Unexpected type)" for _ in range(5)]
            column_output.append(f"does_it_have_nan_values: N/A")
            results[col_name] = column_output
            is_processable = False

        if is_processable:
            column_output = []
            # Get top 5 most frequent values and their counts/percentages
            for i in range(5):
                if i < len(current_value_counts):
                    value = current_value_counts.index[i]
                    count = current_value_counts.iloc[i]
                    percentage = (count / current_total_count) * 100
                    column_output.append(f"{value} (Count: {count}, Pct: {percentage:.2f}%)")
                else:
                    column_output.append("") # Add empty string if fewer than 5 unique values
            
            # Check for NaN/NA values
            column_output.append(f"does_it_have_nan_values: {'Yes' if has_nan else 'No'}")
            results[col_name] = column_output

    # Create a DataFrame from the results
    # The index will be for the top 5 values plus the NaN check row
    index_names = [f"Top {i+1} Most Frequent" for i in range(5)] + ["NaN Check"]
    output_df = pd.DataFrame(results, index=index_names)
    
    return output_df

##=
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import itertools
import pandas as pd

def plot_venn_diagram(df, colnames):
    num_cols = len(colnames)
    df_binary = (df[colnames] > 0).astype(int)
    segment_data = []
    segment_counts = {}
    col_masks = [df_binary[col].astype(bool) for col in colnames]
    for i in range(1, 2**num_cols):
        binary_pattern = bin(i)[2:].zfill(num_cols)
        current_mask = pd.Series(True, index=df_binary.index)
        segment_name_parts = []
        for j, col_name in enumerate(colnames):
            if binary_pattern[j] == '1':
                current_mask &= col_masks[j]
                segment_name_parts.append(col_name)
            else:
                current_mask &= ~col_masks[j]
        count = df_binary[current_mask].shape[0]
        if count > 0:
            segment_key = binary_pattern
            segment_counts[segment_key] = count
            if len(segment_name_parts) == num_cols:
                segment_desc = " & ".join(segment_name_parts)
            elif len(segment_name_parts) == 1:
                segment_desc = segment_name_parts[0] #+ " only"
            else:
                segment_desc = " & ".join(segment_name_parts)
            segment_data.append({
                'Segment': segment_desc,
                'Count': count,
                'Percentage': 0
            })
    union_mask = (df_binary[colnames].sum(axis=1) > 0)
    total_elements_in_union = df_binary[union_mask].shape[0]
    for seg_dict in segment_data:
        seg_dict['Percentage'] = (seg_dict['Count'] / total_elements_in_union * 100) if total_elements_in_union > 0 else 0
    segment_table = pd.DataFrame(segment_data)
    segment_table = segment_table.sort_values(by='Count', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    venn_obj = None
    if num_cols == 2:
        venn_subsets = (segment_counts.get('10', 0), segment_counts.get('01', 0), segment_counts.get('11', 0))
        venn_obj = venn2(subsets=venn_subsets, set_labels=colnames, ax=ax)
        label_order_ids = ['10', '01', '11']
    elif num_cols == 3:
        venn_subsets = (segment_counts.get('100', 0), segment_counts.get('010', 0), segment_counts.get('110', 0),
                        segment_counts.get('001', 0), segment_counts.get('101', 0), segment_counts.get('011', 0),
                        segment_counts.get('111', 0))
        venn_obj = venn3(subsets=venn_subsets, set_labels=colnames, ax=ax)
        label_order_ids = ['100', '010', '110', '001', '101', '011', '111']
    for label_id in label_order_ids:
        if venn_obj.get_label_by_id(label_id):
            count = segment_counts.get(label_id, 0)
            percentage = 0.0
            temp_segment_name_parts = []
            for j, col_name in enumerate(colnames):
                if label_id[j] == '1':
                    temp_segment_name_parts.append(col_name)
            if len(temp_segment_name_parts) == num_cols:
                temp_segment_desc = " & ".join(temp_segment_name_parts)
            elif len(temp_segment_name_parts) == 1:
                temp_segment_desc = temp_segment_name_parts[0] #+ " only"
            else:
                temp_segment_desc = " & ".join(temp_segment_name_parts)
            matched_row = segment_table[segment_table['Segment'] == temp_segment_desc]
            if not matched_row.empty:
                percentage = matched_row['Percentage'].iloc[0]
            venn_obj.get_label_by_id(label_id).set_text(f'{count}\n({percentage:.1f}%)')
            venn_obj.get_label_by_id(label_id).set_fontsize(10)
    ax.set_title(f"Venn Diagram of Clonotype Overlap in {', '.join(colnames)}", fontsize=14)
    plt.close(fig)
    return fig, segment_table

##=

#--------------------
import types


tl = types.SimpleNamespace(
    summarize_df=summarize_df,
    )

pl = types.SimpleNamespace(
    theme_nizar=theme_nizar,
    sampleinfo2graph=sampleinfo2graph,
    plot_venn_diagram=plot_venn_diagram,
    )

string = types.SimpleNamespace(
    suffix_remove=suffix_remove,
    prefix_add=prefix_add,
    clean_str_values=clean_str_values,
    parse_sample_id=parse_sample_id,
    )
