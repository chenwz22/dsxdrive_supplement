import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
from Levenshtein import hamming


INPUT_FILE = 'MND_Alleles_frequency_table.txt'
output_csv = 'MND_Alleles_Merged_Results.csv'
output_plot = 'MND_Alleles_Frequency_Plot.png'

START_TARGET, END_TARGET = 60, 200  # focus region
DISPLAY_START, DISPLAY_END = 60, 200 # plot
THRESHOLD = 0.1                      
MIN_READ_FREQ = 1/500                # 0.002, 0.2% threshold for filtering



def main():
    
    df = pd.read_csv(INPUT_FILE, sep='\t')
    df = df[df['Aligned_Sequence'].str.replace('-', '', regex=False).str.len() > 0].copy()
    ref_full = df.iloc[0]['Reference_Sequence']
    
    def get_masked_seq(aln):
        res = list(ref_full)
        res[START_TARGET:END_TARGET] = list(aln[START_TARGET:END_TARGET])
        return "".join(res)

    df['Masked_Sequence'] = df['Aligned_Sequence'].apply(get_masked_seq)
    df_grouped = df.groupby('Masked_Sequence').agg({
        '#Reads': 'sum',
        'Read_Status': 'first'
    }).reset_index()
    
    df_grouped = df_grouped.sort_values('#Reads', ascending=False).reset_index(drop=True)
    seqs = df_grouped['Masked_Sequence'].tolist()
    counts = df_grouped['#Reads'].tolist()
    n = len(seqs)
    merged = [False] * n

    #clustering by calculating Hamming distance
    start_t = time.time()
    half = len(seqs[0]) // 2
    index_map = {}
    for idx, s in enumerate(seqs):
        p1, p2 = s[:half], s[half:]
        for p in (p1, p2):
            if p not in index_map: index_map[p] = []
            index_map[p].append(idx)

    for i in range(n):
        if merged[i]: continue
        s_i = seqs[i]
        p1, p2 = s_i[:half], s_i[half:]
        candidates = set(index_map.get(p1, []) + index_map.get(p2, []))
        for j in candidates:
            if i == j or merged[j]: continue
            if hamming(s_i, seqs[j]) <= 1: 
                counts[i] += counts[j]
                merged[j] = True
    
    df_grouped['#Reads_Final'] = counts
    df_final = df_grouped[~pd.Series(merged)].copy()

    # filtering: counting frequency, and exclude the alleles which frequency < 1/500 
    temp_total = df_final['#Reads_Final'].sum()  
    df_final = df_final[df_final['#Reads_Final'] / temp_total >= MIN_READ_FREQ].copy()
    
    # recaculating the frequency
    final_total_reads = df_final['#Reads_Final'].sum()
    df_final['%Reads_Final'] = (df_final['#Reads_Final'] / final_total_reads) * 100

    df_final.to_csv(output_csv, index=False)
    
    # plot
    df_plot = df_final[df_final['%Reads_Final'] > THRESHOLD].copy().reset_index(drop=True)
    

def plot_alleles(df, ref_full):
    # sort
    df = df.sort_values('#Reads_Final', ascending=False).reset_index(drop=True)
    
    n = len(df)
    ref_seg = ref_full[DISPLAY_START:DISPLAY_END]
    fig, ax1 = plt.subplots(figsize=(20, max(4, 0.45 * (n + 1))))
    
    colors = {
    'A': '#009E73', # A
    'T': '#D55E00', # T 
    'C': '#0072B2', # C
    'G': '#F0E442', # G
    '-': '#F0F0F0', # delete
    'N': '#999999'  # unknown
}
    text_props = dict(ha='center', va='center', fontsize=11, fontweight='bold')
    
    ref_y = n  
    for j, c in enumerate(ref_seg):
        f_col = colors.get(c, '#bdc3c7')
        rect = patches.Rectangle((j, ref_y), 1, 0.8, facecolor=f_col, alpha=0.3)
        ax1.add_patch(rect)
        ax1.text(j + 0.5, ref_y + 0.4, c, **text_props)

    for i, row in df.iterrows():
        s_seg = row['Masked_Sequence'][DISPLAY_START:DISPLAY_END]
        y = n - 1 - i 
        
        for j, c in enumerate(s_seg):
            f_col = colors.get(c, '#bdc3c7')
            is_diff = (c != ref_seg[j]) and (c != '-')
            e_col = 'none'
            l_width = 1.0 if is_diff else 0
            
            rect = patches.Rectangle((j, y), 1, 0.8, facecolor=f_col, 
                                     edgecolor=e_col, linewidth=l_width)
            ax1.add_patch(rect)            
            if c == '-':
                char_col = 'black'
                ax1.text(j + 0.5, y + 0.4, c, color=char_col, **text_props)

        freq_val = row['%Reads_Final']
        ax1.text(len(ref_seg) + 0.8, y + 0.4, f"{freq_val:.2f}%", 
                 va='center', ha='left', fontsize=13, fontweight='bold', color='#2c3e50')

    ax1.set_xlim(0, len(ref_seg) + 10) 
    ax1.set_ylim(-0.5, n + 1.2)
    
    n = len(df)
    ref_seg = ref_full[DISPLAY_START:DISPLAY_END]
    seq_len = len(ref_seg)
    target_intervals = [(88, 107), (123, 142), (146, 165)] #red boxes to show target sites
    for start, end in target_intervals:
        rect_x_start = start - DISPLAY_START
        rect_width = end - start + 1
        if rect_x_start >= 0 and rect_x_start + rect_width <= seq_len:
            target_rect = patches.Rectangle(
                (rect_x_start, ref_y - 0.05), rect_width, 0.8, 
                linewidth=2, edgecolor='red', facecolor='none', zorder=10
            )
            ax1.add_patch(target_rect)
    cut_sites = [91, 140, 163]
    for site in cut_sites:
        line_x = site - DISPLAY_START
        if 0 <= line_x <= seq_len: #cleavage sites
            ax1.axvline(x=line_x, color='black', linestyle='-', linewidth=1.9, alpha=0.7, zorder=15)
            
    xticks = range(0, len(ref_seg), 10)
    ax1.set_xticks([x + 0.5 for x in xticks])
    ax1.set_xticklabels([DISPLAY_START + x for x in xticks], fontsize=13)

    y_locs = [n] + [n - 1 - i for i in range(n)]
    y_labels = ["REF"] + [f"Allele {i+1}" for i in range(n)]
    
    ax1.set_yticks([y + 0.4 for y in y_locs])
    ax1.set_yticklabels(y_labels, fontsize=13)
    
    ax1.set_title(f"Allele Frequency Top Results (Window: {DISPLAY_START}-{DISPLAY_END} bp)", pad=20)
    
    for spine in ['top', 'right', 'left']:
        ax1.spines[spine].set_visible(False)

    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()