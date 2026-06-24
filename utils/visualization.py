import py3Dmol
from stmol import showmol
import plotly.express as px
import pandas as pd

def render_3d_protein(cif_data, pocket_pdbs=None):
    view = py3Dmol.view(width=800, height=600)
    
    # 主蛋白渲染
    view.addModel(cif_data, 'cif')
    view.setStyle({'model': 0}, {'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 100}}})
    
    # 渲染 Alpha 球
    if pocket_pdbs:
        colors = ['magenta', 'yellow', 'cyan']
        for i, (pocket_name, pdb_data) in enumerate(pocket_pdbs.items()):
            color = colors[i % len(colors)]
            view.addModel(pdb_data, 'pdb')
            # 使用 model: -1 选择最新添加的模型，设置半透明球体
            view.setStyle({'model': -1}, {'sphere': {'color': color, 'alpha': 0.6}})
            
    view.zoomTo()
    showmol(view, height=600, width=800)

def render_pae_heatmap(pae_matrix):
    fig = px.imshow(
        pae_matrix,
        labels=dict(x="Scored Residue", y="Aligned Residue", color="PAE (Å)"),
        color_continuous_scale="Blues_r",
        zmin=0,
        zmax=30
    )
    fig.update_layout(
        xaxis_title="Scored Residue",
        yaxis_title="Aligned Residue",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def render_domain_timeline(domains_df, seq_length):
    if domains_df.empty or 'Domain' not in domains_df.columns or domains_df.shape[0] == 0:
        fig = px.bar(
            x=[seq_length], 
            y=["Domains"],
            orientation='h'
        )
        fig.update_layout(
            xaxis_title="氨基酸残基位置 (Residue Position)",
            yaxis_title="",
            xaxis=dict(range=[1, max(seq_length, 10)]),
            title="未检测到明确的功能结构域"
        )
        return fig
        
    plot_df = domains_df.copy()
    plot_df['Y'] = 'Domains'
    plot_df['Domain_Length'] = plot_df['End'] - plot_df['Start'] + 1
    
    fig = px.bar(
        plot_df,
        base="Start",
        x="Domain_Length",
        y="Y",
        color="Domain",
        orientation='h',
        hover_name="Domain",
        hover_data={
            "Description": True, 
            "Start": True, 
            "End": True, 
            "Domain_Length": False,
            "Y": False
        },
        labels={'Domain_Length': '结构域长度', 'Domain': 'Pfam 家族'}
    )
    
    fig.update_layout(
        xaxis_title="氨基酸序列残基位置 (Residue Position)",
        yaxis_title="",
        xaxis=dict(range=[1, max(seq_length, 10)], dtick=max(1, seq_length // 10)),
        yaxis=dict(showticklabels=True),
        margin=dict(l=40, r=40, t=40, b=40),
        height=250,
        showlegend=True,
        barmode='stack'
    )
    return fig