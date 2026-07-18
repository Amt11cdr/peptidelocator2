"""
PeptideLocator2 — Gradio Demo
Predicts cleavage sites and peptide regions in a protein sequence using ESM2-8M.

Deploy to Hugging Face Spaces:
    gradio deploy

Or run locally:
    pip install gradio
    python app.py
"""

import os
import sys
import numpy as np
import gradio as gr
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from peptidelocator.inference import PeptideLocatorPredictor


# ── Load model once at startup ────────────────────────────────────────────────

FINETUNE_SITES    = os.environ.get("FINETUNE_SITES_PATH",   None)
FINETUNE_PEPTIDES = os.environ.get("FINETUNE_PEPTIDES_PATH", None)

predictor = PeptideLocatorPredictor(
    finetune_sites_path=FINETUNE_SITES,
    finetune_peptides_path=FINETUNE_PEPTIDES,
    sites_head_path="models/sites_head.pt",
    peptides_head_path="models/peptides_head.pt",
)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _prob_to_red(p: float) -> str:
    """Map probability [0,1] to a red RGBA colour string."""
    alpha = 0.08 + 0.82 * p
    return f"rgba(220,50,47,{alpha:.2f})"

def _prob_to_blue(p: float) -> str:
    """Map probability [0,1] to a blue RGBA colour string."""
    alpha = 0.08 + 0.82 * p
    return f"rgba(38,139,210,{alpha:.2f})"

def _prob_to_purple(p_site: float, p_pep: float) -> str:
    """Both signals present — blend to purple."""
    alpha = 0.08 + 0.82 * max(p_site, p_pep)
    return f"rgba(108,53,180,{alpha:.2f})"


def build_sequence_html(sequence: str, sites_proba: np.ndarray,
                        peptides_proba: np.ndarray,
                        threshold: float = 0.5) -> str:
    """
    Render the sequence as coloured HTML spans.
      Red   = cleavage site signal
      Blue  = peptide region signal
      Purple = both
      Grey  = neither
    Residues predicted above threshold get a bold underline.
    """
    SITE_THRESH = threshold
    PEP_THRESH  = threshold

    spans = []
    for i, aa in enumerate(sequence):
        p_site = float(sites_proba[i])
        p_pep  = float(peptides_proba[i])
        is_site = p_site >= SITE_THRESH
        is_pep  = p_pep  >= PEP_THRESH

        if is_site and is_pep:
            bg = _prob_to_purple(p_site, p_pep)
        elif is_site:
            bg = _prob_to_red(p_site)
        elif is_pep:
            bg = _prob_to_blue(p_pep)
        else:
            # Still colour lightly if any probability > 0.2
            if p_site > 0.2:
                bg = _prob_to_red(p_site)
            elif p_pep > 0.2:
                bg = _prob_to_blue(p_pep)
            else:
                bg = "transparent"

        weight = "bold" if (is_site or is_pep) else "normal"
        border = "2px solid rgba(0,0,0,0.3)" if (is_site or is_pep) else "none"

        tooltip = (
            f"Pos {i+1} | {aa} | "
            f"Site: {p_site:.3f} | Peptide: {p_pep:.3f}"
        )
        spans.append(
            f'<span title="{tooltip}" style="'
            f'background:{bg};'
            f'font-weight:{weight};'
            f'border-bottom:{border};'
            f'padding:2px 1px;'
            f'border-radius:2px;'
            f'cursor:default;'
            f'">{aa}</span>'
        )

    # Wrap into lines of 60 AA
    line_size = 60
    lines = []
    for start in range(0, len(sequence), line_size):
        chunk = spans[start:start + line_size]
        pos_label = (
            f'<span style="color:#888;font-size:11px;'
            f'margin-right:8px;user-select:none;">'
            f'{start+1:>5}</span>'
        )
        lines.append(pos_label + "".join(chunk))

    html = (
        '<div style="'
        'font-family:monospace;'
        'font-size:14px;'
        'line-height:2.2;'
        'word-break:break-all;'
        'padding:12px;'
        'background:#fafafa;'
        'border:1px solid #e0e0e0;'
        'border-radius:8px;'
        'overflow-x:auto;'
        '">'
        + "<br>".join(lines)
        + "</div>"
    )
    return html


def build_legend_html() -> str:
    return """
    <div style="display:flex;gap:20px;padding:6px 0;font-size:13px;font-family:sans-serif;">
      <span><span style="display:inline-block;width:14px;height:14px;background:rgba(220,50,47,0.8);
        border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Cleavage site</span>
      <span><span style="display:inline-block;width:14px;height:14px;background:rgba(38,139,210,0.8);
        border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Peptide region</span>
      <span><span style="display:inline-block;width:14px;height:14px;background:rgba(108,53,180,0.8);
        border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Both</span>
      <span style="color:#888;font-size:12px;">Hover over residues for per-position scores</span>
    </div>
    """


def build_probability_plot(sequence: str, sites_proba: np.ndarray,
                           peptides_proba: np.ndarray,
                           threshold: float = 0.5) -> go.Figure:
    positions = list(range(1, len(sequence) + 1))

    fig = go.Figure()

    # Shaded regions where peptide is predicted
    pep_regions = []
    in_region = False
    for i, p in enumerate(peptides_proba):
        if p >= threshold and not in_region:
            region_start = i + 1
            in_region = True
        elif p < threshold and in_region:
            pep_regions.append((region_start, i))
            in_region = False
    if in_region:
        pep_regions.append((region_start, len(sequence)))

    for start, end in pep_regions:
        fig.add_vrect(
            x0=start - 0.5, x1=end + 0.5,
            fillcolor="rgba(38,139,210,0.08)",
            layer="below", line_width=0,
        )

    # Peptide region probability
    fig.add_trace(go.Scatter(
        x=positions,
        y=peptides_proba,
        mode="lines",
        name="Peptide region",
        line=dict(color="rgba(38,139,210,0.9)", width=2),
        hovertemplate="Pos %{x} (%{customdata})<br>Peptide prob: %{y:.3f}<extra></extra>",
        customdata=list(sequence),
        fill="tozeroy",
        fillcolor="rgba(38,139,210,0.05)",
    ))

    # Cleavage site probability
    fig.add_trace(go.Scatter(
        x=positions,
        y=sites_proba,
        mode="lines",
        name="Cleavage site",
        line=dict(color="rgba(220,50,47,0.9)", width=2),
        hovertemplate="Pos %{x} (%{customdata})<br>Site prob: %{y:.3f}<extra></extra>",
        customdata=list(sequence),
    ))

    # Mark predicted cleavage site positions as vertical dotted lines
    site_positions = [i + 1 for i, p in enumerate(sites_proba) if p >= threshold]
    for pos in site_positions:
        fig.add_vline(
            x=pos, line_dash="dot",
            line_color="rgba(220,50,47,0.5)", line_width=1,
        )

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color="rgba(0,0,0,0.3)", line_width=1,
        annotation_text=f"threshold ({threshold})",
        annotation_position="top right",
        annotation_font_size=10,
    )

    fig.update_layout(
        xaxis_title="Position",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        height=280,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def build_summary(sequence: str, sites_proba: np.ndarray,
                  peptides_proba: np.ndarray,
                  threshold: float = 0.5) -> str:
    site_positions = [i + 1 for i, p in enumerate(sites_proba) if p >= threshold]

    # Find contiguous peptide regions
    pep_regions = []
    in_region = False
    for i, p in enumerate(peptides_proba):
        if p >= threshold and not in_region:
            region_start = i + 1
            in_region = True
        elif p < threshold and in_region:
            pep_regions.append((region_start, i))
            in_region = False
    if in_region:
        pep_regions.append((region_start, len(sequence)))

    site_str = (
        f"**{len(site_positions)} cleavage site(s)** predicted at position(s): "
        + (", ".join(str(p) for p in site_positions) if site_positions else "none")
    )
    pep_str = (
        f"**{len(pep_regions)} peptide region(s)** predicted: "
        + (", ".join(f"{s}–{e} ({e-s+1} aa)" for s, e in pep_regions)
           if pep_regions else "none")
    )
    coverage = 100 * sum(1 for p in peptides_proba if p >= threshold) / len(sequence)

    return (
        f"{site_str}  \n"
        f"{pep_str}  \n"
        f"Peptide coverage: **{coverage:.1f}%** of sequence  \n"
        f"Sequence length: **{len(sequence)} aa**"
    )


# ── Prediction function called by Gradio ─────────────────────────────────────

def predict(sequence: str, threshold: float):
    if not sequence or not sequence.strip():
        return (
            "<p style='color:#888;padding:12px;'>Enter a protein sequence above.</p>",
            None,
            "",
            "",
        )

    try:
        result = predictor.predict(sequence, threshold=threshold)
    except Exception as e:
        return (
            f"<p style='color:red;padding:12px;'>Error: {e}</p>",
            None,
            "",
            "",
        )

    seq_html   = build_sequence_html(result.sequence, result.sites_proba,
                                     result.peptides_proba, threshold)
    prob_plot  = build_probability_plot(result.sequence, result.sites_proba,
                                        result.peptides_proba, threshold)
    summary    = build_summary(result.sequence, result.sites_proba,
                               result.peptides_proba, threshold)

    return seq_html, prob_plot, summary, result.model_info


# ── Example sequences ─────────────────────────────────────────────────────────

EXAMPLES = [
    ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHD"
     "FSAGEGLYTH", 0.5],
    ["MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSL"
     "QKRGIVEQCCTSICSLYQLENYCN", 0.5],
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
**PeptideLocator2** predicts cleavage sites and peptide regions in protein sequences
using ESM2-8M protein language model embeddings.

- 🔴 **Red** = cleavage site probability
- 🔵 **Blue** = peptide region probability
- 🟣 **Purple** = both signals present
- Hover over residues for exact per-position scores
"""

with gr.Blocks(
    title="PeptideLocator2",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container { max-width: 960px !important; }",
) as demo:

    gr.Markdown("# PeptideLocator2")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=4):
            seq_input = gr.Textbox(
                label="Protein Sequence",
                placeholder="Paste your amino acid sequence here (single-letter codes)...",
                lines=4,
                max_lines=8,
            )
        with gr.Column(scale=1):
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Prediction threshold",
            )
            predict_btn = gr.Button("Predict", variant="primary", size="lg")

    gr.HTML(build_legend_html())

    seq_display = gr.HTML(label="Sequence")
    prob_plot   = gr.Plot(label="Per-residue Probabilities", show_label=True)

    with gr.Row():
        with gr.Column():
            summary_md = gr.Markdown(label="Summary")
        with gr.Column():
            model_info = gr.Textbox(label="Model", interactive=False, lines=2)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[seq_input, threshold_slider],
        outputs=[seq_display, prob_plot, summary_md, model_info],
        fn=predict,
        cache_examples=False,
        label="Example sequences",
    )

    predict_btn.click(
        fn=predict,
        inputs=[seq_input, threshold_slider],
        outputs=[seq_display, prob_plot, summary_md, model_info],
    )
    seq_input.submit(
        fn=predict,
        inputs=[seq_input, threshold_slider],
        outputs=[seq_display, prob_plot, summary_md, model_info],
    )

    gr.Markdown(
        "<div style='text-align:center;color:#888;font-size:12px;margin-top:16px;'>"
        "PeptideLocator2 · UCD Shields Lab · ESM2-8M · "
        "<a href='https://github.com/shields-lab/peptide-locator-2' target='_blank'>GitHub</a>"
        "</div>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
