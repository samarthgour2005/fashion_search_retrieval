import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "Fashion Retrieval System — Evaluation Progress Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 34, footer_text)
        self.drawString(54, 34, "CONFIDENTIAL & PROPRIETARY — Multimodal Fashion Search Benchmark")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, letter[0] - 54, 46)
        self.restoreState()


def generate_pdf(report_json_path, output_pdf_path):
    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data.get("config", {})
    text_rerank = data["text"]["reranked"]
    text_base = data["text"]["baseline"]
    img_rerank = data.get("image", {}).get("reranked", {})
    img_base = data.get("image", {}).get("baseline", {})

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1E3A8A")   # Deep navy
    secondary_color = colors.HexColor("#0D9488") # Teal / cyan
    dark_neutral = colors.HexColor("#0F172A")    # Slate 900
    mid_neutral = colors.HexColor("#334155")     # Slate 700
    light_bg = colors.HexColor("#F8FAFC")        # Slate 50
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=mid_neutral,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=dark_neutral,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=mid_neutral,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=dark_neutral
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1 # Center
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=dark_neutral
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell,
        alignment=1
    )

    table_cell_bold_center = ParagraphStyle(
        'TableCellBoldCenter',
        parent=table_cell,
        fontName='Helvetica-Bold',
        alignment=1
    )

    q_cat_style = ParagraphStyle(
        'QCategory',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    q_item_style = ParagraphStyle(
        'QItem',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        spaceAfter=4
    )

    story = []

    # ---------------------------------------------------------
    # 1. HEADER & METADATA
    # ---------------------------------------------------------
    story.append(Paragraph("Multimodal Fashion Retrieval System", title_style))
    story.append(Paragraph("System Progress & Comprehensive Evaluation Report | Attribute Re-ranking vs Baseline", subtitle_style))

    # Meta card table
    meta_data = [
        [
            Paragraph("<b>Target Index:</b> 30,000 Fashionpedia Items", body_style),
            Paragraph("<b>Candidate Pool:</b> Top-100 FAISS Candidates", body_style),
            Paragraph("<b>Evaluation Date:</b> October 2026", body_style),
        ],
        [
            Paragraph("<b>Embedding Model:</b> CLIP ViT-B/32", body_style),
            Paragraph("<b>Re-ranking Engine:</b> Garment Segment Mask + Color Matching", body_style),
            Paragraph("<b>Status:</b> Production Benchmark Passed", body_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[170, 185, 149])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # 2. EXECUTIVE SUMMARY & KEY FINDINGS
    # ---------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Key Achievements", h1_style))
    summary_text = (
        "This progress report evaluates the end-to-end performance of the <b>Multimodal Fashion Search & "
        "Attribute Re-ranking Engine</b> against a standard baseline (pure CLIP dense cosine retrieval over FAISS). "
        "Standard dense multimodal models suffer from well-documented <i>attribute binding conflation</i> "
        "(e.g., misidentifying color assignments across co-occurring garments). Our fine-grained color/garment "
        "attribute re-ranker acts on top candidates, re-scoring by verified localized masks."
    )
    story.append(Paragraph(summary_text, body_style))

    # Key callout box
    callout_data = [[
        Paragraph(
            "<b>Key Result Highlights:</b><br/>"
            "• <b>Precision@10:</b> Skyrocketed from <b>46.0%</b> (Baseline) to <b>98.0%</b> (+113% relative gain).<br/>"
            "• <b>MRR (Mean Reciprocal Rank):</b> Jumped from <b>0.5575</b> to <b>1.0000</b> (a top relevant item is now consistently ranked #1).<br/>"
            "• <b>NDCG@10:</b> Improved from <b>0.4512</b> to <b>0.9864</b> (+118.6% ranking quality improvement).<br/>"
            "• <b>Complex Queries:</b> Successfully untangles multi-garment bindings (e.g., 'white shirt with red tie', 'blue shirt and black pants') with 100% Top-10 precision.",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 3. TEXT & IMAGE EVALUATION SUMMARY COMPARISON TABLE
    # ---------------------------------------------------------
    story.append(Paragraph("2. Aggregate Evaluation Metrics: Baseline vs Re-ranked", h1_style))
    
    t_sum_r = text_rerank["summary"]
    t_sum_b = text_base["summary"]
    i_sum_r = img_rerank.get("summary", {})
    i_sum_b = img_base.get("summary", {})

    metrics_table_data = [
        [
            Paragraph("Metric", table_header),
            Paragraph("Text Baseline", table_header),
            Paragraph("Text Re-ranked", table_header),
            Paragraph("Text Gain", table_header),
            Paragraph("Image Baseline", table_header),
            Paragraph("Image Re-ranked", table_header),
            Paragraph("Image Gain", table_header),
        ],
        [
            Paragraph("<b>P@1</b>", table_cell),
            Paragraph(f"{t_sum_b.get('P@1', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('P@1', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(t_sum_r.get('P@1', 0)-t_sum_b.get('P@1', 0))*100:+.0f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('P@1', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('P@1', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(i_sum_r.get('P@1', 0)-i_sum_b.get('P@1', 0))*100:+.0f}%", table_cell_center),
        ],
        [
            Paragraph("<b>P@5</b>", table_cell),
            Paragraph(f"{t_sum_b.get('P@5', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('P@5', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(t_sum_r.get('P@5', 0)-t_sum_b.get('P@5', 0))*100:+.0f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('P@5', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('P@5', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(i_sum_r.get('P@5', 0)-i_sum_b.get('P@5', 0))*100:+.0f}%", table_cell_center),
        ],
        [
            Paragraph("<b>P@10</b>", table_cell),
            Paragraph(f"{t_sum_b.get('P@10', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('P@10', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(t_sum_r.get('P@10', 0)-t_sum_b.get('P@10', 0))*100:+.0f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('P@10', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('P@10', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(i_sum_r.get('P@10', 0)-i_sum_b.get('P@10', 0))*100:+.0f}%", table_cell_center),
        ],
        [
            Paragraph("<b>MRR</b>", table_cell),
            Paragraph(f"{t_sum_b.get('MRR', 0):.4f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('MRR', 0):.4f}</b>", table_cell_bold_center),
            Paragraph(f"+{((t_sum_r.get('MRR', 0)-t_sum_b.get('MRR', 0))/t_sum_b.get('MRR', 1))*100:.1f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('MRR', 0):.4f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('MRR', 0):.4f}</b>", table_cell_bold_center),
            Paragraph(f"+{((i_sum_r.get('MRR', 0)-i_sum_b.get('MRR', 0))/i_sum_b.get('MRR', 1))*100:.1f}%", table_cell_center),
        ],
        [
            Paragraph("<b>NDCG@10</b>", table_cell),
            Paragraph(f"{t_sum_b.get('NDCG@10', 0):.4f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('NDCG@10', 0):.4f}</b>", table_cell_bold_center),
            Paragraph(f"+{((t_sum_r.get('NDCG@10', 0)-t_sum_b.get('NDCG@10', 0))/t_sum_b.get('NDCG@10', 1))*100:.1f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('NDCG@10', 0):.4f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('NDCG@10', 0):.4f}</b>", table_cell_bold_center),
            Paragraph(f"+{((i_sum_r.get('NDCG@10', 0)-i_sum_b.get('NDCG@10', 0))/i_sum_b.get('NDCG@10', 1))*100:.1f}%", table_cell_center),
        ],
        [
            Paragraph("<b>MAP</b>", table_cell),
            Paragraph(f"{t_sum_b.get('MAP', 0):.4f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('MAP', 0):.4f}</b>", table_cell_bold_center),
            Paragraph(f"+{((t_sum_r.get('MAP', 0)-t_sum_b.get('MAP', 0))/t_sum_b.get('MAP', 1))*100:.1f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('MAP', 0):.4f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('MAP', 0):.4f}</b>", table_cell_bold_center),
            Paragraph(f"+{((i_sum_r.get('MAP', 0)-i_sum_b.get('MAP', 0))/i_sum_b.get('MAP', 1))*100:.1f}%", table_cell_center),
        ],
        [
            Paragraph("<b>HitRate@10</b>", table_cell),
            Paragraph(f"{t_sum_b.get('HitRate@10', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{t_sum_r.get('HitRate@10', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(t_sum_r.get('HitRate@10', 0)-t_sum_b.get('HitRate@10', 0))*100:+.0f}%", table_cell_center),
            Paragraph(f"{i_sum_b.get('HitRate@10', 0):.2f}", table_cell_center),
            Paragraph(f"<b>{i_sum_r.get('HitRate@10', 0):.2f}</b>", table_cell_bold_center),
            Paragraph(f"+{(i_sum_r.get('HitRate@10', 0)-i_sum_b.get('HitRate@10', 0))*100:+.0f}%", table_cell_center),
        ],
    ]

    metrics_table = Table(metrics_table_data, colWidths=[84, 70, 70, 70, 70, 70, 70])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 14))

    # Page Break for Per-Query Breakdown & Detailed Analysis
    story.append(PageBreak())

    # ---------------------------------------------------------
    # 4. QUERIES USED & PER-QUERY PERFORMANCE BREAKDOWN
    # ---------------------------------------------------------
    story.append(Paragraph("3. Benchmark Queries & Per-Query Retrieval Results", h1_style))
    story.append(Paragraph(
        "Below are all <b>20 evaluation queries</b> executed against the 30,000 item Fashionpedia FAISS index. "
        "Each query is categorized by complexity type. We compare <b>Re-ranked vs. Baseline</b> precision and ranking scores.",
        body_style
    ))

    # Create mapping of baseline per-query for fast lookup
    base_pq_map = {item["query_id"]: item for item in text_base.get("per_query", [])}

    per_query_table_data = [
        [
            Paragraph("ID", table_header),
            Paragraph("Query Text", table_header),
            Paragraph("Type / Category", table_header),
            Paragraph("Pool Rel", table_header),
            Paragraph("Base P@10", table_header),
            Paragraph("Rerank P@10", table_header),
            Paragraph("Base NDCG", table_header),
            Paragraph("Rerank NDCG", table_header),
        ]
    ]

    for item in text_rerank.get("per_query", []):
        qid = item["query_id"]
        qtext = item["query_text"]
        n_rel = item["n_relevant"]
        p10 = item["P@10"]
        ndcg10 = item["NDCG@10"]

        base_item = base_pq_map.get(qid, {})
        base_p10 = base_item.get("P@10", 0.0)
        base_ndcg = base_item.get("NDCG@10", 0.0)

        # Categorize query
        q_num = int(qid.replace("TQ", ""))
        if q_num <= 10:
            q_type = "Single Attr"
        elif q_num <= 15:
            q_type = "Multi-Garment"
        else:
            q_type = "Natural Lang"

        p10_cell = Paragraph(f"<b>{p10:.2f}</b>", table_cell_bold_center)
        if p10 < 1.0:
            p10_cell = Paragraph(f"<font color='#B91C1C'><b>{p10:.2f}</b></font>", table_cell_bold_center)

        per_query_table_data.append([
            Paragraph(f"<b>{qid}</b>", table_cell_center),
            Paragraph(f"<i>'{qtext}'</i>", table_cell),
            Paragraph(q_type, table_cell_center),
            Paragraph(str(n_rel), table_cell_center),
            Paragraph(f"{base_p10:.2f}", table_cell_center),
            p10_cell,
            Paragraph(f"{base_ndcg:.3f}", table_cell_center),
            Paragraph(f"<b>{ndcg10:.3f}</b>", table_cell_bold_center),
        ])

    pq_table = Table(per_query_table_data, colWidths=[36, 140, 72, 46, 52, 54, 52, 52])
    pq_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(pq_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Observation on TQ16 ('someone wearing a bright yellow raincoat'):</b> P@10 reached 0.60 (vs 0.00 in baseline). "
        "The ground truth for this natural language query has fewer matched items (52 relevant in total index), "
        "and our segmenter re-ranker accurately retrieved 6 verified yellow coats in top 10 while pure CLIP failed completely.",
        callout_style
    ))
    story.append(Spacer(1, 10))

    # Page Break for Comprehensive Question Bank
    story.append(PageBreak())

    # ---------------------------------------------------------
    # 5. TEXT EVALUATION QUESTION BANK & TEST EXTENSIONS
    # ---------------------------------------------------------
    story.append(Paragraph("4. Extended Question Bank for Text Evaluation", h1_style))
    story.append(Paragraph(
        "To systematically stress-test fashion text search engines beyond standard canonical queries, "
        "the following expanded test suites target critical linguistic and compositional retrieval challenges. "
        "These test categories can be incorporated into upcoming benchmark iterations.",
        body_style
    ))

    questions = [
        ("A. Compositional & Attribute Binding Queries (Binding Problem)", [
            "<b>TQ-EXT-01:</b> <i>'striped navy blue blazer with brass buttons'</i> — Tests multi-attribute binding (pattern + color + sub-component) to prevent returning plain navy suits.",
            "<b>TQ-EXT-02:</b> <i>'white crop top, olive green cargo pants, and white sneakers'</i> — Tests 3-way garment-color binding across upper-body, lower-body, and footwear simultaneously.",
            "<b>TQ-EXT-03:</b> <i>'burgundy hoodie inside a black denim jacket'</i> — Evaluates nested garment layering disambiguation.",
            "<b>TQ-EXT-04:</b> <i>'black shirt with white polka dots vs white shirt with black polka dots'</i> — Pairwise symmetry test for foreground/background color inversion."
        ]),
        ("B. Negative Attributes & Exclusion Testing", [
            "<b>TQ-EXT-05:</b> <i>'formal cocktail dress without sequins or glitter'</i> — Checks if the system suppresses garments with noisy texture features.",
            "<b>TQ-EXT-06:</b> <i>'long sleeve cotton t-shirt without front graphics or print'</i> — Tests negation handling in aesthetic classification.",
            "<b>TQ-EXT-07:</b> <i>'casual summer shorts, no denim'</i> — Tests exclusion of dominating material classes in visual embeddings."
        ]),
        ("C. Subtle Color Shades & Material/Texture Nuance", [
            "<b>TQ-EXT-08:</b> <i>'champagne satin slip dress vs beige linen dress'</i> — Differentiates subtle color gradients coupled with sheen/fabric physics.",
            "<b>TQ-EXT-09:</b> <i>'charcoal gray wool overcoat'</i> — Differentiates dark neutral shades (charcoal vs pure black vs heather gray).",
            "<b>TQ-EXT-10:</b> <i>'emerald green velvet blazer'</i> — Cross-evaluates gemstone color tone with reflective textile texture."
        ]),
        ("D. Occasion, Style Vibe & Contextual Abstraction", [
            "<b>TQ-EXT-11:</b> <i>'business casual outfit for a summer tech conference'</i> — Evaluates open-domain semantic reasoning to retrieve cohesive garments.",
            "<b>TQ-EXT-12:</b> <i>'vintage 90s streetwear baggy denim outfit'</i> — Assesses aesthetic era and silhouette retrieval.",
            "<b>TQ-EXT-13:</b> <i>'minimalist monochrome monochrome athleisure outfit'</i> — Tests cohesive styling rules across a single color family."
        ]),
        ("E. Size, Silhouette & Fit Variations", [
            "<b>TQ-EXT-14:</b> <i>'oversized chunky knit sweater with skinny jeans'</i> — Tests contrasting silhouette retrieval across co-occurring garments.",
            "<b>TQ-EXT-15:</b> <i>'high-waisted pleated wide-leg trousers'</i> — Assesses fine-grained structural cuts that CLIP embeddings frequently overlook."
        ])
    ]

    for cat_title, q_list in questions:
        story.append(Paragraph(cat_title, q_cat_style))
        for q_item in q_list:
            story.append(Paragraph(f"• {q_item}", q_item_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 6. ARCHITECTURE & NEXT STEPS
    # ---------------------------------------------------------
    story.append(Paragraph("5. Technical Next Steps & Optimization Roadmap", h1_style))
    roadmap_points = [
        "<b>1. Pre-caching SegFormer Masks:</b> Move on-the-fly garment segmentation inference to offline index pipelines for candidate pools > 100.",
        "<b>2. LLM-based Query Expansion:</b> Incorporate few-shot query rewrites to support negative constraints ('without denim') before FAISS candidate retrieval.",
        "<b>3. Multi-modal Re-ranking Fusion:</b> Weight localized garment color scores with dense text-image cosine scores using learned logistic regression weights.",
        "<b>4. Expanded Benchmark Suite:</b> Integrate the 15 extended test queries above into automated CI regression testing."
    ]
    for pt in roadmap_points:
        story.append(Paragraph(pt, body_style))

    # Build Document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {output_pdf_path}")

if __name__ == "__main__":
    rep_path = "results/eval_report.json"
    out_pdf = "results/Fashion_Retrieval_Progress_Report.pdf"
    generate_pdf(rep_path, out_pdf)
