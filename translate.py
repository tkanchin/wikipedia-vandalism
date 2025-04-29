import fitz  # pymupdf
import pdfplumber
import json
import sys

if len(sys.argv) != 3:
    print("Usage: python extract_hybrid_pdf.py input.pdf extracted.json")
    sys.exit(1)

input_pdf = sys.argv[1]
output_json = sys.argv[2]

doc = fitz.open(input_pdf)
pdf = pdfplumber.open(input_pdf)

extracted_pages = []

for page_num, (fitz_page, plumber_page) in enumerate(zip(doc, pdf.pages), 1):
    page_content = {
        "page_number": page_num,
        "width": fitz_page.rect.width,
        "height": fitz_page.rect.height,
        "blocks": [],
        "tables": []
    }

    # 1. Extract text blocks (paragraphs/headings)
    blocks = fitz_page.get_text("blocks")
    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        if block_type == 0 and text.strip():
            page_content["blocks"].append({
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text.strip()
            })

    # 2. Extract tables
    tables = plumber_page.extract_tables()
    for table in tables:
        page_content["tables"].append({
            "table": table  # keep table as list of lists
        })

    extracted_pages.append(page_content)

doc.close()
pdf.close()

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(extracted_pages, f, ensure_ascii=False, indent=2)

print(f"Hybrid extraction completed. Saved to {output_json}")


import json
import sys
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

if len(sys.argv) != 3:
    print("Usage: python translate_hybrid_pdf.py extracted.json translated.json")
    sys.exit(1)

input_json = sys.argv[1]
output_json = sys.argv[2]

with open(input_json, 'r', encoding='utf-8') as f:
    pdf_data = json.load(f)

llm = VertexAI(model_name="text-bison@001", temperature=0.1, max_output_tokens=2048)

# Robust business Spanish prompt
prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a professional Spanish translator. Translate the following English content into formal Business Spanish.\n"
        "Translate exactly. Do not ask questions or explain. Retain formatting.\n"
        "Only output the translated Spanish content.\n\n"
        "English:\n{text}\n\nSpanish:"
    )
)

chain = LLMChain(llm=llm, prompt=prompt)

for page in pdf_data:
    # Translate text blocks
    for block in page['blocks']:
        english_text = block.get('text', '')
        if english_text.strip():
            block['text'] = chain.run(text=english_text).strip()
        else:
            block['text'] = ""

    # Translate tables (cell by cell)
    for table_entry in page['tables']:
        new_table = []
        for row in table_entry['table']:
            new_row = []
            for cell in row:
                cell_text = cell.strip() if cell else ""
                if cell_text:
                    translated_cell = chain.run(text=cell_text).strip()
                    new_row.append(translated_cell)
                else:
                    new_row.append("")
            new_table.append(new_row)
        table_entry['table'] = new_table

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(pdf_data, f, ensure_ascii=False, indent=2)

print(f"Translation to Spanish completed. Saved to {output_json}")


import json
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, LongTable, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

if len(sys.argv) != 3:
    print("Usage: python rebuild_hybrid_pdf.py translated.json output.pdf")
    sys.exit(1)

input_json = sys.argv[1]
output_pdf = sys.argv[2]

with open(input_json, 'r', encoding='utf-8') as f:
    pdf_data = json.load(f)

styles = getSampleStyleSheet()
business_style = ParagraphStyle(
    'Business',
    parent=styles['Normal'],
    fontSize=10,
    leading=13,
    spaceAfter=10
)

doc = SimpleDocTemplate(output_pdf, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

elements = []

for page in pdf_data:
    # Add translated text blocks
    for block in page['blocks']:
        text = block.get('text', '')
        if text:
            clean_text = text.replace('<br>', '<br/>')  # Fix <br> tags
            elements.append(Paragraph(clean_text.strip(), business_style))
            elements.append(Spacer(1, 8))

    # Add translated tables
    for table_entry in page['tables']:
        table_data = table_entry['table']
        table = LongTable(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    elements.append(PageBreak())

doc.build(elements)

print(f"Final Spanish PDF rebuilt successfully at {output_pdf}")
