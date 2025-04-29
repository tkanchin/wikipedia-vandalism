import pdfplumber
import json
import sys

if len(sys.argv) != 3:
    print("Usage: python extract_pdf.py input.pdf extracted.json")
    sys.exit(1)

input_pdf, output_json = sys.argv[1], sys.argv[2]

pdf_data = []
with pdfplumber.open(input_pdf) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        tables = page.extract_tables()
        text = page.extract_text(layout=True)

        pdf_data.append({
            'page': page_num,
            'width': page.width,
            'height': page.height,
            'tables': tables,
            'text': text
        })

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(pdf_data, f, ensure_ascii=False, indent=2)

print(f"Extracted PDF content to {output_json}")

import json
import sys
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

if len(sys.argv) != 3:
    print("Usage: python translate_pdf.py extracted.json translated.json")
    sys.exit(1)

input_json, output_json = sys.argv[1], sys.argv[2]

with open(input_json, 'r', encoding='utf-8') as f:
    pdf_data = json.load(f)

llm = VertexAI(model_name="text-bison@001", temperature=0.1, max_output_tokens=2048)

prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Translate the following English content to formal business Spanish. "
        "Ensure accuracy, professionalism, and appropriate terminology:\n\n"
        "{text}\n\n"
        "Business Spanish Translation:"
    )
)

chain = LLMChain(llm=llm, prompt=prompt)

for page in pdf_data:
    # Translate page text
    page_text = page.get('text', '')
    page['translated_text'] = chain.run(text=page_text.strip()) if page_text else ""

    # Translate tables cell-by-cell
    translated_tables = []
    for table in page.get('tables', []):
        translated_table = []
        for row in table:
            translated_row = []
            for cell in row:
                cell_text = cell.strip() if cell else ""
                translated_cell = chain.run(text=cell_text) if cell_text else ""
                translated_row.append(translated_cell.strip())
            translated_table.append(translated_row)
        translated_tables.append(translated_table)

    page['translated_tables'] = translated_tables

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(pdf_data, f, ensure_ascii=False, indent=2)

print(f"Translation completed. Saved as {output_json}")


import json
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

if len(sys.argv) != 3:
    print("Usage: python rebuild_pdf.py translated.json translated_output.pdf")
    sys.exit(1)

input_json, output_pdf = sys.argv[1], sys.argv[2]

with open(input_json, 'r', encoding='utf-8') as f:
    pdf_data = json.load(f)

styles = getSampleStyleSheet()
business_style = ParagraphStyle(
    'Business',
    parent=styles['Normal'],
    fontSize=11,
    leading=14,
    spaceAfter=12
)

doc = SimpleDocTemplate(output_pdf, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
elements = []

for page in pdf_data:
    # Add translated text
    text_content = page.get('translated_text', '')
    if text_content:
        for para in text_content.split('\n'):
            if para.strip():
                elements.append(Paragraph(para.strip(), business_style))
                elements.append(Spacer(1, 8))

    # Add translated tables
    for table_data in page.get('translated_tables', []):
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    elements.append(PageBreak())  # Maintain original page breaks

doc.build(elements)

print(f"Translated PDF rebuilt successfully at {output_pdf}")
