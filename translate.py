import os
import fitz  # PyMuPDF
import PyPDF2
from langchain_google_vertexai import VertexAI

def extract_pdf_content_with_structure(pdf_path):
    """
    Extract content from PDF with precise positioning and structure.
    """
    doc = fitz.open(pdf_path)
    structured_content = []
    
    for page_num, page in enumerate(doc):
        # Extract text blocks with position information
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                # Get position information
                x0, y0, x1, y1 = block["bbox"]
                
                # Extract text content
                text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"]
                        
                        # Check if there should be a space
                        if len(line["spans"]) > 1:
                            text += " "
                    
                    # Add newline between lines
                    text += "\n"
                
                # Remove extra whitespace
                text = text.strip()
                
                # Determine if it's a table based on layout or content
                is_table = False
                # Simple heuristic - adjust based on your document
                if "$" in text and len(text.split('\n')) > 1:
                    is_table = True
                
                # Add to structured content
                structured_content.append({
                    "type": "Table" if is_table else "Text",
                    "content": text,
                    "page": page_num,
                    "bbox": (x0, y0, x1, y1),
                    "block_no": len(structured_content)
                })
    
    return structured_content

def translate_pdf(input_path, output_path, project_id, target_lang="es"):
    """
    Translate PDF while preserving structure.
    """
    # Extract content with structure
    print(f"Extracting content from {input_path}...")
    structured_content = extract_pdf_content_with_structure(input_path)
    print(f"Extracted {len(structured_content)} elements")
    
    # Set up Vertex AI
    llm = VertexAI(
        model_name="gemini-pro",
        project=project_id,
        location="us-central1",
        temperature=0.1
    )
    
    # Translate content with structure preservation
    translated_content = []
    
    for i, element in enumerate(structured_content):
        element_type = element["type"]
        content = element["content"]
        
        # Skip empty content
        if not content.strip():
            translated_content.append(element)
            continue
        
        print(f"Translating element {i+1}/{len(structured_content)} ({element_type})...")
        
        # Create specific prompt based on element type
        if element_type == "Table":
            prompt = f"""Translate this table from English to {target_lang}.
            CRUCIAL: Keep ALL dollar amounts, numbers, and symbols EXACTLY as they are.
            Preserve the EXACT table structure with all rows and columns.
            
            TABLE:
            {content}
            
            TRANSLATED TABLE:"""
        else:
            prompt = f"""Translate this text from English to {target_lang}.
            Preserve any special formatting, product names, and terminology.
            
            TEXT:
            {content}
            
            TRANSLATED TEXT:"""
        
        # Get translation
        try:
            translation = llm.invoke(prompt)
            
            # Create translated element
            translated_element = element.copy()
            translated_element["content"] = translation.strip()
            translated_content.append(translated_element)
            
        except Exception as e:
            print(f"Error translating element {i+1}: {str(e)}")
            # Keep original if translation fails
            translated_content.append(element)
    
    # Create new PDF with translated content
    print("Creating translated PDF...")
    output_doc = fitz.open()
    
    # Group elements by page
    elements_by_page = {}
    for element in translated_content:
        page_num = element["page"]
        if page_num not in elements_by_page:
            elements_by_page[page_num] = []
        elements_by_page[page_num].append(element)
    
    # Create pages with translated content
    for page_num in sorted(elements_by_page.keys()):
        # Create new page
        page = output_doc.new_page(width=612, height=792)  # Standard letter size
        
        # Add elements to page
        for element in sorted(elements_by_page[page_num], key=lambda x: x["block_no"]):
            x0, y0, x1, y1 = element["bbox"]
            text = element["content"]
            
            # Insert text at original position
            rect = fitz.Rect(x0, y0, x1, y1)
            page.insert_textbox(rect, text, fontsize=11, align=fitz.TEXT_ALIGN_LEFT)
    
    # Save the translated PDF
    output_doc.save(output_path)
    print(f"Translated PDF saved to: {output_path}")
    
    return output_path

# Example usage
if __name__ == "__main__":
    input_file = "business_document.pdf"
    output_file = "business_document_spanish.pdf"
    google_cloud_project_id = "your-google-cloud-project-id"
    
    translate_pdf(input_file, output_file, project_id=google_cloud_project_id)