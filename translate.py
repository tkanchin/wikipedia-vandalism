import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_transformers import DoctranTextTranslator
from langchain_google_vertexai import VertexAI
from docx import Document

# Create a custom translator using Vertex AI
class VertexTranslator:
    def __init__(self, llm, target_lang):
        self.llm = llm
        self.target_lang = target_lang

    def transform_documents(self, documents):
        translated_docs = []
        for doc in documents:
            # Determine if this is a table
            is_table = doc.metadata.get("category", "") == "Table"
            
            # Create appropriate prompt based on element type
            if is_table:
                prompt = f"""Translate this table from English to {self.target_lang}. 
                IMPORTANT: Preserve ALL formatting, dollar amounts, numbers, and table structure exactly.
                
                TABLE: {doc.page_content}
                
                TRANSLATED TABLE:"""
            else:
                prompt = f"""Translate this text from English to {self.target_lang}:
                
                {doc.page_content}
                
                TRANSLATION:"""
            
            # Get translation
            translation = self.llm.invoke(prompt)
            
            # Create new document with translation
            translated_doc = doc.copy()
            translated_doc.page_content = translation
            translated_docs.append(translated_doc)
        
        return translated_docs

def translate_document_with_tables(
    input_path, 
    output_path
    target_lang="es"
):
    """
    Translate a Word document with tables using LangChain tools.
    """
    # Step 1: Load the document with structure preservation
    print(f"Loading document from {input_path}...")
    loader = UnstructuredWordDocumentLoader(
        input_path,
        mode="elements",  # This preserves document structure including tables
        strategy="fast"
    )
    elements = loader.load()
    print(f"Extracted {len(elements)} elements")
    
    llm = VertexAI(
        model_name="gemini-pro",
        project=project_id,
        location="us-central1",
        temperature=0.1
    )

    translator = VertexTranslator(llm, target_lang)
    print("Using Vertex AI for translation")
    
    # Step 3: Translate the elements
    print("Translating document elements...")
    translated_elements = translator.transform_documents(elements)
    print(f"Translated {len(translated_elements)} elements")
    
    # Step 4: Rebuild the document
    # This is the simplified approach - for better formatting,
    # you would need more complex handling
    output_doc = Document()
    
    # Group elements by type and add them to the document
    for element in translated_elements:
        element_type = element.metadata.get("category", "")
        content = element.page_content
        
        if element_type == "Title":
            output_doc.add_heading(content, level=0)
        elif element_type.startswith("Heading"):
            # Extract heading level if available
            level = 1
            if len(element_type) > 7:
                try:
                    level = int(element_type[7:])
                except ValueError:
                    pass
            output_doc.add_heading(content, level=level)
        elif element_type == "Table":
            # For tables, we do our best to preserve structure
            rows = content.strip().split('\n')
            if rows:
                # Estimate columns from the first row
                first_row = rows[0]
                cols = max(1, len(first_row.split('\t')) if '\t' in first_row else 1)
                
                table = output_doc.add_table(rows=len(rows), cols=cols)
                # Apply a style
                table.style = 'Table Grid'
                
                # Fill the table
                for i, row_text in enumerate(rows):
                    cells = row_text.split('\t') if '\t' in row_text else [row_text]
                    for j, cell_text in enumerate(cells):
                        if j < cols:
                            table.cell(i, j).text = cell_text
        else:
            # Regular paragraph
            output_doc.add_paragraph(content)
    
    # Save the translated document
    output_doc.save(output_path)
    print(f"Translated document saved to: {output_path}")
    
    return output_path

# Example usage
if __name__ == "__main__":
    input_file = "business_checking.docx"
    output_file = "business_checking_spanish.docx"
        
    translate_document_with_tables(
        input_file,
        output_file
    )