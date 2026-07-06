import logfire
from pypdf import PdfReader
from bs4 import BeautifulSoup
from unstructured.partition.auto import partition
 
# PDF Parsing 
def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text (e.g. image-heavy pages).
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")

            text_parts: list[str] = []
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(i + 1)

            # Fallback: use pdfplumber for any pages pypdf returned blank
            if blank_pages:
                logfire.info(f"pypdf returned blank on pages {blank_pages} — retrying with pdfplumber.")
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""
                            if fallback_text.strip():
                                text_parts.append(fallback_text)
                except Exception as plumber_err:
                    logfire.warning(f"pdfplumber fallback failed: {plumber_err}")

            full_text = "\n".join(text_parts)

            if not full_text.strip():
                logfire.warning(f"No text extracted from {file_path}. File may be fully image-based.")
            else:
                logfire.info(f"Extracted {len(full_text)} characters from {file_path}.")

            return full_text

        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise

# Office Document Parsing
def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
    """
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            # Unstructured automatically detects if it's docx or pptx
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements])
            
            if not full_text.strip():
                logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e
        
# Text Parsing
def parse_text(file_path: str):
    """
    Parses plain text files.
    """
    with logfire.span("📄 Text Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logfire.error(f"❌ Text Parse Failed: {e}")
            raise e
        
# HTML Parsing
def parse_html(file_path: str):
    """
    Parses HTML content using BeautifulSoup.
    Cleans scripts, styles, and extracts readable text for RAG.
    """
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, "html.parser")
            
            # 1. Remove Junk (Scripts, Styles, Metadata)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()
                
            # 2. Extract Text
            text = soup.get_text(separator="\n")
            
            # 3. Clean Whitespace (Collapse multiple newlines)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text_clean
        except Exception as e:
            logfire.error(f"❌ HTML Parse Failed: {e}")
            raise e