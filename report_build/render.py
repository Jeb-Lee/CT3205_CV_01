"""docx -> pdf (via Word) -> page PNGs (via PyMuPDF), for visual checking."""
import os
import sys
import glob

import pymupdf
import win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))
docx = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "CT3205_Assignment1_Report_rev04.docx")
pdf = os.path.splitext(docx)[0] + ".pdf"

for old in glob.glob(os.path.join(HERE, "page-*.png")):
    os.remove(old)
if os.path.exists(pdf):
    os.remove(pdf)

word = win32com.client.Dispatch("Word.Application")
word.Visible = False
try:
    doc = word.Documents.Open(docx, ReadOnly=True)
    doc.SaveAs(pdf, FileFormat=17)  # wdFormatPDF
    doc.Close(False)
finally:
    word.Quit()

d = pymupdf.open(pdf)
print("pages:", d.page_count)
for i, page in enumerate(d, 1):
    pix = page.get_pixmap(dpi=100)
    pix.save(os.path.join(HERE, f"page-{i:02d}.png"))
print("rendered", d.page_count, "pages")
