import fitz
doc = fitz.open('Dataset.pdf')
total_blocks = 0
for page in doc:
    blocks = page.get_text('dict')['blocks']
    total_blocks += len(blocks)
print(f"Total blocks in PDF: {total_blocks}")
