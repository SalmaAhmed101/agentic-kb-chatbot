import docx

# Open the Word file. "document" is now an object representing the whole file.
document = docx.Document("data/raw/DataLine.docx")

# document.paragraphs is a list of every paragraph in the file, in order.
# Let's look at the first 15 so we don't flood the screen.
for paragraph in document.paragraphs[:15]:
    # .style.name tells us if this paragraph is "Heading 1", "Heading 2", or "Normal"
    style = paragraph.style.name
    text = paragraph.text

    # Skip empty paragraphs (Word files often have blank lines)
    if text.strip() == "":
        continue

    print(f"[{style}] {text}")