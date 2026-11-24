import pypdf

r = pypdf.PdfReader('research-discovery/EdgeCascade_IEEE_Conference_A4.pdf')
print('Total pages:', len(r.pages))
for i in range(len(r.pages)):
    t = r.pages[i].extract_text()
    lines = [l.strip() for l in t.split('\n') if l.strip()]
    first_line = lines[0] if lines else 'EMPTY'
    last_line = lines[-1] if lines else 'EMPTY'
    print(f'Page {i+1}: start="{first_line[:40]}" | end="{last_line[:40]}"')
