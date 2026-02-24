# Remove null bytes
with open(r'E:\VSCode-Projects\Thesis\src\BERT_Diagnosis.py', 'rb') as f:
    content = f.read()

cleaned = content.replace(b'\x00', b'')

with open(r'E:\VSCode-Projects\Thesis\src\BERT_Diagnosis.py', 'wb') as f:
    f.write(cleaned)