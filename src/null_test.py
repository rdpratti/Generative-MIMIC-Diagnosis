with open(r'E:\VSCode-Projects\Thesis\src\BERT_Diagnosis.py', 'rb') as f:
    content = f.read()
    null_positions = [i for i, byte in enumerate(content) if byte == 0]
    if null_positions:
        print(f"Found {len(null_positions)} null bytes at positions: {null_positions[:10]}")
    else:
        print("No null bytes found")