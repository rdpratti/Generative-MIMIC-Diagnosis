REM Navigate to your script directory
cd /d "E:\vscode-projects\thesis\src"
call conda activate rag_thesis

REM Run Python script with arguments
python Gemma_Diagnosis.py --temperature 0.1 --train_ct 100 --test_ct 10 --train_seq_size 300 --test_seq_size 3000 --example_size 425 --test_size 5000 --example_ct 8 --use_rag 

echo.
echo Experiment completed!
pause
