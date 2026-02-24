"""
Test script for MedicalRAGRetriever
Tests building, saving, loading, and retrieving from a RAG index
This code was used to test each chunk of code at a time 
and is not a full solution
"""
# %%
import pandas as pd
import logging
from pathlib import Path
from MimicRAG import MedicalRAGRetriever
import gemmaUtils as gu


def main():
    # Configuration
    import time

    start_time = time.time()

    mpath = 'E:/Education/llama/models'
    model_name = "/gemma-3n-E4B-it-Q4_K_M.gguf"
    model = mpath + model_name
    dpath = "E:/Education/CCSU-Thesis-2024/Data/"
    ipath = "E:/Education/llama/models/rag_index"
    trfile = dpath + 'intsl_train_group_full_no3c.snappy.parquet'
    tsfile = dpath + 'intsl_train_group_full_no3c.snappy.parquet'
    lpath = "E:/Education/CCSU-Thesis-2024/Data/"
    model_name = "/gemma-3n-E4B-it-Q4_K_M.gguf"
    TEXT_COLUMN = "text"  # Your clinical notes column name
    K_EXAMPLES = 5  # Number of similar examples to retrieve


    # Initiate Logger
    logger = gu.setup_logging(log_dir=lpath, log_level=logging.DEBUG)


    try:
        logger.info("="*60)
        logger.info("Starting RAG Retriever Test")
        logger.info("="*60)

    
        # 1. Load training data
        logger.info(f"\n1. Loading training data from {tsfile}")
        train_sample = 15
        test_sample = 15
        seq_size = 1000
        train_df, test_df = gu.load_data(trfile, tsfile, train_sample, test_sample, seq_size, logger)
    
        logger.info(f"   Loaded {len(train_df)} training examples")
        logger.info(f"   Columns: {train_df.columns.tolist()}")
    
        # Show sample
        if len(train_df) > 0:
            logger.info(f"\n   Sample text (first 200 chars):")
            logger.info(f"   {train_df[TEXT_COLUMN].iloc[0][:200]}...")
    
        # 2. Initialize retriever and build index
        logger.info(f"\n2. Building RAG index...")
        retriever = MedicalRAGRetriever(
            embedding_model_name=model,
            logger=logger
        )
    
        retriever.build_index(
            train_df=train_df,
            text_column=TEXT_COLUMN,
            save_path=ipath
        )
    
        logger.info(f"   Index built and saved to {ipath}")
    
        # 3. Test retrieval on a training example
        logger.info(f"\n3. Testing retrieval with a query from training set...")
        test_idx = 10  # Use 10th example as test query
        query_text = train_df[TEXT_COLUMN].iloc[test_idx]
        query_id = train_df['id'].iloc[test_idx] if 'id' in train_df.columns else None
    
        logger.info(f"   Query text (first 200 chars):")
        logger.info(f"   {query_text[:200]}...")
    
        # Retrieve without filtering
        logger.info(f"\n   Retrieving {K_EXAMPLES} most similar examples (no filtering)...")
        results_no_filter = retriever.retrieve(query_text, k=K_EXAMPLES)
    
        logger.info(f"\n   Retrieved {len(results_no_filter)} examples:")
        for i, row in results_no_filter.iterrows():
            distance = row['similarity_distance']
            text_preview = row[TEXT_COLUMN][:100].replace('\n', ' ')
            logger.info(f"   [{i}] Distance: {distance:.4f} | Text: {text_preview}...")
    
        # Retrieve with filtering (if ID column exists)
        if query_id is not None:
            logger.info(f"\n   Retrieving {K_EXAMPLES} examples (excluding same patient ID: {query_id})...")
            results_filtered = retriever.retrieve(
                query_text, 
                k=K_EXAMPLES, 
                exclude_same_id=query_id
            )
        
            logger.info(f"\n   Retrieved {len(results_filtered)} examples (filtered):")
            for i, row in results_filtered.iterrows():
                distance = row['similarity_distance']
                text_preview = row[TEXT_COLUMN][:100].replace('\n', ' ')
                patient_id = row['id'] if 'id' in row else 'N/A'
                logger.info(f"   [{i}] ID: {patient_id} | Distance: {distance:.4f} | Text: {text_preview}...")
    
        # 4. Test loading from disk
        logger.info(f"\n4. Testing index loading from disk...")
        retriever_loaded = MedicalRAGRetriever(logger=logger)
        retriever_loaded.load_index(ipath)
    
        logger.info(f"   Index loaded successfully")
        logger.info(f"   Index contains {retriever_loaded.index.ntotal} vectors")
        logger.info(f"   Examples DataFrame has {len(retriever_loaded.examples_df)} rows")
    
        # 5. Verify loaded index works
        logger.info(f"\n5. Testing retrieval with loaded index...")
        results_loaded = retriever_loaded.retrieve(query_text, k=K_EXAMPLES)
    
        logger.info(f"   Retrieved {len(results_loaded)} examples:")
        for i, row in results_loaded.iterrows():
            distance = row['similarity_distance']
            text_preview = row[TEXT_COLUMN][:100].replace('\n', ' ')
            logger.info(f"   [{i}] Distance: {distance:.4f} | Text: {text_preview}...")
    
        # 6. Verify results are identical
        logger.info(f"\n6. Verifying consistency...")
        distances_match = (results_no_filter['similarity_distance'].values == 
                          results_loaded['similarity_distance'].values).all()
    
        if distances_match:
            logger.info(f"   ✓ Results match! Index loads correctly.")
        else:
            logger.warning(f"   ✗ Results don't match - possible issue with save/load")
    
        # 7. Show file sizes
        logger.info(f"\n7. Index file information:")
        index_path = Path(ipath)
        for file in index_path.glob("*"):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   {file.name}: {size_mb:.2f} MB")
    
        logger.info("\n" + "="*60)
        logger.info("RAG Retriever Test Complete!")
        logger.info("="*60)
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        
        # Force flush all handlers
        for handler in logger.handlers:
            handler.flush()
        raise
    finally:
        logger.info("Cleanup complete")
        
        # Flush and close all handlers
        for handler in logger.handlers:
            handler.flush()
            handler.close()
    return


if __name__ == "__main__":
    main()


# %%
