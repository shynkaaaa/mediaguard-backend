class Config:
    # Paths
    BEST_MODEL_PATH  = None  # set dynamically in detector.py

    # Data
    IMAGE_SIZE       = 224

    # Model
    NUM_CLASSES      = 2
    DROPOUT_RATE     = 0.3
    DROP_CONNECT     = 0.1
