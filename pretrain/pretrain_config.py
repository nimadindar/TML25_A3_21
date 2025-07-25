class PretrainConfig:
    """
    Configuration file for setting up the adversarial pretraining of a model.
    """
    MODEL = "resnet50"  # available options: 1. resnet18 2. resnet34 3. resnet50

    EPOCHS = 100
    LR = 0.1
    BATCH_SIZE = 128
    TEST_BS = 128
    OPTIMIZER = "sgd"   # available options: 1. SGD 2.Adam
    NUM_CLASSES = 1000
    MOMENTUM = 0.9
    DECAY = 0.0005

    NUM_WORKERS = 0
    NUM_GPU = 1

    TEST = False

    LOAD = ''           # Checkpoint path to resume/test.
    SAVE_PATH = "./results/pretrain_method"
    DATASET = 'imagenet'