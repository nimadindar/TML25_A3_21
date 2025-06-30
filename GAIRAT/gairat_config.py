
class GAIRATconfig:

    EPOCHS = 120
    WEIGHT_DECAY = 2e-4
    MOMENTUM = 0.9
    EPSILON = 0.031
    NUM_STEPS = 10
    STEP_SIZE = 0.007
    SEED = 1
    MODEL = "resnet18"
    DATASET = "task" # Choices: ['task', 'cifar10']
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    RANDOM = True   # whether to initiat adversarial sample with random noise
    RESUME = "./results/GAIRAT_method/checkpoint.pth.tar"   # whether to resume training
    OUT_DIR = "./results/GAIRAT_method"
    LR_SCHEDULER = 'piecewise' # Choices: ['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine']
    LR_MAX = 0.1
    LR_ONE_DROP = 0.01
    LR_DROP_EPOCH = 100
    LAMBDA = '-1.0' # parameter for GAIR
    LAMBDA_MAX = float('inf')
    LAMBDA_SCHEDULE = 'fixed' # Choices: ['linear', 'piecewise', 'fixed']
    WEIGHT_ASSIGNMENT_FUNCTION = "Tanh" # Choices: ['Discrete','Sigmoid','Tanh']
    BEGIN_EPOCH = 60 # When to use GAIR