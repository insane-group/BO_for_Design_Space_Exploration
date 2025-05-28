import torch

# XI = 0.01  # Adjust this value to change the exploration-exploitation balance
ITER = 120 # Adjust this value to change the number of BO iterations
PRECISION = torch.float64 # variable adju sting the precision
INITS = 3  # number of initial points from the design space to start the BO
BATCH_SIZE = 5  # Define the batch size for sampling
TOP = 10
next_threshold = 50
bo_points_dict = {}
nb_iterations = ITER

# Define exploration-exploitation parameter for acquisition function
INITIAL_XI = 0.0  # Initial value for xi
FINAL_XI = 0.0  # Final value for xi after threshold
XI_THRESHOLD = 100  # Threshold iteration to change xi
GPmodel = "single"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")