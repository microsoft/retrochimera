import os

# Needed for cases when reaction data preprocessing does not release the lock on the `*.h5` file.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
