import pickle
import os
import json

def pickle_io(func, file_path, metadata={}, rerun=True):
    '''
        file_path: File path to pickle, should NOT be a directory.
        This function also writes metadata
    '''
    def wrapper(*args, **kwargs):
        if not os.path.exists(file_path) or rerun is True:
            file_dir = file_path.rsplit(os.sep, 1)[0]
            data = func(*args, **kwargs)
            os.makedirs(file_dir, exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            with open(os.path.join(file_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        return data
    return wrapper

# Test function
def test_wrapper(func):
    def wrapper(*args, **kwargs):
        print("Testing the function")
        data = func(*args, **kwargs)
        return data

    return wrapper


if __name__ == "__main__":
    print(test_wrapper(lambda: 3+4))