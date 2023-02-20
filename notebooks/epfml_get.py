import torch
import epfml.store
import sys

if __name__ == "__main__":
    accs = epfml.store.get(sys.argv[1])
    torch.save(accs, str(sys.argv[1]))

# Usage: python epfml_get.py my_key
