def print_matlab_format(arr, name=None):
    """
    Print a numpy array in MATLAB format.
    
    Args:
        arr: numpy array (1D or 2D)
        name: optional name for the array
    """
    if name:
        print(f"{name} = ", end="")
    
    if arr.ndim == 1:
        # 1D array
        print("[", end="")
        print("; ".join(f"{x:.6f}" for x in arr), end="")
        print("];")
    elif arr.ndim == 2:
        # 2D array
        print("[", end="")
        rows = []
        for i in range(arr.shape[0]):
            row = " ".join(f"{x:.6f}" for x in arr[i])
            rows.append(row)
        print("; \n".join(rows), end="")
        print("];")
    else:
        print("Error: Only 1D and 2D arrays are supported")