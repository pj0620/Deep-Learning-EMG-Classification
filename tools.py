def find_max_length(data):
    max_length = 0
    length = 0
    cur = 0
    for subject in range(32):
        for session in range(2):
            X = data[subject][session]
            for k in range(X.shape[0]):
                if X[k, -1] == cur:
                    length += 1
                else:
                    if cur != 0 and length > max_length:
                        max_length = length
                        print(f"new max length = {max_length}, class = {cur}")
                    cur = X[k, -1]
                    length = 0