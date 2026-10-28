def get_severity(area_ratio):

    if area_ratio < 0.001:
        return 0

    elif area_ratio < 0.005:
        return 1

    elif area_ratio < 0.01:
        return 2

    else:
        return 3