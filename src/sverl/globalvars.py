RESET_SEED = 42
CP_STATE_FEATURE_NAMES = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
CP_RANGES = [(-2.4, 2.4), ("ninf", "inf"), (-0.2095, 0.2095), ("ninf", "inf")]
POS_VEL_G = [[0, 2], [1, 3]]  # Grouped positional features and grouped velocity features

"__all__" == [CP_RANGES, CP_STATE_FEATURE_NAMES, RESET_SEED]
