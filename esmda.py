import numpy as np
def ES(OBS, static, dynamic, alpha=1, stdErrOfDynamic_percentage=0.05, add_noise = False):
    """
    Ensemble Smoother function that assimilates observations into an ensemble.

    Parameters:
    OBS (numpy.ndarray): Array of observations.
    static (numpy.ndarray): Array of static variables. should be provided (number of ensemble x number of static variables).
    dynamic (numpy.ndarray): Array of dynamic variables. should be provided (number of ensemble x number of dynamic variables).
    alpha (float): Smoothing factor (default is 1).
    stdErrOfDynamic (float): Standard error of dynamic variables (default is 0.1).
    add_noise (bool): Add noise to dynamic variables (default is False).

    Returns:
    numpy.ndarray: Array of updated static variables.
    """

    # Reshape static and dynamic arrays
    static = static.reshape(static.shape[0], -1).T
    dynamic = dynamic.reshape(dynamic.shape[0], -1).T
    stdErrOfDynamic = stdErrOfDynamic_percentage * dynamic.std(axis=1)
    if add_noise:
        dynamic = dynamic + np.random.normal(loc = 0, scale = stdErrOfDynamic.reshape(-1,1), size = dynamic.shape)

    # Concatenate static and dynamic arrays
    ensemble = np.concatenate((static, dynamic), axis=0)
    No_realization = ensemble.shape[1]

    # Calculate ensemble mean
    En_Mean = ensemble.mean(axis=1).reshape(-1, 1)
    En_Mean = np.repeat(En_Mean, No_realization, axis=1)

    # Reshape OBS array
    OBS = OBS.reshape(-1, 1)
    num_static = static.shape[0]
    num_dynamic = dynamic.shape[0]
    num_state_vector = num_dynamic + num_static
    ref_OBS = np.repeat(OBS, No_realization, axis=1)


    ## This is where we apply EnKF/ES:
    # Create Cd matrix
    sizeOfCd = num_dynamic
    Cd = np.zeros((sizeOfCd, sizeOfCd))
    for i in range(sizeOfCd):
        Cd[i, i] = stdErrOfDynamic[i] ** 2

    # Create H matrix
    H = np.zeros((num_dynamic, num_state_vector))
    for i in range(num_dynamic):
        H[i, num_static + i] = 1

    # Calculate Le and H_Le
    Le = 1 / np.sqrt(num_state_vector - 1) * (ensemble - En_Mean)
    H_Le = np.dot(H, Le)

    # Calculate Kalman Gain, K
    K_ = np.dot(np.dot(Le, H_Le.T), np.linalg.inv(np.dot(H_Le, H_Le.T) + Cd * alpha))

    # Add stdErroOfDynamic to refOBS
    for i in range(num_dynamic):
        for j in range(No_realization):
            ref_OBS[i, j] = ref_OBS[i, j] + np.random.normal(scale=stdErrOfDynamic[i] * (alpha) ** (1/2))

    # Assimilate En
    ensemble_new = ensemble + np.dot(K_, ref_OBS - np.dot(H, ensemble))
    static_new = ensemble_new[:num_static, :]

    return static_new.T 
