from gasture_utils.determine_gasture import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from gasture_utils.FingerPoseEstimate import FingerPoseEstimate

import numpy as np


kp = [[271.81194474, 294.46667608],
 [237.32257567, 282.60038501],
 [226.85569651, 251.14047847],
 [243.82225078, 233.01816561],
 [266.56400725, 233.13435745],
 [258.5667789 , 217.41947697],
 [258.40560666, 177.79926815],
 [258.9566953 , 154.02692726],
 [260.82724649, 136.1184036 ],
 [282.13946874, 224.39420853],
 [276.44139256, 211.42950703],
 [268.45173309, 231.26520884],
 [267.95315216, 241.15159461],
 [301.99894531, 235.82395811],
 [294.96898366, 228.60854278],
 [279.74167331, 250.01736309],
 [274.79961957, 262.26831049],
 [319.55138944, 249.77060617],
 [305.64699862, 242.90206605],
 [290.64717868, 256.57700053],
 [285.72704107, 263.94390567]]
kp = np.array(kp,dtype=np.int32)
if __name__=="__main__":
    known_finger_poses = create_known_finger_poses()
    print(kp.shape)
    fingerPoseEstimate = FingerPoseEstimate(kp)
    fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info=True)
    obtained_positions = determine_position(fingerPoseEstimate.finger_curled,
                                            fingerPoseEstimate.finger_position, known_finger_poses,
                                            0.45 * 10)
    print(obtained_positions)
    L = sorted(obtained_positions.items(),key=lambda item:item[1],reverse=True)
    print(L)
    print(L[0][0])