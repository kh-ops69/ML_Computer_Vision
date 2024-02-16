# can add more body part movement functions as needed
def verify_arms_up(points):
    head, right_wrist, left_wrist = 0, 0, 0
    for i, point in enumerate(points):
        # print(i, point)
        if i == 0:
            head = point[1]
        elif i == 4:
            right_wrist = point[1]
        elif i == 7:
            left_wrist = point[1]

    # print(head, right_wrist, left_wrist)
    if right_wrist < head and left_wrist < head:
        return True
    else:
        return False

# sample result for verify arms up- the main file appends coordinates of each body part and its int id 'i' in list points
# the points themselves should have a difference as expected above , and i is checking for which body part number it is-

# [(760, 180), (760, 292), (680, 382), (560, 517), (480, 495), (880, 405), (920, 585), (920, 697), (640, 675), (840, 630), (560, 517), (760, 697), (840, 607), (920, 697)]
# 0.6653487086296082

def verify_legs_apart(points):
  left_hip, right_hip = 0, 0
  left_ankle, right_ankle = 0, 0

  for i, point in enumerate(points):
    if i == 11:
      left_hip = point[0]
    elif i == 8:
      right_hip = point[0]
    elif i == 13:
      left_ankle = point[0]
    elif i == 10:
      right_ankle = point[0]

  if (left_ankle > left_hip) and (right_ankle < right_hip):
    return True
  else:
    return False