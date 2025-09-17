# THIS FILE HAS SIDE EFFECTS, DON'T IMPORT IN TOP OF FILES
from pydantic import BaseModel

from typing import Literal
import torch

from .datatypes import LinkInfo

GOLIATH_NUM_KEYPOINTS = 308

# sapiens palette
# fmt: off
COLOR_DODGE_BLUE  = (51,153,255)
COLOR_GREEN       = (0,255,0)
COLOR_ORANGE      = (255,128,0)
COLOR_PINK        = (255,153,255)
COLOR_SKY_BLUE    = (102,178,255)
COLOR_RED         = (255,51,51)
COLOR_MULBERRY    = (192,64,128)
COLOR_INDIGO      = (64,32,192)
COLOR_TEAL        = (64,192,128)
COLOR_LIME_GREEN  = (64,192,32)
COLOR_DARK_GREEN  = (0,192,0)
COLOR_MAROON      = (192,0,0)
COLOR_TURQUOISE   = (0,192,192)
COLOR_YELLOW      = (200,200,0)
COLOR_CYAN        = (0,200,200)
COLOR_OLIVE_GREEN = (128,192,64)
COLOR_BURGUNDY    = (192,32,64)
COLOR_BROWN       = (192,128,64)
COLOR_LIGHT_CYAN  = (32,192,192)
COLOR_WHITE       = (255,255,255)

COCO_KPTS_COLORS = [
    COLOR_DODGE_BLUE,  # 0: nose
    COLOR_DODGE_BLUE,  # 1: left_eye
    COLOR_DODGE_BLUE,  # 2: right_eye
    COLOR_DODGE_BLUE,  # 3: left_ear
    COLOR_DODGE_BLUE,  # 4: right_ear
    COLOR_GREEN,       # 5: left_shoulder
    COLOR_ORANGE,      # 6: right_shoulder
    COLOR_GREEN,       # 7: left_elbow
    COLOR_ORANGE,      # 8: right_elbow
    COLOR_GREEN,       # 9: left_wrist
    COLOR_ORANGE,      # 10: right_wrist
    COLOR_GREEN,       # 11: left_hip
    COLOR_ORANGE,      # 12: right_hip
    COLOR_GREEN,       # 13: left_knee
    COLOR_ORANGE,      # 14: right_knee
    COLOR_GREEN,       # 15: left_ankle
    COLOR_ORANGE,      # 16: right_ankle
]

COCO_WHOLEBODY_KPTS_COLORS = [
    COLOR_DODGE_BLUE,  # 0: nose
    COLOR_DODGE_BLUE,  # 1: left_eye
    COLOR_DODGE_BLUE,  # 2: right_eye
    COLOR_DODGE_BLUE,  # 3: left_ear
    COLOR_DODGE_BLUE,  # 4: right_ear
    COLOR_GREEN,       # 5: left_shoulder
    COLOR_ORANGE,      # 6: right_shoulder
    COLOR_GREEN,       # 7: left_elbow
    COLOR_ORANGE,      # 8: right_elbow
    COLOR_GREEN,       # 9: left_wrist
    COLOR_ORANGE,      # 10: right_wrist
    COLOR_GREEN,       # 11: left_hip
    COLOR_ORANGE,      # 12: right_hip
    COLOR_GREEN,       # 13: left_knee
    COLOR_ORANGE,      # 14: right_knee
    COLOR_GREEN,       # 15: left_ankle
    COLOR_ORANGE,      # 16: right_ankle
    COLOR_ORANGE,      # 17: left_big_toe
    COLOR_ORANGE,      # 18: left_small_toe
    COLOR_ORANGE,      # 19: left_heel
    COLOR_ORANGE,      # 20: right_big_toe
    COLOR_ORANGE,      # 21: right_small_toe
    COLOR_ORANGE,      # 22: right_heel
    COLOR_WHITE,       # 23: face-0
    COLOR_WHITE,       # 24: face-1
    COLOR_WHITE,       # 25: face-2
    COLOR_WHITE,       # 26: face-3
    COLOR_WHITE,       # 27: face-4
    COLOR_WHITE,       # 28: face-5
    COLOR_WHITE,       # 29: face-6
    COLOR_WHITE,       # 30: face-7
    COLOR_WHITE,       # 31: face-8
    COLOR_WHITE,       # 32: face-9
    COLOR_WHITE,       # 33: face-10
    COLOR_WHITE,       # 34: face-11
    COLOR_WHITE,       # 35: face-12
    COLOR_WHITE,       # 36: face-13
    COLOR_WHITE,       # 37: face-14
    COLOR_WHITE,       # 38: face-15
    COLOR_WHITE,       # 39: face-16
    COLOR_WHITE,       # 40: face-17
    COLOR_WHITE,       # 41: face-18
    COLOR_WHITE,       # 42: face-19
    COLOR_WHITE,       # 43: face-20
    COLOR_WHITE,       # 44: face-21
    COLOR_WHITE,       # 45: face-22
    COLOR_WHITE,       # 46: face-23
    COLOR_WHITE,       # 47: face-24
    COLOR_WHITE,       # 48: face-25
    COLOR_WHITE,       # 49: face-26
    COLOR_WHITE,       # 50: face-27
    COLOR_WHITE,       # 51: face-28
    COLOR_WHITE,       # 52: face-29
    COLOR_WHITE,       # 53: face-30
    COLOR_WHITE,       # 54: face-31
    COLOR_WHITE,       # 55: face-32
    COLOR_WHITE,       # 56: face-33
    COLOR_WHITE,       # 57: face-34
    COLOR_WHITE,       # 58: face-35
    COLOR_WHITE,       # 59: face-36
    COLOR_WHITE,       # 60: face-37
    COLOR_WHITE,       # 61: face-38
    COLOR_WHITE,       # 62: face-39
    COLOR_WHITE,       # 63: face-40
    COLOR_WHITE,       # 64: face-41
    COLOR_WHITE,       # 65: face-42
    COLOR_WHITE,       # 66: face-43
    COLOR_WHITE,       # 67: face-44
    COLOR_WHITE,       # 68: face-45
    COLOR_WHITE,       # 69: face-46
    COLOR_WHITE,       # 70: face-47
    COLOR_WHITE,       # 71: face-48
    COLOR_WHITE,       # 72: face-49
    COLOR_WHITE,       # 73: face-50
    COLOR_WHITE,       # 74: face-51
    COLOR_WHITE,       # 75: face-52
    COLOR_WHITE,       # 76: face-53
    COLOR_WHITE,       # 77: face-54
    COLOR_WHITE,       # 78: face-55
    COLOR_WHITE,       # 79: face-56
    COLOR_WHITE,       # 80: face-57
    COLOR_WHITE,       # 81: face-58
    COLOR_WHITE,       # 82: face-59
    COLOR_WHITE,       # 83: face-60
    COLOR_WHITE,       # 84: face-61
    COLOR_WHITE,       # 85: face-62
    COLOR_WHITE,       # 86: face-63
    COLOR_WHITE,       # 87: face-64
    COLOR_WHITE,       # 88: face-65
    COLOR_WHITE,       # 89: face-66
    COLOR_WHITE,       # 90: face-67
    COLOR_WHITE,       # 91: left_hand_root
    COLOR_ORANGE,      # 92: left_thumb1
    COLOR_ORANGE,      # 93: left_thumb2
    COLOR_ORANGE,      # 94: left_thumb3
    COLOR_ORANGE,      # 95: left_thumb4
    COLOR_PINK,        # 96: left_forefinger1
    COLOR_PINK,        # 97: left_forefinger2
    COLOR_PINK,        # 98: left_forefinger3
    COLOR_PINK,        # 99: left_forefinger4
    COLOR_SKY_BLUE,    # 100: left_middle_finger1
    COLOR_SKY_BLUE,    # 101: left_middle_finger2
    COLOR_SKY_BLUE,    # 102: left_middle_finger3
    COLOR_SKY_BLUE,    # 103: left_middle_finger4
    COLOR_RED,         # 104: left_ring_finger1
    COLOR_RED,         # 105: left_ring_finger2
    COLOR_RED,         # 106: left_ring_finger3
    COLOR_RED,         # 107: left_ring_finger4
    COLOR_GREEN,       # 108: left_pinky_finger1
    COLOR_GREEN,       # 109: left_pinky_finger2
    COLOR_GREEN,       # 110: left_pinky_finger3
    COLOR_GREEN,       # 111: left_pinky_finger4
    COLOR_WHITE,       # 112: right_hand_root
    COLOR_ORANGE,      # 113: right_thumb1
    COLOR_ORANGE,      # 114: right_thumb2
    COLOR_ORANGE,      # 115: right_thumb3
    COLOR_ORANGE,      # 116: right_thumb4
    COLOR_PINK,        # 117: right_forefinger1
    COLOR_PINK,        # 118: right_forefinger2
    COLOR_PINK,        # 119: right_forefinger3
    COLOR_PINK,        # 120: right_forefinger4
    COLOR_SKY_BLUE,    # 121: right_middle_finger1
    COLOR_SKY_BLUE,    # 122: right_middle_finger2
    COLOR_SKY_BLUE,    # 123: right_middle_finger3
    COLOR_SKY_BLUE,    # 124: right_middle_finger4
    COLOR_RED,         # 125: right_ring_finger1
    COLOR_RED,         # 126: right_ring_finger2
    COLOR_RED,         # 127: right_ring_finger3
    COLOR_RED,         # 128: right_ring_finger4
    COLOR_GREEN,       # 129: right_pinky_finger1
    COLOR_GREEN,       # 130: right_pinky_finger2
    COLOR_GREEN,       # 131: right_pinky_finger3
    COLOR_GREEN,       # 132: right_pinky_finger4
]


GOLIATH_KPTS_COLORS = [
    COLOR_DODGE_BLUE,   # 0: nose
    COLOR_DODGE_BLUE,   # 1: left_eye
    COLOR_DODGE_BLUE,   # 2: right_eye
    COLOR_DODGE_BLUE,   # 3: left_ear
    COLOR_DODGE_BLUE,   # 4: right_ear
    COLOR_DODGE_BLUE,   # 5: left_shoulder
    COLOR_DODGE_BLUE,   # 6: right_shoulder
    COLOR_DODGE_BLUE,   # 7: left_elbow
    COLOR_DODGE_BLUE,   # 8: right_elbow
    COLOR_DODGE_BLUE,   # 9: left_hip
    COLOR_DODGE_BLUE,   # 10: right_hip
    COLOR_DODGE_BLUE,   # 11: left_knee
    COLOR_DODGE_BLUE,   # 12: right_knee
    COLOR_DODGE_BLUE,   # 13: left_ankle
    COLOR_DODGE_BLUE,   # 14: right_ankle
    COLOR_DODGE_BLUE,   # 15: left_big_toe
    COLOR_DODGE_BLUE,   # 16: left_small_toe
    COLOR_DODGE_BLUE,   # 17: left_heel
    COLOR_DODGE_BLUE,   # 18: right_big_toe
    COLOR_DODGE_BLUE,   # 19: right_small_toe
    COLOR_DODGE_BLUE,   # 20: right_heel
    COLOR_DODGE_BLUE,   # 21: right_thumb4
    COLOR_DODGE_BLUE,   # 22: right_thumb3
    COLOR_DODGE_BLUE,   # 23: right_thumb2
    COLOR_DODGE_BLUE,   # 24: right_thumb_third_joint
    COLOR_DODGE_BLUE,   # 25: right_forefinger4
    COLOR_DODGE_BLUE,   # 26: right_forefinger3
    COLOR_DODGE_BLUE,   # 27: right_forefinger2
    COLOR_DODGE_BLUE,   # 28: right_forefinger_third_joint
    COLOR_DODGE_BLUE,   # 29: right_middle_finger4
    COLOR_DODGE_BLUE,   # 30: right_middle_finger3
    COLOR_DODGE_BLUE,   # 31: right_middle_finger2
    COLOR_DODGE_BLUE,   # 32: right_middle_finger_third_joint
    COLOR_DODGE_BLUE,   # 33: right_ring_finger4
    COLOR_DODGE_BLUE,   # 34: right_ring_finger3
    COLOR_DODGE_BLUE,   # 35: right_ring_finger2
    COLOR_DODGE_BLUE,   # 36: right_ring_finger_third_joint
    COLOR_DODGE_BLUE,   # 37: right_pinky_finger4
    COLOR_DODGE_BLUE,   # 38: right_pinky_finger3
    COLOR_DODGE_BLUE,   # 39: right_pinky_finger2
    COLOR_DODGE_BLUE,   # 40: right_pinky_finger_third_joint
    COLOR_DODGE_BLUE,   # 41: right_wrist
    COLOR_DODGE_BLUE,   # 42: left_thumb4
    COLOR_DODGE_BLUE,   # 43: left_thumb3
    COLOR_DODGE_BLUE,   # 44: left_thumb2
    COLOR_DODGE_BLUE,   # 45: left_thumb_third_joint
    COLOR_DODGE_BLUE,   # 46: left_forefinger4
    COLOR_DODGE_BLUE,   # 47: left_forefinger3
    COLOR_DODGE_BLUE,   # 48: left_forefinger2
    COLOR_DODGE_BLUE,   # 49: left_forefinger_third_joint
    COLOR_DODGE_BLUE,   # 50: left_middle_finger4
    COLOR_DODGE_BLUE,   # 51: left_middle_finger3
    COLOR_DODGE_BLUE,   # 52: left_middle_finger2
    COLOR_DODGE_BLUE,   # 53: left_middle_finger_third_joint
    COLOR_DODGE_BLUE,   # 54: left_ring_finger4
    COLOR_DODGE_BLUE,   # 55: left_ring_finger3
    COLOR_DODGE_BLUE,   # 56: left_ring_finger2
    COLOR_DODGE_BLUE,   # 57: left_ring_finger_third_joint
    COLOR_DODGE_BLUE,   # 58: left_pinky_finger4
    COLOR_DODGE_BLUE,   # 59: left_pinky_finger3
    COLOR_DODGE_BLUE,   # 60: left_pinky_finger2
    COLOR_DODGE_BLUE,   # 61: left_pinky_finger_third_joint
    COLOR_DODGE_BLUE,   # 62: left_wrist
    COLOR_DODGE_BLUE,   # 63: left_olecranon
    COLOR_DODGE_BLUE,   # 64: right_olecranon
    COLOR_DODGE_BLUE,   # 65: left_cubital_fossa
    COLOR_DODGE_BLUE,   # 66: right_cubital_fossa
    COLOR_DODGE_BLUE,   # 67: left_acromion
    COLOR_DODGE_BLUE,   # 68: right_acromion
    COLOR_DODGE_BLUE,   # 69: neck
    COLOR_WHITE,        # 70: center_of_glabella
    COLOR_WHITE,        # 71: center_of_nose_root
    COLOR_WHITE,        # 72: tip_of_nose_bridge
    COLOR_WHITE,        # 73: midpoint_1_of_nose_bridge
    COLOR_WHITE,        # 74: midpoint_2_of_nose_bridge
    COLOR_WHITE,        # 75: midpoint_3_of_nose_bridge
    COLOR_WHITE,        # 76: center_of_labiomental_groove
    COLOR_WHITE,        # 77: tip_of_chin
    COLOR_WHITE,        # 78: upper_startpoint_of_r_eyebrow
    COLOR_WHITE,        # 79: lower_startpoint_of_r_eyebrow
    COLOR_WHITE,        # 80: end_of_r_eyebrow
    COLOR_WHITE,        # 81: upper_midpoint_1_of_r_eyebrow
    COLOR_WHITE,        # 82: lower_midpoint_1_of_r_eyebrow
    COLOR_WHITE,        # 83: upper_midpoint_2_of_r_eyebrow
    COLOR_WHITE,        # 84: upper_midpoint_3_of_r_eyebrow
    COLOR_WHITE,        # 85: lower_midpoint_2_of_r_eyebrow
    COLOR_WHITE,        # 86: lower_midpoint_3_of_r_eyebrow
    COLOR_WHITE,        # 87: upper_startpoint_of_l_eyebrow
    COLOR_WHITE,        # 88: lower_startpoint_of_l_eyebrow
    COLOR_WHITE,        # 89: end_of_l_eyebrow
    COLOR_WHITE,        # 90: upper_midpoint_1_of_l_eyebrow
    COLOR_WHITE,        # 91: lower_midpoint_1_of_l_eyebrow
    COLOR_WHITE,        # 92: upper_midpoint_2_of_l_eyebrow
    COLOR_WHITE,        # 93: upper_midpoint_3_of_l_eyebrow
    COLOR_WHITE,        # 94: lower_midpoint_2_of_l_eyebrow
    COLOR_WHITE,        # 95: lower_midpoint_3_of_l_eyebrow
    COLOR_MULBERRY,     # 96: l_inner_end_of_upper_lash_line
    COLOR_MULBERRY,     # 97: l_outer_end_of_upper_lash_line
    COLOR_MULBERRY,     # 98: l_centerpoint_of_upper_lash_line
    COLOR_MULBERRY,     # 99: l_midpoint_2_of_upper_lash_line
    COLOR_MULBERRY,     # 100: l_midpoint_1_of_upper_lash_line
    COLOR_MULBERRY,     # 101: l_midpoint_6_of_upper_lash_line
    COLOR_MULBERRY,     # 102: l_midpoint_5_of_upper_lash_line
    COLOR_MULBERRY,     # 103: l_midpoint_4_of_upper_lash_line
    COLOR_MULBERRY,     # 104: l_midpoint_3_of_upper_lash_line
    COLOR_MULBERRY,     # 105: l_outer_end_of_upper_eyelid_line
    COLOR_MULBERRY,     # 106: l_midpoint_6_of_upper_eyelid_line
    COLOR_MULBERRY,     # 107: l_midpoint_2_of_upper_eyelid_line
    COLOR_MULBERRY,     # 108: l_midpoint_5_of_upper_eyelid_line
    COLOR_MULBERRY,     # 109: l_centerpoint_of_upper_eyelid_line
    COLOR_MULBERRY,     # 110: l_midpoint_4_of_upper_eyelid_line
    COLOR_MULBERRY,     # 111: l_midpoint_1_of_upper_eyelid_line
    COLOR_MULBERRY,     # 112: l_midpoint_3_of_upper_eyelid_line
    COLOR_MULBERRY,     # 113: l_midpoint_6_of_upper_crease_line
    COLOR_MULBERRY,     # 114: l_midpoint_2_of_upper_crease_line
    COLOR_MULBERRY,     # 115: l_midpoint_5_of_upper_crease_line
    COLOR_MULBERRY,     # 116: l_centerpoint_of_upper_crease_line
    COLOR_MULBERRY,     # 117: l_midpoint_4_of_upper_crease_line
    COLOR_MULBERRY,     # 118: l_midpoint_1_of_upper_crease_line
    COLOR_MULBERRY,     # 119: l_midpoint_3_of_upper_crease_line
    COLOR_INDIGO,       # 120: r_inner_end_of_upper_lash_line
    COLOR_INDIGO,       # 121: r_outer_end_of_upper_lash_line
    COLOR_INDIGO,       # 122: r_centerpoint_of_upper_lash_line
    COLOR_INDIGO,       # 123: r_midpoint_1_of_upper_lash_line
    COLOR_INDIGO,       # 124: r_midpoint_2_of_upper_lash_line
    COLOR_INDIGO,       # 125: r_midpoint_3_of_upper_lash_line
    COLOR_INDIGO,       # 126: r_midpoint_4_of_upper_lash_line
    COLOR_INDIGO,       # 127: r_midpoint_5_of_upper_lash_line
    COLOR_INDIGO,       # 128: r_midpoint_6_of_upper_lash_line
    COLOR_INDIGO,       # 129: r_outer_end_of_upper_eyelid_line
    COLOR_INDIGO,       # 130: r_midpoint_3_of_upper_eyelid_line
    COLOR_INDIGO,       # 131: r_midpoint_1_of_upper_eyelid_line
    COLOR_INDIGO,       # 132: r_midpoint_4_of_upper_eyelid_line
    COLOR_INDIGO,       # 133: r_centerpoint_of_upper_eyelid_line
    COLOR_INDIGO,       # 134: r_midpoint_5_of_upper_eyelid_line
    COLOR_INDIGO,       # 135: r_midpoint_2_of_upper_eyelid_line
    COLOR_INDIGO,       # 136: r_midpoint_6_of_upper_eyelid_line
    COLOR_INDIGO,       # 137: r_midpoint_3_of_upper_crease_line
    COLOR_INDIGO,       # 138: r_midpoint_1_of_upper_crease_line
    COLOR_INDIGO,       # 139: r_midpoint_4_of_upper_crease_line
    COLOR_INDIGO,       # 140: r_centerpoint_of_upper_crease_line
    COLOR_INDIGO,       # 141: r_midpoint_5_of_upper_crease_line
    COLOR_INDIGO,       # 142: r_midpoint_2_of_upper_crease_line
    COLOR_INDIGO,       # 143: r_midpoint_6_of_upper_crease_line
    COLOR_TEAL,         # 144: l_inner_end_of_lower_lash_line
    COLOR_TEAL,         # 145: l_outer_end_of_lower_lash_line
    COLOR_TEAL,         # 146: l_centerpoint_of_lower_lash_line
    COLOR_TEAL,         # 147: l_midpoint_2_of_lower_lash_line
    COLOR_TEAL,         # 148: l_midpoint_1_of_lower_lash_line
    COLOR_TEAL,         # 149: l_midpoint_6_of_lower_lash_line
    COLOR_TEAL,         # 150: l_midpoint_5_of_lower_lash_line
    COLOR_TEAL,         # 151: l_midpoint_4_of_lower_lash_line
    COLOR_TEAL,         # 152: l_midpoint_3_of_lower_lash_line
    COLOR_TEAL,         # 153: l_outer_end_of_lower_eyelid_line
    COLOR_TEAL,         # 154: l_midpoint_6_of_lower_eyelid_line
    COLOR_TEAL,         # 155: l_midpoint_2_of_lower_eyelid_line
    COLOR_TEAL,         # 156: l_midpoint_5_of_lower_eyelid_line
    COLOR_TEAL,         # 157: l_centerpoint_of_lower_eyelid_line
    COLOR_TEAL,         # 158: l_midpoint_4_of_lower_eyelid_line
    COLOR_TEAL,         # 159: l_midpoint_1_of_lower_eyelid_line
    COLOR_TEAL,         # 160: l_midpoint_3_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 161: r_inner_end_of_lower_lash_line
    COLOR_LIME_GREEN,   # 162: r_outer_end_of_lower_lash_line
    COLOR_LIME_GREEN,   # 163: r_centerpoint_of_lower_lash_line
    COLOR_LIME_GREEN,   # 164: r_midpoint_1_of_lower_lash_line
    COLOR_LIME_GREEN,   # 165: r_midpoint_2_of_lower_lash_line
    COLOR_LIME_GREEN,   # 166: r_midpoint_3_of_lower_lash_line
    COLOR_LIME_GREEN,   # 167: r_midpoint_4_of_lower_lash_line
    COLOR_LIME_GREEN,   # 168: r_midpoint_5_of_lower_lash_line
    COLOR_LIME_GREEN,   # 169: r_midpoint_6_of_lower_lash_line
    COLOR_LIME_GREEN,   # 170: r_outer_end_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 171: r_midpoint_3_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 172: r_midpoint_1_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 173: r_midpoint_4_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 174: r_centerpoint_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 175: r_midpoint_5_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 176: r_midpoint_2_of_lower_eyelid_line
    COLOR_LIME_GREEN,   # 177: r_midpoint_6_of_lower_eyelid_line
    COLOR_DARK_GREEN,   # 178: tip_of_nose
    COLOR_DARK_GREEN,   # 179: bottom_center_of_nose
    COLOR_DARK_GREEN,   # 180: r_outer_corner_of_nose
    COLOR_DARK_GREEN,   # 181: l_outer_corner_of_nose
    COLOR_DARK_GREEN,   # 182: inner_corner_of_r_nostril
    COLOR_DARK_GREEN,   # 183: outer_corner_of_r_nostril
    COLOR_DARK_GREEN,   # 184: upper_corner_of_r_nostril
    COLOR_DARK_GREEN,   # 185: inner_corner_of_l_nostril
    COLOR_DARK_GREEN,   # 186: outer_corner_of_l_nostril
    COLOR_DARK_GREEN,   # 187: upper_corner_of_l_nostril
    COLOR_MAROON,       # 188: r_outer_corner_of_mouth
    COLOR_MAROON,       # 189: l_outer_corner_of_mouth
    COLOR_MAROON,       # 190: center_of_cupid_bow
    COLOR_MAROON,       # 191: center_of_lower_outer_lip
    COLOR_MAROON,       # 192: midpoint_1_of_upper_outer_lip
    COLOR_MAROON,       # 193: midpoint_2_of_upper_outer_lip
    COLOR_MAROON,       # 194: midpoint_1_of_lower_outer_lip
    COLOR_MAROON,       # 195: midpoint_2_of_lower_outer_lip
    COLOR_MAROON,       # 196: midpoint_3_of_upper_outer_lip
    COLOR_MAROON,       # 197: midpoint_4_of_upper_outer_lip
    COLOR_MAROON,       # 198: midpoint_5_of_upper_outer_lip
    COLOR_MAROON,       # 199: midpoint_6_of_upper_outer_lip
    COLOR_MAROON,       # 200: midpoint_3_of_lower_outer_lip
    COLOR_MAROON,       # 201: midpoint_4_of_lower_outer_lip
    COLOR_MAROON,       # 202: midpoint_5_of_lower_outer_lip
    COLOR_MAROON,       # 203: midpoint_6_of_lower_outer_lip
    COLOR_TURQUOISE,    # 204: r_inner_corner_of_mouth
    COLOR_TURQUOISE,    # 205: l_inner_corner_of_mouth
    COLOR_TURQUOISE,    # 206: center_of_upper_inner_lip
    COLOR_TURQUOISE,    # 207: center_of_lower_inner_lip
    COLOR_TURQUOISE,    # 208: midpoint_1_of_upper_inner_lip
    COLOR_TURQUOISE,    # 209: midpoint_2_of_upper_inner_lip
    COLOR_TURQUOISE,    # 210: midpoint_1_of_lower_inner_lip
    COLOR_TURQUOISE,    # 211: midpoint_2_of_lower_inner_lip
    COLOR_TURQUOISE,    # 212: midpoint_3_of_upper_inner_lip
    COLOR_TURQUOISE,    # 213: midpoint_4_of_upper_inner_lip
    COLOR_TURQUOISE,    # 214: midpoint_5_of_upper_inner_lip
    COLOR_TURQUOISE,    # 215: midpoint_6_of_upper_inner_lip
    COLOR_TURQUOISE,    # 216: midpoint_3_of_lower_inner_lip
    COLOR_TURQUOISE,    # 217: midpoint_4_of_lower_inner_lip
    COLOR_TURQUOISE,    # 218: midpoint_5_of_lower_inner_lip
    COLOR_TURQUOISE,    # 219: midpoint_6_of_lower_inner_lip. teeths removed
    COLOR_YELLOW,       # 256: l_top_end_of_inferior_crus
    COLOR_YELLOW,       # 257: l_top_end_of_superior_crus
    COLOR_YELLOW,       # 258: l_start_of_antihelix
    COLOR_YELLOW,       # 259: l_end_of_antihelix
    COLOR_YELLOW,       # 260: l_midpoint_1_of_antihelix
    COLOR_YELLOW,       # 261: l_midpoint_1_of_inferior_crus
    COLOR_YELLOW,       # 262: l_midpoint_2_of_antihelix
    COLOR_YELLOW,       # 263: l_midpoint_3_of_antihelix
    COLOR_YELLOW,       # 264: l_point_1_of_inner_helix
    COLOR_YELLOW,       # 265: l_point_2_of_inner_helix
    COLOR_YELLOW,       # 266: l_point_3_of_inner_helix
    COLOR_YELLOW,       # 267: l_point_4_of_inner_helix
    COLOR_YELLOW,       # 268: l_point_5_of_inner_helix
    COLOR_YELLOW,       # 269: l_point_6_of_inner_helix
    COLOR_YELLOW,       # 270: l_point_7_of_inner_helix
    COLOR_YELLOW,       # 271: l_highest_point_of_antitragus
    COLOR_YELLOW,       # 272: l_bottom_point_of_tragus
    COLOR_YELLOW,       # 273: l_protruding_point_of_tragus
    COLOR_YELLOW,       # 274: l_top_point_of_tragus
    COLOR_YELLOW,       # 275: l_start_point_of_crus_of_helix
    COLOR_YELLOW,       # 276: l_deepest_point_of_concha
    COLOR_YELLOW,       # 277: l_tip_of_ear_lobe
    COLOR_YELLOW,       # 278: l_midpoint_between_22_15
    COLOR_YELLOW,       # 279: l_bottom_connecting_point_of_ear_lobe
    COLOR_YELLOW,       # 280: l_top_connecting_point_of_helix
    COLOR_YELLOW,       # 281: l_point_8_of_inner_helix
    COLOR_CYAN,         # 282: r_top_end_of_inferior_crus
    COLOR_CYAN,         # 283: r_top_end_of_superior_crus
    COLOR_CYAN,         # 284: r_start_of_antihelix
    COLOR_CYAN,         # 285: r_end_of_antihelix
    COLOR_CYAN,         # 286: r_midpoint_1_of_antihelix
    COLOR_CYAN,         # 287: r_midpoint_1_of_inferior_crus
    COLOR_CYAN,         # 288: r_midpoint_2_of_antihelix
    COLOR_CYAN,         # 289: r_midpoint_3_of_antihelix
    COLOR_CYAN,         # 290: r_point_1_of_inner_helix
    COLOR_CYAN,         # 291: r_point_8_of_inner_helix
    COLOR_CYAN,         # 292: r_point_3_of_inner_helix
    COLOR_CYAN,         # 293: r_point_4_of_inner_helix
    COLOR_CYAN,         # 294: r_point_5_of_inner_helix
    COLOR_CYAN,         # 295: r_point_6_of_inner_helix
    COLOR_CYAN,         # 296: r_point_7_of_inner_helix
    COLOR_CYAN,         # 297: r_highest_point_of_antitragus
    COLOR_CYAN,         # 298: r_bottom_point_of_tragus
    COLOR_CYAN,         # 299: r_protruding_point_of_tragus
    COLOR_CYAN,         # 300: r_top_point_of_tragus
    COLOR_CYAN,         # 301: r_start_point_of_crus_of_helix
    COLOR_CYAN,         # 302: r_deepest_point_of_concha
    COLOR_CYAN,         # 303: r_tip_of_ear_lobe
    COLOR_CYAN,         # 304: r_midpoint_between_22_15
    COLOR_CYAN,         # 305: r_bottom_connecting_point_of_ear_lobe
    COLOR_CYAN,         # 306: r_top_connecting_point_of_helix
    COLOR_CYAN,         # 307: r_point_2_of_inner_helix
    COLOR_OLIVE_GREEN,  # 308: l_center_of_iris
    COLOR_OLIVE_GREEN,  # 309: l_border_of_iris_3
    COLOR_OLIVE_GREEN,  # 310: l_border_of_iris_midpoint_1
    COLOR_OLIVE_GREEN,  # 311: l_border_of_iris_12
    COLOR_OLIVE_GREEN,  # 312: l_border_of_iris_midpoint_4
    COLOR_OLIVE_GREEN,  # 313: l_border_of_iris_9
    COLOR_OLIVE_GREEN,  # 314: l_border_of_iris_midpoint_3
    COLOR_OLIVE_GREEN,  # 315: l_border_of_iris_6
    COLOR_OLIVE_GREEN,  # 316: l_border_of_iris_midpoint_2
    COLOR_BURGUNDY,     # 317: r_center_of_iris
    COLOR_BURGUNDY,     # 318: r_border_of_iris_3
    COLOR_BURGUNDY,     # 319: r_border_of_iris_midpoint_1
    COLOR_BURGUNDY,     # 320: r_border_of_iris_12
    COLOR_BURGUNDY,     # 321: r_border_of_iris_midpoint_4
    COLOR_BURGUNDY,     # 322: r_border_of_iris_9
    COLOR_BURGUNDY,     # 323: r_border_of_iris_midpoint_3
    COLOR_BURGUNDY,     # 324: r_border_of_iris_6
    COLOR_BURGUNDY,     # 325: r_border_of_iris_midpoint_2
    COLOR_BROWN,        # 326: l_center_of_pupil
    COLOR_BROWN,        # 327: l_border_of_pupil_3
    COLOR_BROWN,        # 328: l_border_of_pupil_midpoint_1
    COLOR_BROWN,        # 329: l_border_of_pupil_12
    COLOR_BROWN,        # 330: l_border_of_pupil_midpoint_4
    COLOR_BROWN,        # 331: l_border_of_pupil_9
    COLOR_BROWN,        # 332: l_border_of_pupil_midpoint_3
    COLOR_BROWN,        # 333: l_border_of_pupil_6
    COLOR_BROWN,        # 334: l_border_of_pupil_midpoint_2
    COLOR_LIGHT_CYAN,   # 335: r_center_of_pupil
    COLOR_LIGHT_CYAN,   # 336: r_border_of_pupil_3
    COLOR_LIGHT_CYAN,   # 337: r_border_of_pupil_midpoint_1
    COLOR_LIGHT_CYAN,   # 338: r_border_of_pupil_12
    COLOR_LIGHT_CYAN,   # 339: r_border_of_pupil_midpoint_4
    COLOR_LIGHT_CYAN,   # 340: r_border_of_pupil_9
    COLOR_LIGHT_CYAN,   # 341: r_border_of_pupil_midpoint_3
    COLOR_LIGHT_CYAN,   # 342: r_border_of_pupil_6
    COLOR_LIGHT_CYAN,   # 343: r_border_of_pupil_midpoint_2
]


GOLIATH_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
    "center_of_glabella",
    "center_of_nose_root",
    "tip_of_nose_bridge",
    "midpoint_1_of_nose_bridge",
    "midpoint_2_of_nose_bridge",
    "midpoint_3_of_nose_bridge",
    "center_of_labiomental_groove",
    "tip_of_chin",
    "upper_startpoint_of_r_eyebrow",
    "lower_startpoint_of_r_eyebrow",
    "end_of_r_eyebrow",
    "upper_midpoint_1_of_r_eyebrow",
    "lower_midpoint_1_of_r_eyebrow",
    "upper_midpoint_2_of_r_eyebrow",
    "upper_midpoint_3_of_r_eyebrow",
    "lower_midpoint_2_of_r_eyebrow",
    "lower_midpoint_3_of_r_eyebrow",
    "upper_startpoint_of_l_eyebrow",
    "lower_startpoint_of_l_eyebrow",
    "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow",
    "lower_midpoint_1_of_l_eyebrow",
    "upper_midpoint_2_of_l_eyebrow",
    "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow",
    "lower_midpoint_3_of_l_eyebrow",
    "l_inner_end_of_upper_lash_line",
    "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line",
    "l_midpoint_2_of_upper_lash_line",
    "l_midpoint_1_of_upper_lash_line",
    "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line",
    "l_midpoint_4_of_upper_lash_line",
    "l_midpoint_3_of_upper_lash_line",
    "l_outer_end_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_eyelid_line",
    "l_midpoint_2_of_upper_eyelid_line",
    "l_midpoint_5_of_upper_eyelid_line",
    "l_centerpoint_of_upper_eyelid_line",
    "l_midpoint_4_of_upper_eyelid_line",
    "l_midpoint_1_of_upper_eyelid_line",
    "l_midpoint_3_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_crease_line",
    "l_midpoint_2_of_upper_crease_line",
    "l_midpoint_5_of_upper_crease_line",
    "l_centerpoint_of_upper_crease_line",
    "l_midpoint_4_of_upper_crease_line",
    "l_midpoint_1_of_upper_crease_line",
    "l_midpoint_3_of_upper_crease_line",
    "r_inner_end_of_upper_lash_line",
    "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line",
    "r_midpoint_1_of_upper_lash_line",
    "r_midpoint_2_of_upper_lash_line",
    "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line",
    "r_midpoint_5_of_upper_lash_line",
    "r_midpoint_6_of_upper_lash_line",
    "r_outer_end_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_eyelid_line",
    "r_midpoint_1_of_upper_eyelid_line",
    "r_midpoint_4_of_upper_eyelid_line",
    "r_centerpoint_of_upper_eyelid_line",
    "r_midpoint_5_of_upper_eyelid_line",
    "r_midpoint_2_of_upper_eyelid_line",
    "r_midpoint_6_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_crease_line",
    "r_midpoint_1_of_upper_crease_line",
    "r_midpoint_4_of_upper_crease_line",
    "r_centerpoint_of_upper_crease_line",
    "r_midpoint_5_of_upper_crease_line",
    "r_midpoint_2_of_upper_crease_line",
    "r_midpoint_6_of_upper_crease_line",
    "l_inner_end_of_lower_lash_line",
    "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line",
    "l_midpoint_2_of_lower_lash_line",
    "l_midpoint_1_of_lower_lash_line",
    "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line",
    "l_midpoint_4_of_lower_lash_line",
    "l_midpoint_3_of_lower_lash_line",
    "l_outer_end_of_lower_eyelid_line",
    "l_midpoint_6_of_lower_eyelid_line",
    "l_midpoint_2_of_lower_eyelid_line",
    "l_midpoint_5_of_lower_eyelid_line",
    "l_centerpoint_of_lower_eyelid_line",
    "l_midpoint_4_of_lower_eyelid_line",
    "l_midpoint_1_of_lower_eyelid_line",
    "l_midpoint_3_of_lower_eyelid_line",
    "r_inner_end_of_lower_lash_line",
    "r_outer_end_of_lower_lash_line",
    "r_centerpoint_of_lower_lash_line",
    "r_midpoint_1_of_lower_lash_line",
    "r_midpoint_2_of_lower_lash_line",
    "r_midpoint_3_of_lower_lash_line",
    "r_midpoint_4_of_lower_lash_line",
    "r_midpoint_5_of_lower_lash_line",
    "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line",
    "r_midpoint_3_of_lower_eyelid_line",
    "r_midpoint_1_of_lower_eyelid_line",
    "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line",
    "r_midpoint_5_of_lower_eyelid_line",
    "r_midpoint_2_of_lower_eyelid_line",
    "r_midpoint_6_of_lower_eyelid_line",
    "tip_of_nose",
    "bottom_center_of_nose",
    "r_outer_corner_of_nose",
    "l_outer_corner_of_nose",
    "inner_corner_of_r_nostril",
    "outer_corner_of_r_nostril",
    "upper_corner_of_r_nostril",
    "inner_corner_of_l_nostril",
    "outer_corner_of_l_nostril",
    "upper_corner_of_l_nostril",
    "r_outer_corner_of_mouth",
    "l_outer_corner_of_mouth",
    "center_of_cupid_bow",
    "center_of_lower_outer_lip",
    "midpoint_1_of_upper_outer_lip",
    "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip",
    "midpoint_2_of_lower_outer_lip",
    "midpoint_3_of_upper_outer_lip",
    "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip",
    "midpoint_6_of_upper_outer_lip",
    "midpoint_3_of_lower_outer_lip",
    "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip",
    "midpoint_6_of_lower_outer_lip",
    "r_inner_corner_of_mouth",
    "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip",
    "center_of_lower_inner_lip",
    "midpoint_1_of_upper_inner_lip",
    "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip",
    "midpoint_2_of_lower_inner_lip",
    "midpoint_3_of_upper_inner_lip",
    "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip",
    "midpoint_6_of_upper_inner_lip",
    "midpoint_3_of_lower_inner_lip",
    "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip",
    "midpoint_6_of_lower_inner_lip",
    "l_top_end_of_inferior_crus",
    "l_top_end_of_superior_crus",
    "l_start_of_antihelix",
    "l_end_of_antihelix",
    "l_midpoint_1_of_antihelix",
    "l_midpoint_1_of_inferior_crus",
    "l_midpoint_2_of_antihelix",
    "l_midpoint_3_of_antihelix",
    "l_point_1_of_inner_helix",
    "l_point_2_of_inner_helix",
    "l_point_3_of_inner_helix",
    "l_point_4_of_inner_helix",
    "l_point_5_of_inner_helix",
    "l_point_6_of_inner_helix",
    "l_point_7_of_inner_helix",
    "l_highest_point_of_antitragus",
    "l_bottom_point_of_tragus",
    "l_protruding_point_of_tragus",
    "l_top_point_of_tragus",
    "l_start_point_of_crus_of_helix",
    "l_deepest_point_of_concha",
    "l_tip_of_ear_lobe",
    "l_midpoint_between_22_15",
    "l_bottom_connecting_point_of_ear_lobe",
    "l_top_connecting_point_of_helix",
    "l_point_8_of_inner_helix",
    "r_top_end_of_inferior_crus",
    "r_top_end_of_superior_crus",
    "r_start_of_antihelix",
    "r_end_of_antihelix",
    "r_midpoint_1_of_antihelix",
    "r_midpoint_1_of_inferior_crus",
    "r_midpoint_2_of_antihelix",
    "r_midpoint_3_of_antihelix",
    "r_point_1_of_inner_helix",
    "r_point_8_of_inner_helix",
    "r_point_3_of_inner_helix",
    "r_point_4_of_inner_helix",
    "r_point_5_of_inner_helix",
    "r_point_6_of_inner_helix",
    "r_point_7_of_inner_helix",
    "r_highest_point_of_antitragus",
    "r_bottom_point_of_tragus",
    "r_protruding_point_of_tragus",
    "r_top_point_of_tragus",
    "r_start_point_of_crus_of_helix",
    "r_deepest_point_of_concha",
    "r_tip_of_ear_lobe",
    "r_midpoint_between_22_15",
    "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix",
    "r_point_2_of_inner_helix",
    "l_center_of_iris",
    "l_border_of_iris_3",
    "l_border_of_iris_midpoint_1",
    "l_border_of_iris_12",
    "l_border_of_iris_midpoint_4",
    "l_border_of_iris_9",
    "l_border_of_iris_midpoint_3",
    "l_border_of_iris_6",
    "l_border_of_iris_midpoint_2",
    "r_center_of_iris",
    "r_border_of_iris_3",
    "r_border_of_iris_midpoint_1",
    "r_border_of_iris_12",
    "r_border_of_iris_midpoint_4",
    "r_border_of_iris_9",
    "r_border_of_iris_midpoint_3",
    "r_border_of_iris_6",
    "r_border_of_iris_midpoint_2",
    "l_center_of_pupil",
    "l_border_of_pupil_3",
    "l_border_of_pupil_midpoint_1",
    "l_border_of_pupil_12",
    "l_border_of_pupil_midpoint_4",
    "l_border_of_pupil_9",
    "l_border_of_pupil_midpoint_3",
    "l_border_of_pupil_6",
    "l_border_of_pupil_midpoint_2",
    "r_center_of_pupil",
    "r_border_of_pupil_3",
    "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12",
    "r_border_of_pupil_midpoint_4",
    "r_border_of_pupil_9",
    "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6",
    "r_border_of_pupil_midpoint_2",
]

# fmt: on
# precomputed for batch
GOLIATH_SKELETON_INFO_BY_ID = {
    0: dict(link=("left_ankle", "left_knee"), id=0, color=COLOR_GREEN),
    1: dict(link=("left_knee", "left_hip"), id=1, color=COLOR_GREEN),
    2: dict(link=("right_ankle", "right_knee"), id=2, color=COLOR_ORANGE),
    3: dict(link=("right_knee", "right_hip"), id=3, color=COLOR_ORANGE),
    4: dict(link=("left_hip", "right_hip"), id=4, color=COLOR_DODGE_BLUE),
    5: dict(link=("left_shoulder", "left_hip"), id=5, color=COLOR_DODGE_BLUE),
    6: dict(link=("right_shoulder", "right_hip"), id=6, color=COLOR_DODGE_BLUE),
    7: dict(link=("left_shoulder", "right_shoulder"), id=7, color=COLOR_DODGE_BLUE),
    8: dict(link=("left_shoulder", "left_elbow"), id=8, color=COLOR_GREEN),
    9: dict(link=("right_shoulder", "right_elbow"), id=9, color=COLOR_ORANGE),
    10: dict(link=("left_elbow", "left_wrist"), id=10, color=COLOR_GREEN),
    11: dict(link=("right_elbow", "right_wrist"), id=11, color=COLOR_ORANGE),
    12: dict(link=("left_eye", "right_eye"), id=12, color=COLOR_DODGE_BLUE),
    13: dict(link=("nose", "left_eye"), id=13, color=COLOR_DODGE_BLUE),
    14: dict(link=("nose", "right_eye"), id=14, color=COLOR_DODGE_BLUE),
    15: dict(link=("left_eye", "left_ear"), id=15, color=COLOR_DODGE_BLUE),
    16: dict(link=("right_eye", "right_ear"), id=16, color=COLOR_DODGE_BLUE),
    17: dict(link=("left_ear", "left_shoulder"), id=17, color=COLOR_DODGE_BLUE),
    18: dict(link=("right_ear", "right_shoulder"), id=18, color=COLOR_DODGE_BLUE),
    19: dict(link=("left_ankle", "left_big_toe"), id=19, color=COLOR_GREEN),
    20: dict(link=("left_ankle", "left_small_toe"), id=20, color=COLOR_GREEN),
    21: dict(link=("left_ankle", "left_heel"), id=21, color=COLOR_GREEN),
    22: dict(link=("right_ankle", "right_big_toe"), id=22, color=COLOR_ORANGE),
    23: dict(link=("right_ankle", "right_small_toe"), id=23, color=COLOR_ORANGE),
    24: dict(link=("right_ankle", "right_heel"), id=24, color=COLOR_ORANGE),
    25: dict(link=("left_wrist", "left_thumb_third_joint"), id=25, color=COLOR_ORANGE),
    26: dict(link=("left_thumb_third_joint", "left_thumb2"), id=26, color=COLOR_ORANGE),
    27: dict(link=("left_thumb2", "left_thumb3"), id=27, color=COLOR_ORANGE),
    28: dict(link=("left_thumb3", "left_thumb4"), id=28, color=COLOR_ORANGE),
    29: dict(
        link=("left_wrist", "left_forefinger_third_joint"), id=29, color=COLOR_PINK
    ),
    30: dict(
        link=("left_forefinger_third_joint", "left_forefinger2"),
        id=30,
        color=COLOR_PINK,
    ),
    31: dict(link=("left_forefinger2", "left_forefinger3"), id=31, color=COLOR_PINK),
    32: dict(link=("left_forefinger3", "left_forefinger4"), id=32, color=COLOR_PINK),
    33: dict(
        link=("left_wrist", "left_middle_finger_third_joint"),
        id=33,
        color=COLOR_SKY_BLUE,
    ),
    34: dict(
        link=("left_middle_finger_third_joint", "left_middle_finger2"),
        id=34,
        color=COLOR_SKY_BLUE,
    ),
    35: dict(
        link=("left_middle_finger2", "left_middle_finger3"),
        id=35,
        color=COLOR_SKY_BLUE,
    ),
    36: dict(
        link=("left_middle_finger3", "left_middle_finger4"),
        id=36,
        color=COLOR_SKY_BLUE,
    ),
    37: dict(
        link=("left_wrist", "left_ring_finger_third_joint"), id=37, color=COLOR_RED
    ),
    38: dict(
        link=("left_ring_finger_third_joint", "left_ring_finger2"),
        id=38,
        color=COLOR_RED,
    ),
    39: dict(link=("left_ring_finger2", "left_ring_finger3"), id=39, color=COLOR_RED),
    40: dict(link=("left_ring_finger3", "left_ring_finger4"), id=40, color=COLOR_RED),
    41: dict(
        link=("left_wrist", "left_pinky_finger_third_joint"), id=41, color=COLOR_GREEN
    ),
    42: dict(
        link=("left_pinky_finger_third_joint", "left_pinky_finger2"),
        id=42,
        color=COLOR_GREEN,
    ),
    43: dict(
        link=("left_pinky_finger2", "left_pinky_finger3"), id=43, color=COLOR_GREEN
    ),
    44: dict(
        link=("left_pinky_finger3", "left_pinky_finger4"), id=44, color=COLOR_GREEN
    ),
    45: dict(
        link=("right_wrist", "right_thumb_third_joint"), id=45, color=COLOR_ORANGE
    ),
    46: dict(
        link=("right_thumb_third_joint", "right_thumb2"), id=46, color=COLOR_ORANGE
    ),
    47: dict(link=("right_thumb2", "right_thumb3"), id=47, color=COLOR_ORANGE),
    48: dict(link=("right_thumb3", "right_thumb4"), id=48, color=COLOR_ORANGE),
    49: dict(
        link=("right_wrist", "right_forefinger_third_joint"),
        id=49,
        color=COLOR_PINK,
    ),
    50: dict(
        link=("right_forefinger_third_joint", "right_forefinger2"),
        id=50,
        color=COLOR_PINK,
    ),
    51: dict(link=("right_forefinger2", "right_forefinger3"), id=51, color=COLOR_PINK),
    52: dict(link=("right_forefinger3", "right_forefinger4"), id=52, color=COLOR_PINK),
    53: dict(
        link=("right_wrist", "right_middle_finger_third_joint"),
        id=53,
        color=COLOR_SKY_BLUE,
    ),
    54: dict(
        link=("right_middle_finger_third_joint", "right_middle_finger2"),
        id=54,
        color=COLOR_SKY_BLUE,
    ),
    55: dict(
        link=("right_middle_finger2", "right_middle_finger3"),
        id=55,
        color=COLOR_SKY_BLUE,
    ),
    56: dict(
        link=("right_middle_finger3", "right_middle_finger4"),
        id=56,
        color=COLOR_SKY_BLUE,
    ),
    57: dict(
        link=("right_wrist", "right_ring_finger_third_joint"),
        id=57,
        color=COLOR_RED,
    ),
    58: dict(
        link=("right_ring_finger_third_joint", "right_ring_finger2"),
        id=58,
        color=COLOR_RED,
    ),
    59: dict(link=("right_ring_finger2", "right_ring_finger3"), id=59, color=COLOR_RED),
    60: dict(link=("right_ring_finger3", "right_ring_finger4"), id=60, color=COLOR_RED),
    61: dict(
        link=("right_wrist", "right_pinky_finger_third_joint"), id=61, color=COLOR_GREEN
    ),
    62: dict(
        link=("right_pinky_finger_third_joint", "right_pinky_finger2"),
        id=62,
        color=COLOR_GREEN,
    ),
    63: dict(
        link=("right_pinky_finger2", "right_pinky_finger3"), id=63, color=COLOR_GREEN
    ),
    64: dict(
        link=("right_pinky_finger3", "right_pinky_finger4"), id=64, color=COLOR_GREEN
    ),
}
GOLIATH_SKELETON_INFO = {
    0: dict(link=(13, 11), id=0, color=COLOR_GREEN),
    1: dict(link=(11, 9), id=1, color=COLOR_GREEN),
    2: dict(link=(14, 12), id=2, color=COLOR_ORANGE),
    3: dict(link=(12, 10), id=3, color=COLOR_ORANGE),
    4: dict(link=(9, 10), id=4, color=COLOR_DODGE_BLUE),
    5: dict(link=(5, 9), id=5, color=COLOR_DODGE_BLUE),
    6: dict(link=(6, 10), id=6, color=COLOR_DODGE_BLUE),
    7: dict(link=(5, 6), id=7, color=COLOR_DODGE_BLUE),
    8: dict(link=(5, 7), id=8, color=COLOR_GREEN),
    9: dict(link=(6, 8), id=9, color=COLOR_ORANGE),
    10: dict(link=(7, 62), id=10, color=COLOR_GREEN),
    11: dict(link=(8, 41), id=11, color=COLOR_ORANGE),
    12: dict(link=(1, 2), id=12, color=COLOR_DODGE_BLUE),
    13: dict(link=(0, 1), id=13, color=COLOR_DODGE_BLUE),
    14: dict(link=(0, 2), id=14, color=COLOR_DODGE_BLUE),
    15: dict(link=(1, 3), id=15, color=COLOR_DODGE_BLUE),
    16: dict(link=(2, 4), id=16, color=COLOR_DODGE_BLUE),
    17: dict(link=(3, 5), id=17, color=COLOR_DODGE_BLUE),
    18: dict(link=(4, 6), id=18, color=COLOR_DODGE_BLUE),
    19: dict(link=(13, 15), id=19, color=COLOR_GREEN),
    20: dict(link=(13, 16), id=20, color=COLOR_GREEN),
    21: dict(link=(13, 17), id=21, color=COLOR_GREEN),
    22: dict(link=(14, 18), id=22, color=COLOR_ORANGE),
    23: dict(link=(14, 19), id=23, color=COLOR_ORANGE),
    24: dict(link=(14, 20), id=24, color=COLOR_ORANGE),
    25: dict(link=(62, 45), id=25, color=COLOR_ORANGE),
    26: dict(link=(45, 44), id=26, color=COLOR_ORANGE),
    27: dict(link=(44, 43), id=27, color=COLOR_ORANGE),
    28: dict(link=(43, 42), id=28, color=COLOR_ORANGE),
    29: dict(link=(62, 49), id=29, color=COLOR_PINK),
    30: dict(link=(49, 48), id=30, color=COLOR_PINK),
    31: dict(link=(48, 47), id=31, color=COLOR_PINK),
    32: dict(link=(47, 46), id=32, color=COLOR_PINK),
    33: dict(link=(62, 53), id=33, color=COLOR_SKY_BLUE),
    34: dict(link=(53, 52), id=34, color=COLOR_SKY_BLUE),
    35: dict(link=(52, 51), id=35, color=COLOR_SKY_BLUE),
    36: dict(link=(51, 50), id=36, color=COLOR_SKY_BLUE),
    37: dict(link=(62, 57), id=37, color=COLOR_RED),
    38: dict(link=(57, 56), id=38, color=COLOR_RED),
    39: dict(link=(56, 55), id=39, color=COLOR_RED),
    40: dict(link=(55, 54), id=40, color=COLOR_RED),
    41: dict(link=(62, 61), id=41, color=COLOR_GREEN),
    42: dict(link=(61, 60), id=42, color=COLOR_GREEN),
    43: dict(link=(60, 59), id=43, color=COLOR_GREEN),
    44: dict(link=(59, 58), id=44, color=COLOR_GREEN),
    45: dict(link=(41, 24), id=45, color=COLOR_ORANGE),
    46: dict(link=(24, 23), id=46, color=COLOR_ORANGE),
    47: dict(link=(23, 22), id=47, color=COLOR_ORANGE),
    48: dict(link=(22, 21), id=48, color=COLOR_ORANGE),
    49: dict(link=(41, 28), id=49, color=COLOR_PINK),
    50: dict(link=(28, 27), id=50, color=COLOR_PINK),
    51: dict(link=(27, 26), id=51, color=COLOR_PINK),
    52: dict(link=(26, 25), id=52, color=COLOR_PINK),
    53: dict(link=(41, 32), id=53, color=COLOR_SKY_BLUE),
    54: dict(link=(32, 31), id=54, color=COLOR_SKY_BLUE),
    55: dict(link=(31, 30), id=55, color=COLOR_SKY_BLUE),
    56: dict(link=(30, 29), id=56, color=COLOR_SKY_BLUE),
    57: dict(link=(41, 36), id=57, color=COLOR_RED),
    58: dict(link=(36, 35), id=58, color=COLOR_RED),
    59: dict(link=(35, 34), id=59, color=COLOR_RED),
    60: dict(link=(34, 33), id=60, color=COLOR_RED),
    61: dict(link=(41, 40), id=61, color=COLOR_GREEN),
    62: dict(link=(40, 39), id=62, color=COLOR_GREEN),
    63: dict(link=(39, 38), id=63, color=COLOR_GREEN),
    64: dict(link=(38, 37), id=64, color=COLOR_GREEN),
}

# region kornia
GOLIATH_KPTS_COLORS_TENSOR = (
    torch.tensor(GOLIATH_KPTS_COLORS, dtype=torch.float32) / 255.0
)  # (num_kpts, 3)

GOLIATH_SKELETON_COLORS_TENSOR = (
    torch.tensor(
        [info["color"] for info in GOLIATH_SKELETON_INFO.values()], dtype=torch.float32
    )
    / 255.0
)  # (num_links, 3)
# end region


KEYPOINT_NAME_TO_ID = {name: i for i, name in enumerate(GOLIATH_KEYPOINTS)}


SKELETON_LINKS_BY_INDEX = []
for link_info in GOLIATH_SKELETON_INFO_BY_ID.values():
    pt1_name, pt2_name = link_info["link"]

    # Get the indices from our mapping
    pt1_idx = KEYPOINT_NAME_TO_ID.get(pt1_name)
    pt2_idx = KEYPOINT_NAME_TO_ID.get(pt2_name)

    # Add to our purely numerical lookup list
    if pt1_idx is not None and pt2_idx is not None:
        SKELETON_LINKS_BY_INDEX.append(
            {"link_indices": (pt1_idx, pt2_idx), "color": link_info["color"]}
        )


GOLIATH_MEMBER_GROUPS = {
    "face": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "center_of_glabella",
        "center_of_nose_root",
        "tip_of_nose_bridge",
        "midpoint_1_of_nose_bridge",
        "midpoint_2_of_nose_bridge",
        "midpoint_3_of_nose_bridge",
        "center_of_labiomental_groove",
        "tip_of_chin",
        "upper_startpoint_of_r_eyebrow",
        "lower_startpoint_of_r_eyebrow",
        "end_of_r_eyebrow",
        "upper_midpoint_1_of_r_eyebrow",
        "lower_midpoint_1_of_r_eyebrow",
        "upper_midpoint_2_of_r_eyebrow",
        "upper_midpoint_3_of_r_eyebrow",
        "lower_midpoint_2_of_r_eyebrow",
        "lower_midpoint_3_of_r_eyebrow",
        "upper_startpoint_of_l_eyebrow",
        "lower_startpoint_of_l_eyebrow",
        "end_of_l_eyebrow",
        "upper_midpoint_1_of_l_eyebrow",
        "lower_midpoint_1_of_l_eyebrow",
        "upper_midpoint_2_of_l_eyebrow",
        "upper_midpoint_3_of_l_eyebrow",
        "lower_midpoint_2_of_l_eyebrow",
        "lower_midpoint_3_of_l_eyebrow",
        "l_inner_end_of_upper_lash_line",
        "l_outer_end_of_upper_lash_line",
        "l_centerpoint_of_upper_lash_line",
        "l_midpoint_2_of_upper_lash_line",
        "l_midpoint_1_of_upper_lash_line",
        "l_midpoint_6_of_upper_lash_line",
        "l_midpoint_5_of_upper_lash_line",
        "l_midpoint_4_of_upper_lash_line",
        "l_midpoint_3_of_upper_lash_line",
        "l_outer_end_of_upper_eyelid_line",
        "l_midpoint_6_of_upper_eyelid_line",
        "l_midpoint_2_of_upper_eyelid_line",
        "l_midpoint_5_of_upper_eyelid_line",
        "l_centerpoint_of_upper_eyelid_line",
        "l_midpoint_4_of_upper_eyelid_line",
        "l_midpoint_1_of_upper_eyelid_line",
        "l_midpoint_3_of_upper_eyelid_line",
        "l_midpoint_6_of_upper_crease_line",
        "l_midpoint_2_of_upper_crease_line",
        "l_midpoint_5_of_upper_crease_line",
        "l_centerpoint_of_upper_crease_line",
        "l_midpoint_4_of_upper_crease_line",
        "l_midpoint_1_of_upper_crease_line",
        "l_midpoint_3_of_upper_crease_line",
        "r_inner_end_of_upper_lash_line",
        "r_outer_end_of_upper_lash_line",
        "r_centerpoint_of_upper_lash_line",
        "r_midpoint_1_of_upper_lash_line",
        "r_midpoint_2_of_upper_lash_line",
        "r_midpoint_3_of_upper_lash_line",
        "r_midpoint_4_of_upper_lash_line",
        "r_midpoint_5_of_upper_lash_line",
        "r_midpoint_6_of_upper_lash_line",
        "r_outer_end_of_upper_eyelid_line",
        "r_midpoint_3_of_upper_eyelid_line",
        "r_midpoint_1_of_upper_eyelid_line",
        "r_midpoint_4_of_upper_eyelid_line",
        "r_centerpoint_of_upper_eyelid_line",
        "r_midpoint_5_of_upper_eyelid_line",
        "r_midpoint_2_of_upper_eyelid_line",
        "r_midpoint_6_of_upper_eyelid_line",
        "r_midpoint_3_of_upper_crease_line",
        "r_midpoint_1_of_upper_crease_line",
        "r_midpoint_4_of_upper_crease_line",
        "r_centerpoint_of_upper_crease_line",
        "r_midpoint_5_of_upper_crease_line",
        "r_midpoint_2_of_upper_crease_line",
        "r_midpoint_6_of_upper_crease_line",
        "l_inner_end_of_lower_lash_line",
        "l_outer_end_of_lower_lash_line",
        "l_centerpoint_of_lower_lash_line",
        "l_midpoint_2_of_lower_lash_line",
        "l_midpoint_1_of_lower_lash_line",
        "l_midpoint_6_of_lower_lash_line",
        "l_midpoint_5_of_lower_lash_line",
        "l_midpoint_4_of_lower_lash_line",
        "l_midpoint_3_of_lower_lash_line",
        "l_outer_end_of_lower_eyelid_line",
        "l_midpoint_6_of_lower_eyelid_line",
        "l_midpoint_2_of_lower_eyelid_line",
        "l_midpoint_5_of_lower_eyelid_line",
        "l_centerpoint_of_lower_eyelid_line",
        "l_midpoint_4_of_lower_eyelid_line",
        "l_midpoint_1_of_lower_eyelid_line",
        "l_midpoint_3_of_lower_eyelid_line",
        "r_inner_end_of_lower_lash_line",
        "r_outer_end_of_lower_lash_line",
        "r_centerpoint_of_lower_lash_line",
        "r_midpoint_1_of_lower_lash_line",
        "r_midpoint_2_of_lower_lash_line",
        "r_midpoint_3_of_lower_lash_line",
        "r_midpoint_4_of_lower_lash_line",
        "r_midpoint_5_of_lower_lash_line",
        "r_midpoint_6_of_lower_lash_line",
        "r_outer_end_of_lower_eyelid_line",
        "r_midpoint_3_of_lower_eyelid_line",
        "r_midpoint_1_of_lower_eyelid_line",
        "r_midpoint_4_of_lower_eyelid_line",
        "r_centerpoint_of_lower_eyelid_line",
        "r_midpoint_5_of_lower_eyelid_line",
        "r_midpoint_2_of_lower_eyelid_line",
        "r_midpoint_6_of_lower_eyelid_line",
        "tip_of_nose",
        "bottom_center_of_nose",
        "r_outer_corner_of_nose",
        "l_outer_corner_of_nose",
        "inner_corner_of_r_nostril",
        "outer_corner_of_r_nostril",
        "upper_corner_of_r_nostril",
        "inner_corner_of_l_nostril",
        "outer_corner_of_l_nostril",
        "upper_corner_of_l_nostril",
        "r_outer_corner_of_mouth",
        "l_outer_corner_of_mouth",
        "center_of_cupid_bow",
        "center_of_lower_outer_lip",
        "midpoint_1_of_upper_outer_lip",
        "midpoint_2_of_upper_outer_lip",
        "midpoint_1_of_lower_outer_lip",
        "midpoint_2_of_lower_outer_lip",
        "midpoint_3_of_upper_outer_lip",
        "midpoint_4_of_upper_outer_lip",
        "midpoint_5_of_upper_outer_lip",
        "midpoint_6_of_upper_outer_lip",
        "midpoint_3_of_lower_outer_lip",
        "midpoint_4_of_lower_outer_lip",
        "midpoint_5_of_lower_outer_lip",
        "midpoint_6_of_lower_outer_lip",
        "r_inner_corner_of_mouth",
        "l_inner_corner_of_mouth",
        "center_of_upper_inner_lip",
        "center_of_lower_inner_lip",
        "midpoint_1_of_upper_inner_lip",
        "midpoint_2_of_upper_inner_lip",
        "midpoint_1_of_lower_inner_lip",
        "midpoint_2_of_lower_inner_lip",
        "midpoint_3_of_upper_inner_lip",
        "midpoint_4_of_upper_inner_lip",
        "midpoint_5_of_upper_inner_lip",
        "midpoint_6_of_upper_inner_lip",
        "midpoint_3_of_lower_inner_lip",
        "midpoint_4_of_lower_inner_lip",
        "midpoint_5_of_lower_inner_lip",
        "midpoint_6_of_lower_inner_lip",
        "l_top_end_of_inferior_crus",
        "l_top_end_of_superior_crus",
        "l_start_of_antihelix",
        "l_end_of_antihelix",
        "l_midpoint_1_of_antihelix",
        "l_midpoint_1_of_inferior_crus",
        "l_midpoint_2_of_antihelix",
        "l_midpoint_3_of_antihelix",
        "l_point_1_of_inner_helix",
        "l_point_2_of_inner_helix",
        "l_point_3_of_inner_helix",
        "l_point_4_of_inner_helix",
        "l_point_5_of_inner_helix",
        "l_point_6_of_inner_helix",
        "l_point_7_of_inner_helix",
        "l_highest_point_of_antitragus",
        "l_bottom_point_of_tragus",
        "l_protruding_point_of_tragus",
        "l_top_point_of_tragus",
        "l_start_point_of_crus_of_helix",
        "l_deepest_point_of_concha",
        "l_tip_of_ear_lobe",
        "l_midpoint_between_22_15",
        "l_bottom_connecting_point_of_ear_lobe",
        "l_top_connecting_point_of_helix",
        "l_point_8_of_inner_helix",
        "r_top_end_of_inferior_crus",
        "r_top_end_of_superior_crus",
        "r_start_of_antihelix",
        "r_end_of_antihelix",
        "r_midpoint_1_of_antihelix",
        "r_midpoint_1_of_inferior_crus",
        "r_midpoint_2_of_antihelix",
        "r_midpoint_3_of_antihelix",
        "r_point_1_of_inner_helix",
        "r_point_8_of_inner_helix",
        "r_point_3_of_inner_helix",
        "r_point_4_of_inner_helix",
        "r_point_5_of_inner_helix",
        "r_point_6_of_inner_helix",
        "r_point_7_of_inner_helix",
        "r_highest_point_of_antitragus",
        "r_bottom_point_of_tragus",
        "r_protruding_point_of_tragus",
        "r_top_point_of_tragus",
        "r_start_point_of_crus_of_helix",
        "r_deepest_point_of_concha",
        "r_tip_of_ear_lobe",
        "r_midpoint_between_22_15",
        "r_bottom_connecting_point_of_ear_lobe",
        "r_top_connecting_point_of_helix",
        "r_point_2_of_inner_helix",
        "l_center_of_iris",
        "l_border_of_iris_3",
        "l_border_of_iris_midpoint_1",
        "l_border_of_iris_12",
        "l_border_of_iris_midpoint_4",
        "l_border_of_iris_9",
        "l_border_of_iris_midpoint_3",
        "l_border_of_iris_6",
        "l_border_of_iris_midpoint_2",
        "r_center_of_iris",
        "r_border_of_iris_3",
        "r_border_of_iris_midpoint_1",
        "r_border_of_iris_12",
        "r_border_of_iris_midpoint_4",
        "r_border_of_iris_9",
        "r_border_of_iris_midpoint_3",
        "r_border_of_iris_6",
        "r_border_of_iris_midpoint_2",
        "l_center_of_pupil",
        "l_border_of_pupil_3",
        "l_border_of_pupil_midpoint_1",
        "l_border_of_pupil_12",
        "l_border_of_pupil_midpoint_4",
        "l_border_of_pupil_9",
        "l_border_of_pupil_midpoint_3",
        "l_border_of_pupil_6",
        "l_border_of_pupil_midpoint_2",
        "r_center_of_pupil",
        "r_border_of_pupil_3",
        "r_border_of_pupil_midpoint_1",
        "r_border_of_pupil_12",
        "r_border_of_pupil_midpoint_4",
        "r_border_of_pupil_9",
        "r_border_of_pupil_midpoint_3",
        "r_border_of_pupil_6",
        "r_border_of_pupil_midpoint_2",
    ],
    "clavicle": [
        "left_acromion",
        "right_acromion",
    ],
    "upper_body": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "right_thumb4",
        "right_thumb3",
        "right_thumb2",
        "right_thumb_third_joint",
        "right_forefinger4",
        "right_forefinger3",
        "right_forefinger2",
        "right_forefinger_third_joint",
        "right_middle_finger4",
        "right_middle_finger3",
        "right_middle_finger2",
        "right_middle_finger_third_joint",
        "right_ring_finger4",
        "right_ring_finger3",
        "right_ring_finger2",
        "right_ring_finger_third_joint",
        "right_pinky_finger4",
        "right_pinky_finger3",
        "right_pinky_finger2",
        "right_pinky_finger_third_joint",
        "right_wrist",
        "left_thumb4",
        "left_thumb3",
        "left_thumb2",
        "left_thumb_third_joint",
        "left_forefinger4",
        "left_forefinger3",
        "left_forefinger2",
        "left_forefinger_third_joint",
        "left_middle_finger4",
        "left_middle_finger3",
        "left_middle_finger2",
        "left_middle_finger_third_joint",
        "left_ring_finger4",
        "left_ring_finger3",
        "left_ring_finger2",
        "left_ring_finger_third_joint",
        "left_pinky_finger4",
        "left_pinky_finger3",
        "left_pinky_finger2",
        "left_pinky_finger_third_joint",
        "left_wrist",
        "left_olecranon",
        "right_olecranon",
        "left_cubital_fossa",
        "right_cubital_fossa",
        "neck",
    ],
    "lower_body": [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    ],
}

GOLIATH_MEMBER_GROUPS_BY_INDEX = {}
for group_name, keypoint_names in GOLIATH_MEMBER_GROUPS.items():
    indices = [KEYPOINT_NAME_TO_ID[name] for name in keypoint_names]
    GOLIATH_MEMBER_GROUPS_BY_INDEX[group_name] = indices

# fmt: off

# region openpose
# OPENPOSE_KPTS_COLOR = gen
# fmt:on
# endregion

COLOR_MAPS = {
    "sapiens_default": GOLIATH_KPTS_COLORS,
    # "openpose_style": OPENPOSE_STYLE_KPTS_COLORS,
}


Color = tuple[int, int, int]


class ColorPalette(BaseModel):
    DODGE_BLUE: Color = (51, 153, 255)
    GREEN: Color = (0, 255, 0)
    ORANGE: Color = (255, 128, 0)
    PINK: Color = (255, 153, 255)
    SKY_BLUE: Color = (102, 178, 255)
    RED: Color = (255, 51, 51)
    MULBERRY: Color = (192, 64, 128)
    INDIGO: Color = (64, 32, 192)
    TEAL: Color = (64, 192, 128)
    LIME_GREEN: Color = (64, 192, 32)
    DARK_GREEN: Color = (0, 192, 0)
    MAROON: Color = (192, 0, 0)
    TURQUOISE: Color = (0, 192, 192)
    YELLOW: Color = (200, 200, 0)
    CYAN: Color = (0, 200, 200)
    OLIVE_GREEN: Color = (128, 192, 64)
    BURGUNDY: Color = (192, 32, 64)
    BROWN: Color = (192, 128, 64)
    LIGHT_CYAN: Color = (32, 192, 192)
    WHITE: Color = (255, 255, 255)


def goliath_from_palette(palette: ColorPalette):
    KPTS = [
        palette.DODGE_BLUE,  # 0: nose
        palette.DODGE_BLUE,  # 1: left_eye
        palette.DODGE_BLUE,  # 2: right_eye
        palette.DODGE_BLUE,  # 3: left_ear
        palette.DODGE_BLUE,  # 4: right_ear
        palette.DODGE_BLUE,  # 5: left_shoulder
        palette.DODGE_BLUE,  # 6: right_shoulder
        palette.DODGE_BLUE,  # 7: left_elbow
        palette.DODGE_BLUE,  # 8: right_elbow
        palette.DODGE_BLUE,  # 9: left_hip
        palette.DODGE_BLUE,  # 10: right_hip
        palette.DODGE_BLUE,  # 11: left_knee
        palette.DODGE_BLUE,  # 12: right_knee
        palette.DODGE_BLUE,  # 13: left_ankle
        palette.DODGE_BLUE,  # 14: right_ankle
        palette.DODGE_BLUE,  # 15: left_big_toe
        palette.DODGE_BLUE,  # 16: left_small_toe
        palette.DODGE_BLUE,  # 17: left_heel
        palette.DODGE_BLUE,  # 18: right_big_toe
        palette.DODGE_BLUE,  # 19: right_small_toe
        palette.DODGE_BLUE,  # 20: right_heel
        palette.DODGE_BLUE,  # 21: right_thumb4
        palette.DODGE_BLUE,  # 22: right_thumb3
        palette.DODGE_BLUE,  # 23: right_thumb2
        palette.DODGE_BLUE,  # 24: right_thumb_third_joint
        palette.DODGE_BLUE,  # 25: right_forefinger4
        palette.DODGE_BLUE,  # 26: right_forefinger3
        palette.DODGE_BLUE,  # 27: right_forefinger2
        palette.DODGE_BLUE,  # 28: right_forefinger_third_joint
        palette.DODGE_BLUE,  # 29: right_middle_finger4
        palette.DODGE_BLUE,  # 30: right_middle_finger3
        palette.DODGE_BLUE,  # 31: right_middle_finger2
        palette.DODGE_BLUE,  # 32: right_middle_finger_third_joint
        palette.DODGE_BLUE,  # 33: right_ring_finger4
        palette.DODGE_BLUE,  # 34: right_ring_finger3
        palette.DODGE_BLUE,  # 35: right_ring_finger2
        palette.DODGE_BLUE,  # 36: right_ring_finger_third_joint
        palette.DODGE_BLUE,  # 37: right_pinky_finger4
        palette.DODGE_BLUE,  # 38: right_pinky_finger3
        palette.DODGE_BLUE,  # 39: right_pinky_finger2
        palette.DODGE_BLUE,  # 40: right_pinky_finger_third_joint
        palette.DODGE_BLUE,  # 41: right_wrist
        palette.DODGE_BLUE,  # 42: left_thumb4
        palette.DODGE_BLUE,  # 43: left_thumb3
        palette.DODGE_BLUE,  # 44: left_thumb2
        palette.DODGE_BLUE,  # 45: left_thumb_third_joint
        palette.DODGE_BLUE,  # 46: left_forefinger4
        palette.DODGE_BLUE,  # 47: left_forefinger3
        palette.DODGE_BLUE,  # 48: left_forefinger2
        palette.DODGE_BLUE,  # 49: left_forefinger_third_joint
        palette.DODGE_BLUE,  # 50: left_middle_finger4
        palette.DODGE_BLUE,  # 51: left_middle_finger3
        palette.DODGE_BLUE,  # 52: left_middle_finger2
        palette.DODGE_BLUE,  # 53: left_middle_finger_third_joint
        palette.DODGE_BLUE,  # 54: left_ring_finger4
        palette.DODGE_BLUE,  # 55: left_ring_finger3
        palette.DODGE_BLUE,  # 56: left_ring_finger2
        palette.DODGE_BLUE,  # 57: left_ring_finger_third_joint
        palette.DODGE_BLUE,  # 58: left_pinky_finger4
        palette.DODGE_BLUE,  # 59: left_pinky_finger3
        palette.DODGE_BLUE,  # 60: left_pinky_finger2
        palette.DODGE_BLUE,  # 61: left_pinky_finger_third_joint
        palette.DODGE_BLUE,  # 62: left_wrist
        palette.DODGE_BLUE,  # 63: left_olecranon
        palette.DODGE_BLUE,  # 64: right_olecranon
        palette.DODGE_BLUE,  # 65: left_cubital_fossa
        palette.DODGE_BLUE,  # 66: right_cubital_fossa
        palette.DODGE_BLUE,  # 67: left_acromion
        palette.DODGE_BLUE,  # 68: right_acromion
        palette.DODGE_BLUE,  # 69: neck
        palette.WHITE,  # 70: center_of_glabella
        palette.WHITE,  # 71: center_of_nose_root
        palette.WHITE,  # 72: tip_of_nose_bridge
        palette.WHITE,  # 73: midpoint_1_of_nose_bridge
        palette.WHITE,  # 74: midpoint_2_of_nose_bridge
        palette.WHITE,  # 75: midpoint_3_of_nose_bridge
        palette.WHITE,  # 76: center_of_labiomental_groove
        palette.WHITE,  # 77: tip_of_chin
        palette.WHITE,  # 78: upper_startpoint_of_r_eyebrow
        palette.WHITE,  # 79: lower_startpoint_of_r_eyebrow
        palette.WHITE,  # 80: end_of_r_eyebrow
        palette.WHITE,  # 81: upper_midpoint_1_of_r_eyebrow
        palette.WHITE,  # 82: lower_midpoint_1_of_r_eyebrow
        palette.WHITE,  # 83: upper_midpoint_2_of_r_eyebrow
        palette.WHITE,  # 84: upper_midpoint_3_of_r_eyebrow
        palette.WHITE,  # 85: lower_midpoint_2_of_r_eyebrow
        palette.WHITE,  # 86: lower_midpoint_3_of_r_eyebrow
        palette.WHITE,  # 87: upper_startpoint_of_l_eyebrow
        palette.WHITE,  # 88: lower_startpoint_of_l_eyebrow
        palette.WHITE,  # 89: end_of_l_eyebrow
        palette.WHITE,  # 90: upper_midpoint_1_of_l_eyebrow
        palette.WHITE,  # 91: lower_midpoint_1_of_l_eyebrow
        palette.WHITE,  # 92: upper_midpoint_2_of_l_eyebrow
        palette.WHITE,  # 93: upper_midpoint_3_of_l_eyebrow
        palette.WHITE,  # 94: lower_midpoint_2_of_l_eyebrow
        palette.WHITE,  # 95: lower_midpoint_3_of_l_eyebrow
        palette.MULBERRY,  # 96: l_inner_end_of_upper_lash_line
        palette.MULBERRY,  # 97: l_outer_end_of_upper_lash_line
        palette.MULBERRY,  # 98: l_centerpoint_of_upper_lash_line
        palette.MULBERRY,  # 99: l_midpoint_2_of_upper_lash_line
        palette.MULBERRY,  # 100: l_midpoint_1_of_upper_lash_line
        palette.MULBERRY,  # 101: l_midpoint_6_of_upper_lash_line
        palette.MULBERRY,  # 102: l_midpoint_5_of_upper_lash_line
        palette.MULBERRY,  # 103: l_midpoint_4_of_upper_lash_line
        palette.MULBERRY,  # 104: l_midpoint_3_of_upper_lash_line
        palette.MULBERRY,  # 105: l_outer_end_of_upper_eyelid_line
        palette.MULBERRY,  # 106: l_midpoint_6_of_upper_eyelid_line
        palette.MULBERRY,  # 107: l_midpoint_2_of_upper_eyelid_line
        palette.MULBERRY,  # 108: l_midpoint_5_of_upper_eyelid_line
        palette.MULBERRY,  # 109: l_centerpoint_of_upper_eyelid_line
        palette.MULBERRY,  # 110: l_midpoint_4_of_upper_eyelid_line
        palette.MULBERRY,  # 111: l_midpoint_1_of_upper_eyelid_line
        palette.MULBERRY,  # 112: l_midpoint_3_of_upper_eyelid_line
        palette.MULBERRY,  # 113: l_midpoint_6_of_upper_crease_line
        palette.MULBERRY,  # 114: l_midpoint_2_of_upper_crease_line
        palette.MULBERRY,  # 115: l_midpoint_5_of_upper_crease_line
        palette.MULBERRY,  # 116: l_centerpoint_of_upper_crease_line
        palette.MULBERRY,  # 117: l_midpoint_4_of_upper_crease_line
        palette.MULBERRY,  # 118: l_midpoint_1_of_upper_crease_line
        palette.MULBERRY,  # 119: l_midpoint_3_of_upper_crease_line
        palette.INDIGO,  # 120: r_inner_end_of_upper_lash_line
        palette.INDIGO,  # 121: r_outer_end_of_upper_lash_line
        palette.INDIGO,  # 122: r_centerpoint_of_upper_lash_line
        palette.INDIGO,  # 123: r_midpoint_1_of_upper_lash_line
        palette.INDIGO,  # 124: r_midpoint_2_of_upper_lash_line
        palette.INDIGO,  # 125: r_midpoint_3_of_upper_lash_line
        palette.INDIGO,  # 126: r_midpoint_4_of_upper_lash_line
        palette.INDIGO,  # 127: r_midpoint_5_of_upper_lash_line
        palette.INDIGO,  # 128: r_midpoint_6_of_upper_lash_line
        palette.INDIGO,  # 129: r_outer_end_of_upper_eyelid_line
        palette.INDIGO,  # 130: r_midpoint_3_of_upper_eyelid_line
        palette.INDIGO,  # 131: r_midpoint_1_of_upper_eyelid_line
        palette.INDIGO,  # 132: r_midpoint_4_of_upper_eyelid_line
        palette.INDIGO,  # 133: r_centerpoint_of_upper_eyelid_line
        palette.INDIGO,  # 134: r_midpoint_5_of_upper_eyelid_line
        palette.INDIGO,  # 135: r_midpoint_2_of_upper_eyelid_line
        palette.INDIGO,  # 136: r_midpoint_6_of_upper_eyelid_line
        palette.INDIGO,  # 137: r_midpoint_3_of_upper_crease_line
        palette.INDIGO,  # 138: r_midpoint_1_of_upper_crease_line
        palette.INDIGO,  # 139: r_midpoint_4_of_upper_crease_line
        palette.INDIGO,  # 140: r_centerpoint_of_upper_crease_line
        palette.INDIGO,  # 141: r_midpoint_5_of_upper_crease_line
        palette.INDIGO,  # 142: r_midpoint_2_of_upper_crease_line
        palette.INDIGO,  # 143: r_midpoint_6_of_upper_crease_line
        palette.TEAL,  # 144: l_inner_end_of_lower_lash_line
        palette.TEAL,  # 145: l_outer_end_of_lower_lash_line
        palette.TEAL,  # 146: l_centerpoint_of_lower_lash_line
        palette.TEAL,  # 147: l_midpoint_2_of_lower_lash_line
        palette.TEAL,  # 148: l_midpoint_1_of_lower_lash_line
        palette.TEAL,  # 149: l_midpoint_6_of_lower_lash_line
        palette.TEAL,  # 150: l_midpoint_5_of_lower_lash_line
        palette.TEAL,  # 151: l_midpoint_4_of_lower_lash_line
        palette.TEAL,  # 152: l_midpoint_3_of_lower_lash_line
        palette.TEAL,  # 153: l_outer_end_of_lower_eyelid_line
        palette.TEAL,  # 154: l_midpoint_6_of_lower_eyelid_line
        palette.TEAL,  # 155: l_midpoint_2_of_lower_eyelid_line
        palette.TEAL,  # 156: l_midpoint_5_of_lower_eyelid_line
        palette.TEAL,  # 157: l_centerpoint_of_lower_eyelid_line
        palette.TEAL,  # 158: l_midpoint_4_of_lower_eyelid_line
        palette.TEAL,  # 159: l_midpoint_1_of_lower_eyelid_line
        palette.TEAL,  # 160: l_midpoint_3_of_lower_eyelid_line
        palette.LIME_GREEN,  # 161: r_inner_end_of_lower_lash_line
        palette.LIME_GREEN,  # 162: r_outer_end_of_lower_lash_line
        palette.LIME_GREEN,  # 163: r_centerpoint_of_lower_lash_line
        palette.LIME_GREEN,  # 164: r_midpoint_1_of_lower_lash_line
        palette.LIME_GREEN,  # 165: r_midpoint_2_of_lower_lash_line
        palette.LIME_GREEN,  # 166: r_midpoint_3_of_lower_lash_line
        palette.LIME_GREEN,  # 167: r_midpoint_4_of_lower_lash_line
        palette.LIME_GREEN,  # 168: r_midpoint_5_of_lower_lash_line
        palette.LIME_GREEN,  # 169: r_midpoint_6_of_lower_lash_line
        palette.LIME_GREEN,  # 170: r_outer_end_of_lower_eyelid_line
        palette.LIME_GREEN,  # 171: r_midpoint_3_of_lower_eyelid_line
        palette.LIME_GREEN,  # 172: r_midpoint_1_of_lower_eyelid_line
        palette.LIME_GREEN,  # 173: r_midpoint_4_of_lower_eyelid_line
        palette.LIME_GREEN,  # 174: r_centerpoint_of_lower_eyelid_line
        palette.LIME_GREEN,  # 175: r_midpoint_5_of_lower_eyelid_line
        palette.LIME_GREEN,  # 176: r_midpoint_2_of_lower_eyelid_line
        palette.LIME_GREEN,  # 177: r_midpoint_6_of_lower_eyelid_line
        palette.DARK_GREEN,  # 178: tip_of_nose
        palette.DARK_GREEN,  # 179: bottom_center_of_nose
        palette.DARK_GREEN,  # 180: r_outer_corner_of_nose
        palette.DARK_GREEN,  # 181: l_outer_corner_of_nose
        palette.DARK_GREEN,  # 182: inner_corner_of_r_nostril
        palette.DARK_GREEN,  # 183: outer_corner_of_r_nostril
        palette.DARK_GREEN,  # 184: upper_corner_of_r_nostril
        palette.DARK_GREEN,  # 185: inner_corner_of_l_nostril
        palette.DARK_GREEN,  # 186: outer_corner_of_l_nostril
        palette.DARK_GREEN,  # 187: upper_corner_of_l_nostril
        palette.MAROON,  # 188: r_outer_corner_of_mouth
        palette.MAROON,  # 189: l_outer_corner_of_mouth
        palette.MAROON,  # 190: center_of_cupid_bow
        palette.MAROON,  # 191: center_of_lower_outer_lip
        palette.MAROON,  # 192: midpoint_1_of_upper_outer_lip
        palette.MAROON,  # 193: midpoint_2_of_upper_outer_lip
        palette.MAROON,  # 194: midpoint_1_of_lower_outer_lip
        palette.MAROON,  # 195: midpoint_2_of_lower_outer_lip
        palette.MAROON,  # 196: midpoint_3_of_upper_outer_lip
        palette.MAROON,  # 197: midpoint_4_of_upper_outer_lip
        palette.MAROON,  # 198: midpoint_5_of_upper_outer_lip
        palette.MAROON,  # 199: midpoint_6_of_upper_outer_lip
        palette.MAROON,  # 200: midpoint_3_of_lower_outer_lip
        palette.MAROON,  # 201: midpoint_4_of_lower_outer_lip
        palette.MAROON,  # 202: midpoint_5_of_lower_outer_lip
        palette.MAROON,  # 203: midpoint_6_of_lower_outer_lip
        palette.TURQUOISE,  # 204: r_inner_corner_of_mouth
        palette.TURQUOISE,  # 205: l_inner_corner_of_mouth
        palette.TURQUOISE,  # 206: center_of_upper_inner_lip
        palette.TURQUOISE,  # 207: center_of_lower_inner_lip
        palette.TURQUOISE,  # 208: midpoint_1_of_upper_inner_lip
        palette.TURQUOISE,  # 209: midpoint_2_of_upper_inner_lip
        palette.TURQUOISE,  # 210: midpoint_1_of_lower_inner_lip
        palette.TURQUOISE,  # 211: midpoint_2_of_lower_inner_lip
        palette.TURQUOISE,  # 212: midpoint_3_of_upper_inner_lip
        palette.TURQUOISE,  # 213: midpoint_4_of_upper_inner_lip
        palette.TURQUOISE,  # 214: midpoint_5_of_upper_inner_lip
        palette.TURQUOISE,  # 215: midpoint_6_of_upper_inner_lip
        palette.TURQUOISE,  # 216: midpoint_3_of_lower_inner_lip
        palette.TURQUOISE,  # 217: midpoint_4_of_lower_inner_lip
        palette.TURQUOISE,  # 218: midpoint_5_of_lower_inner_lip
        palette.TURQUOISE,  # 219: midpoint_6_of_lower_inner_lip. teeths removed
        palette.YELLOW,  # 256: l_top_end_of_inferior_crus
        palette.YELLOW,  # 257: l_top_end_of_superior_crus
        palette.YELLOW,  # 258: l_start_of_antihelix
        palette.YELLOW,  # 259: l_end_of_antihelix
        palette.YELLOW,  # 260: l_midpoint_1_of_antihelix
        palette.YELLOW,  # 261: l_midpoint_1_of_inferior_crus
        palette.YELLOW,  # 262: l_midpoint_2_of_antihelix
        palette.YELLOW,  # 263: l_midpoint_3_of_antihelix
        palette.YELLOW,  # 264: l_point_1_of_inner_helix
        palette.YELLOW,  # 265: l_point_2_of_inner_helix
        palette.YELLOW,  # 266: l_point_3_of_inner_helix
        palette.YELLOW,  # 267: l_point_4_of_inner_helix
        palette.YELLOW,  # 268: l_point_5_of_inner_helix
        palette.YELLOW,  # 269: l_point_6_of_inner_helix
        palette.YELLOW,  # 270: l_point_7_of_inner_helix
        palette.YELLOW,  # 271: l_highest_point_of_antitragus
        palette.YELLOW,  # 272: l_bottom_point_of_tragus
        palette.YELLOW,  # 273: l_protruding_point_of_tragus
        palette.YELLOW,  # 274: l_top_point_of_tragus
        palette.YELLOW,  # 275: l_start_point_of_crus_of_helix
        palette.YELLOW,  # 276: l_deepest_point_of_concha
        palette.YELLOW,  # 277: l_tip_of_ear_lobe
        palette.YELLOW,  # 278: l_midpoint_between_22_15
        palette.YELLOW,  # 279: l_bottom_connecting_point_of_ear_lobe
        palette.YELLOW,  # 280: l_top_connecting_point_of_helix
        palette.YELLOW,  # 281: l_point_8_of_inner_helix
        palette.CYAN,  # 282: r_top_end_of_inferior_crus
        palette.CYAN,  # 283: r_top_end_of_superior_crus
        palette.CYAN,  # 284: r_start_of_antihelix
        palette.CYAN,  # 285: r_end_of_antihelix
        palette.CYAN,  # 286: r_midpoint_1_of_antihelix
        palette.CYAN,  # 287: r_midpoint_1_of_inferior_crus
        palette.CYAN,  # 288: r_midpoint_2_of_antihelix
        palette.CYAN,  # 289: r_midpoint_3_of_antihelix
        palette.CYAN,  # 290: r_point_1_of_inner_helix
        palette.CYAN,  # 291: r_point_8_of_inner_helix
        palette.CYAN,  # 292: r_point_3_of_inner_helix
        palette.CYAN,  # 293: r_point_4_of_inner_helix
        palette.CYAN,  # 294: r_point_5_of_inner_helix
        palette.CYAN,  # 295: r_point_6_of_inner_helix
        palette.CYAN,  # 296: r_point_7_of_inner_helix
        palette.CYAN,  # 297: r_highest_point_of_antitragus
        palette.CYAN,  # 298: r_bottom_point_of_tragus
        palette.CYAN,  # 299: r_protruding_point_of_tragus
        palette.CYAN,  # 300: r_top_point_of_tragus
        palette.CYAN,  # 301: r_start_point_of_crus_of_helix
        palette.CYAN,  # 302: r_deepest_point_of_concha
        palette.CYAN,  # 303: r_tip_of_ear_lobe
        palette.CYAN,  # 304: r_midpoint_between_22_15
        palette.CYAN,  # 305: r_bottom_connecting_point_of_ear_lobe
        palette.CYAN,  # 306: r_top_connecting_point_of_helix
        palette.CYAN,  # 307: r_point_2_of_inner_helix
        palette.OLIVE_GREEN,  # 308: l_center_of_iris
        palette.OLIVE_GREEN,  # 309: l_border_of_iris_3
        palette.OLIVE_GREEN,  # 310: l_border_of_iris_midpoint_1
        palette.OLIVE_GREEN,  # 311: l_border_of_iris_12
        palette.OLIVE_GREEN,  # 312: l_border_of_iris_midpoint_4
        palette.OLIVE_GREEN,  # 313: l_border_of_iris_9
        palette.OLIVE_GREEN,  # 314: l_border_of_iris_midpoint_3
        palette.OLIVE_GREEN,  # 315: l_border_of_iris_6
        palette.OLIVE_GREEN,  # 316: l_border_of_iris_midpoint_2
        palette.BURGUNDY,  # 317: r_center_of_iris
        palette.BURGUNDY,  # 318: r_border_of_iris_3
        palette.BURGUNDY,  # 319: r_border_of_iris_midpoint_1
        palette.BURGUNDY,  # 320: r_border_of_iris_12
        palette.BURGUNDY,  # 321: r_border_of_iris_midpoint_4
        palette.BURGUNDY,  # 322: r_border_of_iris_9
        palette.BURGUNDY,  # 323: r_border_of_iris_midpoint_3
        palette.BURGUNDY,  # 324: r_border_of_iris_6
        palette.BURGUNDY,  # 325: r_border_of_iris_midpoint_2
        palette.BROWN,  # 326: l_center_of_pupil
        palette.BROWN,  # 327: l_border_of_pupil_3
        palette.BROWN,  # 328: l_border_of_pupil_midpoint_1
        palette.BROWN,  # 329: l_border_of_pupil_12
        palette.BROWN,  # 330: l_border_of_pupil_midpoint_4
        palette.BROWN,  # 331: l_border_of_pupil_9
        palette.BROWN,  # 332: l_border_of_pupil_midpoint_3
        palette.BROWN,  # 333: l_border_of_pupil_6
        palette.BROWN,  # 334: l_border_of_pupil_midpoint_2
        palette.LIGHT_CYAN,  # 335: r_center_of_pupil
        palette.LIGHT_CYAN,  # 336: r_border_of_pupil_3
        palette.LIGHT_CYAN,  # 337: r_border_of_pupil_midpoint_1
        palette.LIGHT_CYAN,  # 338: r_border_of_pupil_12
        palette.LIGHT_CYAN,  # 339: r_border_of_pupil_midpoint_4
        palette.LIGHT_CYAN,  # 340: r_border_of_pupil_9
        palette.LIGHT_CYAN,  # 341: r_border_of_pupil_midpoint_3
        palette.LIGHT_CYAN,  # 342: r_border_of_pupil_6
        palette.LIGHT_CYAN,  # 343: r_border_of_pupil_midpoint_2
    ]
    SKL = {
        0: dict(link=(13, 11), id=0, color=palette.GREEN),
        1: dict(link=(11, 9), id=1, color=palette.GREEN),
        2: dict(link=(14, 12), id=2, color=palette.ORANGE),
        3: dict(link=(12, 10), id=3, color=palette.ORANGE),
        4: dict(link=(9, 10), id=4, color=palette.DODGE_BLUE),
        5: dict(link=(5, 9), id=5, color=palette.DODGE_BLUE),
        6: dict(link=(6, 10), id=6, color=palette.DODGE_BLUE),
        7: dict(link=(5, 6), id=7, color=palette.DODGE_BLUE),
        8: dict(link=(5, 7), id=8, color=palette.GREEN),
        9: dict(link=(6, 8), id=9, color=palette.ORANGE),
        10: dict(link=(7, 62), id=10, color=palette.GREEN),
        11: dict(link=(8, 41), id=11, color=palette.ORANGE),
        12: dict(link=(1, 2), id=12, color=palette.DODGE_BLUE),
        13: dict(link=(0, 1), id=13, color=palette.DODGE_BLUE),
        14: dict(link=(0, 2), id=14, color=palette.DODGE_BLUE),
        15: dict(link=(1, 3), id=15, color=palette.DODGE_BLUE),
        16: dict(link=(2, 4), id=16, color=palette.DODGE_BLUE),
        17: dict(link=(3, 5), id=17, color=palette.DODGE_BLUE),
        18: dict(link=(4, 6), id=18, color=palette.DODGE_BLUE),
        19: dict(link=(13, 15), id=19, color=palette.GREEN),
        20: dict(link=(13, 16), id=20, color=palette.GREEN),
        21: dict(link=(13, 17), id=21, color=palette.GREEN),
        22: dict(link=(14, 18), id=22, color=palette.ORANGE),
        23: dict(link=(14, 19), id=23, color=palette.ORANGE),
        24: dict(link=(14, 20), id=24, color=palette.ORANGE),
        25: dict(link=(62, 45), id=25, color=palette.ORANGE),
        26: dict(link=(45, 44), id=26, color=palette.ORANGE),
        27: dict(link=(44, 43), id=27, color=palette.ORANGE),
        28: dict(link=(43, 42), id=28, color=palette.ORANGE),
        29: dict(link=(62, 49), id=29, color=palette.PINK),
        30: dict(link=(49, 48), id=30, color=palette.PINK),
        31: dict(link=(48, 47), id=31, color=palette.PINK),
        32: dict(link=(47, 46), id=32, color=palette.PINK),
        33: dict(link=(62, 53), id=33, color=palette.SKY_BLUE),
        34: dict(link=(53, 52), id=34, color=palette.SKY_BLUE),
        35: dict(link=(52, 51), id=35, color=palette.SKY_BLUE),
        36: dict(link=(51, 50), id=36, color=palette.SKY_BLUE),
        37: dict(link=(62, 57), id=37, color=palette.RED),
        38: dict(link=(57, 56), id=38, color=palette.RED),
        39: dict(link=(56, 55), id=39, color=palette.RED),
        40: dict(link=(55, 54), id=40, color=palette.RED),
        41: dict(link=(62, 61), id=41, color=palette.GREEN),
        42: dict(link=(61, 60), id=42, color=palette.GREEN),
        43: dict(link=(60, 59), id=43, color=palette.GREEN),
        44: dict(link=(59, 58), id=44, color=palette.GREEN),
        45: dict(link=(41, 24), id=45, color=palette.ORANGE),
        46: dict(link=(24, 23), id=46, color=palette.ORANGE),
        47: dict(link=(23, 22), id=47, color=palette.ORANGE),
        48: dict(link=(22, 21), id=48, color=palette.ORANGE),
        49: dict(link=(41, 28), id=49, color=palette.PINK),
        50: dict(link=(28, 27), id=50, color=palette.PINK),
        51: dict(link=(27, 26), id=51, color=palette.PINK),
        52: dict(link=(26, 25), id=52, color=palette.PINK),
        53: dict(link=(41, 32), id=53, color=palette.SKY_BLUE),
        54: dict(link=(32, 31), id=54, color=palette.SKY_BLUE),
        55: dict(link=(31, 30), id=55, color=palette.SKY_BLUE),
        56: dict(link=(30, 29), id=56, color=palette.SKY_BLUE),
        57: dict(link=(41, 36), id=57, color=palette.RED),
        58: dict(link=(36, 35), id=58, color=palette.RED),
        59: dict(link=(35, 34), id=59, color=palette.RED),
        60: dict(link=(34, 33), id=60, color=palette.RED),
        61: dict(link=(41, 40), id=61, color=palette.GREEN),
        62: dict(link=(40, 39), id=62, color=palette.GREEN),
        63: dict(link=(39, 38), id=63, color=palette.GREEN),
        64: dict(link=(38, 37), id=64, color=palette.GREEN),
    }

    return (KPTS, SKL)


def create_bitmask_from_ids(
    ids: list[int], mode: Literal["include", "exclude"] = "include"
) -> torch.Tensor:
    """
    Generates a bitmask tensor from a list of keypoint indices.

    Returns:
        torch.Tensor: A tensor of shape (308,) with 0s and 1s.
    """
    if mode == "include":
        bitmask = torch.zeros(GOLIATH_NUM_KEYPOINTS)
        bitmask[ids] = 1
    elif mode == "exclude":
        bitmask = torch.ones(GOLIATH_NUM_KEYPOINTS)
        bitmask[ids] = 0
    else:
        raise ValueError("Mode must be either 'include' or 'exclude'")

    return bitmask


def create_bitmask_from_names(
    names: list[str], mode: Literal["include", "exclude"] = "include"
) -> torch.Tensor:
    """
    Generates a bitmask tensor from a list of keypoint names.

    Returns:
        torch.Tensor: A tensor of shape (308,) with 0s and 1s.
    """
    ids = []
    from .utils import log

    for name in names:
        if name in KEYPOINT_NAME_TO_ID:
            ids.append(KEYPOINT_NAME_TO_ID[name])
        else:
            log.warning(
                f"Warning: Keypoint name '{name}' not found. It will be ignored."
            )

    return create_bitmask_from_ids(ids, mode=mode)
