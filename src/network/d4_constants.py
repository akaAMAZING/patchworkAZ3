"""
Canonical D4 / LUT constants for patchworkaz.

PC_MAX=33: supports piece IDs 1-32 and empty (slot_id=-1).
- Encoding: slot_id in [-1, 1..32] -> p_enc = (slot_id + 1).clamp(0, 33)
  - p_enc=0: empty
  - p_enc=2..33: pieces 1..32 (slot_id 1..32)
- Compact LUT index: p0c = (p0_enc - 1).clamp(0, PC_MAX-1)
  - p0c=0: empty (p0_enc=0 maps to 0)
  - p0c=1..32: pieces 1..32 (p0_enc 2..33 maps to 1..32)
- p0/p1/p2 for orient_tbl: clamp(-1, PC_MAX-1) to cover -1..32
"""

# Max piece ID in compact space (1..32) + empty; total 33 slots in encoding
PC_MAX = 33
COMPACT_SIZE = 8 * (PC_MAX**3)  # 8 * 33^3 = 287496
MULT = 34  # encoding multiplier: p_enc in 0..33
