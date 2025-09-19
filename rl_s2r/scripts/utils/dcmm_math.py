import numpy as np
def qmul_xyzw(q2, q1):
    """四元数乘法（xyzw）：返回 q2 ⊗ q1（先做 q1，再做 q2）"""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
        w2*w1 - x2*x1 - y2*y1 - z2*z1
    ])

def rotate_B_to_A_90_xyzw(qB_xyzw, qA_xyzw=None):
    """把组B(xyzw)右乘标准 Rz(-90°)，返回归一化后的 xyzw；若给了 qA，则符号对齐到 qA。"""
    s = np.sqrt(0.5)
    Rz_m90 = np.array([0.0, 0.0, -s, s])     # Rz(-90°) in xyzw
    q = qmul_xyzw(qB_xyzw, Rz_m90)           # 右乘：q' = qB ⊗ Rz(-90°)
    q = q / np.linalg.norm(q)
    if qA_xyzw is not None and np.dot(q, qA_xyzw) < 0:
        q = -q                                # 符号对齐（q 与 -q 等价）
    return q