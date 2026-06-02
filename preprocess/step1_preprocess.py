"""
脚本一：OCC预处理
STEP文件 → 提取面/边/顶点特征 → 保存成numpy文件

特点：
  - 不预设面数上限，每个文件按真实大小保存
  - OCC逐条处理（OCC不支持多线程）
  - 只跑一次，结果存到硬盘，训练脚本直接读numpy

输出目录结构：
  numpy_data/
    ├── 00001.npz   （每个STEP文件对应一个npz）
    ├── 00002.npz
    ├── ...
    └── index.csv   （记录每个npz文件的元信息）

运行：
  python step1_preprocess.py --step_dir ./abc_steps --out_dir ./numpy_data
  或直接修改下面的 STEP_DIR 和 OUT_DIR
"""

import os
import csv
import argparse
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve
)
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Pnt2d
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape
)

# ============================================================
# 配置（直接修改这里，或用命令行参数）
# ============================================================
STEP_DIR  = "./abc_steps"       # STEP文件所在目录
OUT_DIR   = "./numpy_data"      # 输出numpy文件的目录
N_UV      = 10                  # UV采样分辨率，每个面采N_UV×N_UV个点
MAX_FILES = 1000                # 最多处理多少个文件，None=全部

FACE_SURF_TYPES  = 7   # 曲面类型数量（平面/圆柱/圆锥/球/环/BSpline/其他）
EDGE_CURVE_TYPES = 5   # 曲线类型数量（直线/圆/椭圆/BSpline/其他）
EDGE_FEAT_DIM    = EDGE_CURVE_TYPES + 8  # 边特征总维度


# ============================================================
# OCC工具函数
# ============================================================

def load_step(filepath):
    reader = STEPControl_Reader()
    if reader.ReadFile(filepath) != IFSelect_RetDone:
        raise RuntimeError(f"无法读取: {filepath}")
    reader.TransferRoots()
    return reader.OneShape()


def face_surface_type_onehot(adaptor):
    """曲面类型 → 7维one-hot"""
    mapping = {
        GeomAbs_Plane: 0, GeomAbs_Cylinder: 1, GeomAbs_Cone: 2,
        GeomAbs_Sphere: 3, GeomAbs_Torus: 4, GeomAbs_BSplineSurface: 5,
    }
    v = np.zeros(FACE_SURF_TYPES, dtype=np.float32)
    v[mapping.get(adaptor.GetType(), 6)] = 1.0
    return v


def edge_curve_type_onehot(adaptor):
    """曲线类型 → 5维one-hot"""
    mapping = {
        GeomAbs_Line: 0, GeomAbs_Circle: 1,
        GeomAbs_Ellipse: 2, GeomAbs_BSplineCurve: 3,
    }
    v = np.zeros(EDGE_CURVE_TYPES, dtype=np.float32)
    v[mapping.get(adaptor.GetType(), 4)] = 1.0
    return v


def sample_face(face, adaptor, n=N_UV):
    """
    对一个面采样UV点阵。
    返回：
      grid      [n, n, 7]  xyz(3) + 法向量(3) + 遮罩(1)
      surf_type [7]        曲面类型one-hot
    """
    u0 = float(adaptor.FirstUParameter())
    u1 = float(adaptor.LastUParameter())
    v0 = float(adaptor.FirstVParameter())
    v1 = float(adaptor.LastVParameter())

    if (u1 - u0) < 1e-10 or (v1 - v0) < 1e-10:
        return None, None

    try:
        classifier = BRepTopAdaptor_FClass2d(face, 1e-6)
    except Exception:
        classifier = None

    grid = np.zeros((n, n, 7), dtype=np.float32)

    for i, u in enumerate(np.linspace(u0, u1, n)):
        for j, v in enumerate(np.linspace(v0, v1, n)):
            uf, vf = float(u), float(v)
            pnt, d1u, d1v = gp_Pnt(), gp_Vec(), gp_Vec()
            try:
                adaptor.D1(uf, vf, pnt, d1u, d1v)
                normal = d1u.Crossed(d1v)
                nl = normal.Magnitude()
                if nl > 1e-10:
                    normal.Normalize()
                    nx, ny, nz = normal.X(), normal.Y(), normal.Z()
                else:
                    nx, ny, nz = 0.0, 0.0, 1.0
            except Exception:
                pnt = adaptor.Value(uf, vf)
                nx, ny, nz = 0.0, 0.0, 1.0

            mask = 1.0
            if classifier is not None:
                try:
                    mask = 1.0 if classifier.Perform(gp_Pnt2d(uf, vf)) == 0 else 0.0
                except Exception:
                    pass

            grid[i, j] = [pnt.X(), pnt.Y(), pnt.Z(), nx, ny, nz, mask]

    return grid, face_surface_type_onehot(adaptor)


def extract_edge_feat(edge):
    """
    提取边特征：曲线类型(5) + 长度(1) + 起止点xyz(6) + 凸凹性(1) = 13维
    """
    try:
        adp = BRepAdaptor_Curve(edge)
        t0, t1 = float(adp.FirstParameter()), float(adp.LastParameter())
        if (t1 - t0) < 1e-10:
            return None

        curve_type = edge_curve_type_onehot(adp)
        length = t1 - t0

        ps, pe = gp_Pnt(), gp_Pnt()
        adp.D0(t0, ps)
        adp.D0(t1, pe)
        start = np.array([ps.X(), ps.Y(), ps.Z()], dtype=np.float32)
        end   = np.array([pe.X(), pe.Y(), pe.Z()], dtype=np.float32)

        try:
            tmp, vs, ve = gp_Pnt(), gp_Vec(), gp_Vec()
            adp.D1(t0, tmp, vs)
            adp.D1(t1, tmp, ve)
            lv = vs.Magnitude()
            le = ve.Magnitude()
            convexity = float(vs.Dot(ve) / (lv * le)) if lv > 1e-10 and le > 1e-10 else 1.0
        except Exception:
            convexity = 1.0

        return np.concatenate([curve_type, [length], start, end, [convexity]]).astype(np.float32)
    except Exception:
        return None


def process_step(filepath):
    """
    解析一个STEP文件，返回包含所有特征的字典。
    字典里每个数组大小由文件本身的几何决定，不做任何填充。

    返回：
      faces_uv   [N_f, n, n, 7]   UV点阵（归一化xyz + 法向量 + 遮罩）
      faces_type [N_f, 7]         曲面类型one-hot
      edges_feat [N_e, 13]        边特征
      verts_xyz  [N_v, 3]         顶点坐标（归一化）
      edge_to_faces [(fi,fj),...]  每条边连接的面对（邻接关系）
      n_faces    int
      n_edges    int
      n_verts    int
    """
    shape = load_step(filepath)

    # 收集所有面
    face_list, face_adaptors = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        try:
            adp = BRepAdaptor_Surface(f)
            face_list.append(f)
            face_adaptors.append(adp)
        except Exception:
            pass
        exp.Next()

    if not face_list:
        return None

    # UV采样
    faces_uv_raw, faces_type_raw, valid_idx = [], [], []
    for i, (f, adp) in enumerate(zip(face_list, face_adaptors)):
        grid, stype = sample_face(f, adp)
        if grid is not None:
            faces_uv_raw.append(grid)
            faces_type_raw.append(stype)
            valid_idx.append(i)

    if not faces_uv_raw:
        return None

    # 归一化xyz到[-1, 1]
    all_xyz = np.concatenate([g[:, :, :3].reshape(-1, 3) for g in faces_uv_raw])
    xyz_min = all_xyz.min(0)
    xyz_range = np.where((all_xyz.max(0) - xyz_min) < 1e-6, 1.0, all_xyz.max(0) - xyz_min)

    faces_uv_norm = []
    for g in faces_uv_raw:
        gn = g.copy()
        gn[:, :, :3] = (g[:, :, :3] - xyz_min) / xyz_range * 2 - 1
        faces_uv_norm.append(gn)

    # 提取边特征和邻接关系
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    face_hash = {hash(f): i for i, f in enumerate(face_list)}

    edges_feat_list, edge_to_faces = [], []
    seen = set()
    exp2 = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp2.More():
        edge = exp2.Current()
        ek = hash(edge)
        if ek not in seen:
            seen.add(ek)
            ef = extract_edge_feat(edge)
            if ef is not None and edge_face_map.Contains(edge):
                it = TopTools_ListIteratorOfListOfShape(edge_face_map.FindFromKey(edge))
                foe = []
                while it.More():
                    fh = hash(it.Value())
                    if fh in face_hash and face_hash[fh] in valid_idx:
                        foe.append(valid_idx.index(face_hash[fh]))
                    it.Next()
                if len(foe) >= 2:
                    edges_feat_list.append(ef)
                    edge_to_faces.append((foe[0], foe[1]))
        exp2.Next()

    # 归一化边的起止点坐标
    if edges_feat_list:
        ef_arr = np.stack(edges_feat_list)
        # 长度归一化
        max_len = ef_arr[:, EDGE_CURVE_TYPES].max()
        if max_len > 1e-6:
            ef_arr[:, EDGE_CURVE_TYPES] /= max_len
        # 起止点坐标归一化
        pts = ef_arr[:, EDGE_CURVE_TYPES+1:EDGE_CURVE_TYPES+7].reshape(-1, 3)
        ef_arr[:, EDGE_CURVE_TYPES+1:EDGE_CURVE_TYPES+7] = \
            ((pts - xyz_min) / xyz_range * 2 - 1).reshape(-1, 6)
    else:
        ef_arr = np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

    # 顶点坐标
    verts_raw = []
    seen_v = set()
    exp3 = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp3.More():
        vh = hash(exp3.Current())
        if vh not in seen_v:
            seen_v.add(vh)
            try:
                p = BRep_Tool.Pnt(exp3.Current())
                verts_raw.append([p.X(), p.Y(), p.Z()])
            except Exception:
                pass
        exp3.Next()

    if verts_raw:
        verts = np.array(verts_raw, dtype=np.float32)
        verts = (verts - xyz_min) / xyz_range * 2 - 1
    else:
        verts = np.zeros((0, 3), dtype=np.float32)

    # edge_to_faces存成int数组
    if edge_to_faces:
        etf = np.array(edge_to_faces, dtype=np.int32)
    else:
        etf = np.zeros((0, 2), dtype=np.int32)

    return {
        'faces_uv':     np.stack(faces_uv_norm).astype(np.float32),  # [N_f, n, n, 7]
        'faces_type':   np.stack(faces_type_raw).astype(np.float32), # [N_f, 7]
        'edges_feat':   ef_arr,                                        # [N_e, 13]
        'verts_xyz':    verts,                                         # [N_v, 3]
        'edge_to_faces': etf,                                          # [N_e, 2]
        'n_faces':      len(faces_uv_norm),
        'n_edges':      len(edges_feat_list),
        'n_verts':      len(verts_raw),
    }


# ============================================================
# 主程序：遍历所有STEP文件，逐条处理，存成npz
# ============================================================

def main(step_dir, out_dir, max_files=None):
    os.makedirs(out_dir, exist_ok=True)

    # 收集所有STEP文件路径
    all_files = sorted([
        os.path.join(root, f)
        for root, dirs, files in os.walk(step_dir)
        for f in files
        if f.lower().endswith('.step') or f.lower().endswith('.stp')
    ])
    if max_files:
        all_files = all_files[:max_files]

    print(f"找到 {len(all_files)} 个STEP文件，开始预处理...")
    print(f"输出目录: {out_dir}")
    print("-" * 50)

    index_rows = []
    success = 0
    fail = 0
    face_counts = []

    for i, fp in enumerate(all_files):
        fname = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_dir, f"{fname}.npz")

        # 已经处理过的跳过（支持断点续传）
        if os.path.exists(out_path):
            success += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(all_files)}] 跳过（已存在）: {fname}")
            continue

        try:
            data = process_step(fp)
            if data is None:
                fail += 1
                continue

            # 保存成npz（numpy的压缩格式，比npy更省空间）
            np.savez_compressed(
                out_path,
                faces_uv     = data['faces_uv'],
                faces_type   = data['faces_type'],
                edges_feat   = data['edges_feat'],
                verts_xyz    = data['verts_xyz'],
                edge_to_faces= data['edge_to_faces'],
            )

            face_counts.append(data['n_faces'])
            index_rows.append({
                'fname':   fname,
                'npz':     f"{fname}.npz",
                'n_faces': data['n_faces'],
                'n_edges': data['n_edges'],
                'n_verts': data['n_verts'],
            })
            success += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
                print(f"  [{i+1}/{len(all_files)}] 成功:{success} 失败:{fail}  "
                      f"当前面数:{data['n_faces']}")

        except Exception as e:
            fail += 1
            if fail <= 10:
                print(f"  [失败] {fname}: {e}")

    # 保存索引文件
    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['fname', 'npz', 'n_faces', 'n_edges', 'n_verts'])
        w.writeheader()
        w.writerows(index_rows)

    print("\n" + "=" * 50)
    print(f"预处理完成！")
    print(f"  成功: {success} 个")
    print(f"  失败: {fail} 个")
    if face_counts:
        print(f"  面数统计: 最小={min(face_counts)}, 最大={max(face_counts)}, "
              f"平均={sum(face_counts)/len(face_counts):.1f}")
        print(f"  95%分位数（建议作为Transformer上限参考）: "
              f"{sorted(face_counts)[int(len(face_counts)*0.95)]}")
    print(f"  索引文件: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_dir', type=str, default=STEP_DIR)
    parser.add_argument('--out_dir',  type=str, default=OUT_DIR)
    parser.add_argument('--max_files',type=int, default=MAX_FILES)
    args = parser.parse_args()
    main(args.step_dir, args.out_dir, args.max_files)
