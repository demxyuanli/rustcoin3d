"""Generate a STEP Part21 file for a 100x80x5 box with square, circular (16-gon), and hexagonal through-holes."""
import math

def gen():
    W, H, T = 100.0, 80.0, 5.0
    sq_cx, sq_cy, sq_s = 25.0, 40.0, 12.0
    ci_cx, ci_cy, ci_r, ci_n = 50.0, 40.0, 8.0, 16
    hx_cx, hx_cy, hx_r, hx_n = 75.0, 40.0, 8.0, 6

    eid = [0]
    ents = {}

    def emit(name, params):
        eid[0] += 1
        id_ = eid[0]
        ents[id_] = f"#{id_} = {name}({params});"
        return id_

    def cp(x, y, z):
        return emit("CARTESIAN_POINT", f"'',({x},{y},{z})")

    def vp(cp_id):
        return emit("VERTEX_POINT", f"'',#{cp_id}")

    def dir_(dx, dy, dz):
        return emit("DIRECTION", f"'',({dx},{dy},{dz})")

    def vec(dir_id, mag=1.0):
        return emit("VECTOR", f"'',#{dir_id},{mag}")

    def line(cp_id, vec_id):
        return emit("LINE", f"'',#{cp_id},#{vec_id}")

    def axis3(cp_id, d1_id, d2_id):
        return emit("AXIS2_PLACEMENT_3D", f"'',#{cp_id},#{d1_id},#{d2_id}")

    def plane(axis_id):
        return emit("PLANE", f"'',#{axis_id}")

    def ec(vp0, vp1, curve_id, sense=".T."):
        return emit("EDGE_CURVE", f"'',#{vp0},#{vp1},#{curve_id},{sense}")

    def oe(ec_id, sense=".T."):
        return emit("ORIENTED_EDGE", f"'',*,*,#{ec_id},{sense}")

    def edge_loop(oe_ids):
        refs = ",".join(f"#{i}" for i in oe_ids)
        return emit("EDGE_LOOP", f"'',({refs})")

    def fob(loop_id, sense=".T."):
        return emit("FACE_OUTER_BOUND", f"'',#{loop_id},{sense}")

    def fb(loop_id, sense=".T."):
        return emit("FACE_BOUND", f"'',#{loop_id},{sense}")

    def af(bound_ids, surf_id, sense=".T."):
        refs = ",".join(f"#{i}" for i in bound_ids)
        return emit("ADVANCED_FACE", f"'',({refs}),#{surf_id},{sense}")

    # Shared directions
    d_x = dir_(1, 0, 0)
    d_y = dir_(0, 1, 0)
    d_nx = dir_(-1, 0, 0)
    d_ny = dir_(0, -1, 0)
    d_z = dir_(0, 0, 1)
    d_nz = dir_(0, 0, -1)
    v_x = vec(d_x)
    v_y = vec(d_y)
    v_nx = vec(d_nx)
    v_ny = vec(d_ny)
    v_z = vec(d_z)

    # --- Vertices ---
    # Box bottom (z=0) and top (z=T)
    b_cp = [(cp(0,0,0), cp(W,0,0), cp(W,H,0), cp(0,H,0)),
            (cp(0,0,T), cp(W,0,T), cp(W,H,T), cp(0,H,T))]
    b_vp = [[vp(c) for c in row] for row in b_cp]

    # Square hole
    hs = sq_s / 2
    s_cp = [(cp(sq_cx-hs,sq_cy-hs,0), cp(sq_cx+hs,sq_cy-hs,0), cp(sq_cx+hs,sq_cy+hs,0), cp(sq_cx-hs,sq_cy+hs,0)),
            (cp(sq_cx-hs,sq_cy-hs,T), cp(sq_cx+hs,sq_cy-hs,T), cp(sq_cx+hs,sq_cy+hs,T), cp(sq_cx-hs,sq_cy+hs,T))]
    s_vp = [[vp(c) for c in row] for row in s_cp]

    # Circle (n-gon)
    def poly_pts(cx, cy, r, n, z):
        return [cp(cx + r*math.cos(2*math.pi*i/n), cy + r*math.sin(2*math.pi*i/n), z) for i in range(n)]

    c_cp = [poly_pts(ci_cx, ci_cy, ci_r, ci_n, 0), poly_pts(ci_cx, ci_cy, ci_r, ci_n, T)]
    c_vp = [[vp(c) for c in row] for row in c_cp]

    h_cp = [poly_pts(hx_cx, hx_cy, hx_r, hx_n, 0), poly_pts(hx_cx, hx_cy, hx_r, hx_n, T)]
    h_vp = [[vp(c) for c in row] for row in h_cp]

    # --- Edge curves ---
    def make_edge(vp0, vp1, cp0_id, dir_vec_id):
        ln = line(cp0_id, dir_vec_id)
        return ec(vp0, vp1, ln)

    # Box edges
    be_b = [make_edge(b_vp[0][i], b_vp[0][(i+1)%4], b_cp[0][i], v_x if i==0 else v_y if i==1 else v_nx if i==2 else v_ny) for i in range(4)]
    be_t = [make_edge(b_vp[1][i], b_vp[1][(i+1)%4], b_cp[1][i], v_x if i==0 else v_y if i==1 else v_nx if i==2 else v_ny) for i in range(4)]
    be_v = [make_edge(b_vp[0][i], b_vp[1][i], b_cp[0][i], v_z) for i in range(4)]

    # Square hole edges
    se_b = [make_edge(s_vp[0][i], s_vp[0][(i+1)%4], s_cp[0][i], v_x if i==0 else v_y if i==1 else v_nx if i==2 else v_ny) for i in range(4)]
    se_t = [make_edge(s_vp[1][i], s_vp[1][(i+1)%4], s_cp[1][i], v_x if i==0 else v_y if i==1 else v_nx if i==2 else v_ny) for i in range(4)]
    se_v = [make_edge(s_vp[0][i], s_vp[1][i], s_cp[0][i], v_z) for i in range(4)]

    # Circle (n-gon) edges
    def poly_edges(cp_row, vp_row, n, cx, cy, r):
        edges = []
        for i in range(n):
            ni = (i+1) % n
            dx = math.cos(2*math.pi*ni/n) - math.cos(2*math.pi*i/n)
            dy = math.sin(2*math.pi*ni/n) - math.sin(2*math.pi*i/n)
            l = math.hypot(dx, dy)
            d = dir_(dx/l, dy/l, 0)
            v = vec(d)
            edges.append(make_edge(vp_row[i], vp_row[ni], cp_row[i], v))
        return edges

    ce_b = poly_edges(c_cp[0], c_vp[0], ci_n, ci_cx, ci_cy, ci_r)
    ce_t = poly_edges(c_cp[1], c_vp[1], ci_n, ci_cx, ci_cy, ci_r)
    ce_v = [make_edge(c_vp[0][i], c_vp[1][i], c_cp[0][i], v_z) for i in range(ci_n)]

    he_b = poly_edges(h_cp[0], h_vp[0], hx_n, hx_cx, hx_cy, hx_r)
    he_t = poly_edges(h_cp[1], h_vp[1], hx_n, hx_cx, hx_cy, hx_r)
    he_v = [make_edge(h_vp[0][i], h_vp[1][i], h_cp[0][i], v_z) for i in range(hx_n)]

    # --- Oriented edges for loops ---
    # Top face outer (CCW from above = z+ direction)
    top_outer_oe = [oe(be_t[0]), oe(be_t[1]), oe(be_t[2]), oe(be_t[3])]
    # Top face inner loops (CW from above = .F. on forward edges)
    top_sq_oe = [oe(se_t[0],".F."), oe(se_t[1],".F."), oe(se_t[2],".F."), oe(se_t[3],".F.")]
    top_ci_oe = [oe(ce_t[i],".F.") for i in range(ci_n)]
    top_hx_oe = [oe(he_t[i],".F.") for i in range(hx_n)]

    # Bottom face outer (CCW from below = CW from above = reversed)
    bot_outer_oe = [oe(be_b[2],".F."), oe(be_b[1],".F."), oe(be_b[0],".F."), oe(be_b[3],".F.")]
    bot_sq_oe = [oe(se_b[2]), oe(se_b[1]), oe(se_b[0]), oe(se_b[3])]
    bot_ci_oe = [oe(ce_b[(ci_n - i) % ci_n]) for i in range(ci_n)]
    bot_hx_oe = [oe(he_b[(hx_n - i) % hx_n]) for i in range(hx_n)]

    # Box side faces
    # Front (y=0): bottom-edge0, vertical1, top-edge0(rev), vertical0(rev)
    side_front_oe = [oe(be_b[0]), oe(be_v[1],".F."), oe(be_t[0],".F."), oe(be_v[0])]
    # Right (x=W): bottom-edge1, vertical2, top-edge1(rev), vertical1(rev)
    side_right_oe = [oe(be_b[1]), oe(be_v[2],".F."), oe(be_t[1],".F."), oe(be_v[1])]
    # Back (y=H): bottom-edge2, vertical3, top-edge2(rev), vertical2(rev)
    side_back_oe = [oe(be_b[2]), oe(be_v[3],".F."), oe(be_t[2],".F."), oe(be_v[2])]
    # Left (x=0): bottom-edge3, vertical0, top-edge3(rev), vertical3(rev)
    side_left_oe = [oe(be_b[3]), oe(be_v[0],".F."), oe(be_t[3],".F."), oe(be_v[3])]

    # Square hole side faces
    sq_sides_oe = []
    for i in range(4):
        ni = (i+1) % 4
        sq_sides_oe.append([oe(se_b[i],".F."), oe(se_v[ni],".F."), oe(se_t[i]), oe(se_v[i])])

    # Circle hole side faces
    ci_sides_oe = []
    for i in range(ci_n):
        ni = (i+1) % ci_n
        ci_sides_oe.append([oe(ce_b[i],".F."), oe(ce_v[ni],".F."), oe(ce_t[i]), oe(ce_v[i])])

    # Hexagon hole side faces
    hx_sides_oe = []
    for i in range(hx_n):
        ni = (i+1) % hx_n
        hx_sides_oe.append([oe(he_b[i],".F."), oe(he_v[ni],".F."), oe(he_t[i]), oe(he_v[i])])

    # --- Edge loops ---
    top_outer_el = edge_loop(top_outer_oe)
    top_sq_el = edge_loop(top_sq_oe)
    top_ci_el = edge_loop(top_ci_oe)
    top_hx_el = edge_loop(top_hx_oe)

    bot_outer_el = edge_loop(bot_outer_oe)
    bot_sq_el = edge_loop(bot_sq_oe)
    bot_ci_el = edge_loop(bot_ci_oe)
    bot_hx_el = edge_loop(bot_hx_oe)

    side_front_el = edge_loop(side_front_oe)
    side_right_el = edge_loop(side_right_oe)
    side_back_el = edge_loop(side_back_oe)
    side_left_el = edge_loop(side_left_oe)

    sq_side_els = [edge_loop(oes) for oes in sq_sides_oe]
    ci_side_els = [edge_loop(oes) for oes in ci_sides_oe]
    hx_side_els = [edge_loop(oes) for oes in hx_sides_oe]

    # --- Face bounds ---
    top_fob = fob(top_outer_el)
    top_sq_fb = fb(top_sq_el)
    top_ci_fb = fb(top_ci_el)
    top_hx_fb = fb(top_hx_el)

    bot_fob = fob(bot_outer_el)
    bot_sq_fb = fb(bot_sq_el)
    bot_ci_fb = fb(bot_ci_el)
    bot_hx_fb = fb(bot_hx_el)

    side_front_fb = fob(side_front_el)
    side_right_fb = fob(side_right_el)
    side_back_fb = fob(side_back_el)
    side_left_fb = fob(side_left_el)

    sq_side_fbs = [fob(el) for el in sq_side_els]
    ci_side_fbs = [fob(el) for el in ci_side_els]
    hx_side_fbs = [fob(el) for el in hx_side_els]

    # --- Surfaces (PLANE + AXIS2_PLACEMENT_3D) ---
    def make_plane(origin_cp, axis_d, ref_d):
        a = axis3(origin_cp, axis_d, ref_d)
        return plane(a)

    top_surf = make_plane(cp(0,0,T), d_z, d_x)
    bot_surf = make_plane(cp(0,0,0), d_nz, d_x)

    side_front_surf = make_plane(cp(0,0,0), d_ny, d_x)
    side_right_surf = make_plane(cp(W,0,0), d_x, d_y)
    side_back_surf = make_plane(cp(0,H,0), d_y, d_nx)
    side_left_surf = make_plane(cp(0,0,0), d_nx, d_ny)

    # Square hole side surfaces
    sq_side_surfs = []
    sq_normals = [(0,-1,0), (1,0,0), (0,1,0), (-1,0,0)]
    sq_refs =    [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]
    for i in range(4):
        nd = dir_(*sq_normals[i])
        rd = dir_(*sq_refs[i])
        sq_side_surfs.append(make_plane(s_cp[0][i], nd, rd))

    # Circle hole side surfaces
    ci_side_surfs = []
    for i in range(ci_n):
        ni = (i+1) % ci_n
        mx = (ci_cx + ci_r*math.cos(2*math.pi*(2*i+1)/(2*ci_n)))
        my = (ci_cy + ci_r*math.sin(2*math.pi*(2*i+1)/(2*ci_n)))
        dx = math.cos(2*math.pi*(2*i+1)/(2*ci_n))
        dy = math.sin(2*math.pi*(2*i+1)/(2*ci_n))
        nd = dir_(dx, dy, 0)
        rd = dir_(0, 0, 1)
        ci_side_surfs.append(make_plane(cp(mx, my, 0), nd, rd))

    # Hexagon hole side surfaces
    hx_side_surfs = []
    for i in range(hx_n):
        ni = (i+1) % hx_n
        mx = (hx_cx + hx_r*math.cos(2*math.pi*(2*i+1)/(2*hx_n)))
        my = (hx_cy + hx_r*math.sin(2*math.pi*(2*i+1)/(2*hx_n)))
        dx = math.cos(2*math.pi*(2*i+1)/(2*hx_n))
        dy = math.sin(2*math.pi*(2*i+1)/(2*hx_n))
        nd = dir_(dx, dy, 0)
        rd = dir_(0, 0, 1)
        hx_side_surfs.append(make_plane(cp(mx, my, 0), nd, rd))

    # --- Advanced faces ---
    top_af = af([top_fob, top_sq_fb, top_ci_fb, top_hx_fb], top_surf)
    bot_af = af([bot_fob, bot_sq_fb, bot_ci_fb, bot_hx_fb], bot_surf)
    front_af = af([side_front_fb], side_front_surf)
    right_af = af([side_right_fb], side_right_surf)
    back_af = af([side_back_fb], side_back_surf)
    left_af = af([side_left_fb], side_left_surf)

    sq_side_afs = [af([fb], surf) for fb, surf in zip(sq_side_fbs, sq_side_surfs)]
    ci_side_afs = [af([fb], surf) for fb, surf in zip(ci_side_fbs, ci_side_surfs)]
    hx_side_afs = [af([fb], surf) for fb, surf in zip(hx_side_fbs, hx_side_surfs)]

    # --- Closed shell ---
    all_face_ids = [top_af, bot_af, front_af, right_af, back_af, left_af] + sq_side_afs + ci_side_afs + hx_side_afs
    face_refs = ",".join(f"#{i}" for i in all_face_ids)
    shell_id = emit("CLOSED_SHELL", f"'',({face_refs})")
    msb_id = emit("MANIFOLD_SOLID_BREP", f"'',#{shell_id}")

    # --- Product tree ---
    place_d1 = dir_(0, 0, 1)
    place_d2 = dir_(1, 0, 0)
    place_cp = cp(0, 0, 0)
    place_a = axis3(place_cp, place_d1, place_d2)
    abrep_id = emit("ADVANCED_BREP_SHAPE_REPRESENTATION", f"'',(#{place_a},#{msb_id}),#{eid[0]+5}")

    ac = emit("APPLICATION_CONTEXT", "'core data for automotive mechanical design processes'")
    apd = emit("APPLICATION_PROTOCOL_DEFINITION", f"'international standard','automotive_design',2000,#{ac}")
    pc = emit("PRODUCT_CONTEXT", f"'',#{ac},'mechanical'")
    prod = emit("PRODUCT", "'HoledPlate','HoledPlate','',(#{pc})")
    pdf = emit("PRODUCT_DEFINITION_FORMATION", f"'','',#{prod}")
    pdc = emit("PRODUCT_DEFINITION_CONTEXT", f"'part definition',#{ac},'design'")
    pd = emit("PRODUCT_DEFINITION", f"'design','',#{pdf},#{pdc}")
    pds = emit("PRODUCT_DEFINITION_SHAPE", f"'','',#{pd}")
    sdr = emit("SHAPE_DEFINITION_REPRESENTATION", f"#{pds},#{abrep_id}")

    # Units
    lu = emit("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )", "")
    pau = emit("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )", "")
    sau = emit("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )", "")
    um = emit("UNCERTAINTY_MEASURE_WITH_UNIT", f"LENGTH_MEASURE(1.E-07),#{lu},'distance_accuracy_value','confusion accuracy'")
    rc = emit("( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{um})) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{pau},#{sau})) REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )", "")
    prpc = emit("PRODUCT_RELATED_PRODUCT_CATEGORY", f"'part',$,(#{prod})")

    # --- Output ---
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('Multi-hole plate solid'),'2;1');",
        "FILE_NAME('holed_plate','2026-06-09T00:00:00',('Author'),(''),'','','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
        "ENDSEC;",
        "DATA;",
    ]
    for i in range(1, eid[0]+1):
        if i in ents:
            lines.append(ents[i])
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "HoledPlate.step")
    content = gen()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Written {out_path} ({len(content)} bytes)")
