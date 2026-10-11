"""Single-Bean physical diagnostics and finite spawn geometry."""

import mujoco
import numpy as np

from feedingrobot.sim.model import named_id


def ellipse_radial_extent(centre, covariance):
    """Maximum radius of a translated 2D ellipse, from all stationary points."""
    values, vectors = np.linalg.eigh(covariance)
    a, b = np.sqrt(np.maximum(values, 0))
    x, y = vectors.T @ centre
    difference = b*b - a*a
    coefficients = [b*y, 2*(difference-a*x), 0., -2*(a*x+difference), -b*y]
    angles = [0., np.pi]
    if np.any(coefficients):
        angles.extend(2*np.arctan(root.real) for root in np.polynomial.polynomial.polyroots(coefficients)
                      if abs(root.imag) < 1e-9)
    return max(float(np.hypot(x+a*np.cos(angle), y+b*np.sin(angle))) for angle in angles)


def spawn_envelope(task):
    """Whole rotated ellipsoid projection over the real finite support disk."""
    m, d, idx = task.model, task.data, task.index
    geom, bottom = int(idx.bean_collision_geoms[0]), idx.bowl_bottom
    bowl_rotation = d.geom_xmat[bottom].reshape(3, 3)
    centre = bowl_rotation.T @ (d.geom_xpos[geom] - d.geom_xpos[bottom])
    rotation = bowl_rotation.T @ d.geom_xmat[geom].reshape(3, 3)
    axes = m.geom_size[geom]
    projected = rotation[:2] * axes
    radial = ellipse_radial_extent(centre[:2], projected @ projected.T)
    vertical = np.linalg.norm(rotation[2] * axes)
    walls = [g for g in idx.bowl_geoms if g != bottom]
    rim = max(d.geom_xpos[g, 2] + np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g]
              for g in walls)
    return dict(radial_extent_m=radial, bottom_radius_m=float(m.geom_size[bottom, 0]),
                bottom_clearance_m=float(centre[2] - vertical - m.geom_size[bottom, 1]),
                rim_clearance_m=float(rim - d.geom_xpos[geom, 2] - vertical))


def spawn_clearance(task):
    m, d, idx = task.model, task.data, task.index
    geom = int(idx.bean_collision_geoms[0])
    others = [g for g in range(m.ngeom) if g != geom
              and ((int(m.geom_contype[geom]) & int(m.geom_conaffinity[g]))
                   or (int(m.geom_conaffinity[geom]) & int(m.geom_contype[g])))]
    return min(float(mujoco.mj_geomDistance(m, d, geom, other, .1, None)) for other in others)


def bean_state(model, data, index):
    velocities = np.zeros((len(index.bean_ids), 6))
    for i, body in enumerate(index.bean_bodies):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), velocities[i], 0)
    return dict(bean_ids=list(index.bean_ids), bean_positions=data.xpos[index.bean_bodies].copy(),
                bean_quaternions=data.xquat[index.bean_bodies].copy(),
                bean_linear_velocities_world=velocities[:, 3:].copy(),
                bean_angular_velocities_world=velocities[:, :3].copy())


def bean_diagnostics(task):
    m, d, idx = task.model, task.data, task.index
    state = bean_state(m, d, idx)
    bottom = named_id(m, mujoco.mjtObj.mjOBJ_GEOM, 'collision_bowl_fast_bottom_disk')
    walls = [g for g in idx.bowl_geoms if g != bottom]
    floor_z = d.geom_xpos[bottom, 2] + m.geom_size[bottom, 1]
    rim_z = max(d.geom_xpos[g, 2] + np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g] for g in walls)
    limit = task.bean_acceptance['penetration_limit_m']
    bean_geoms = idx.bean_collision_geoms
    bean_centres = d.geom_xpos[bean_geoms]
    bean_rotations = d.geom_xmat[bean_geoms].reshape(-1, 3, 3)
    axes = m.geom_size[bean_geoms]
    vertical = np.linalg.norm(bean_rotations[:, 2, :] * axes, axis=1)
    inside = ((bean_centres[:, 2] - vertical >= floor_z - limit)
              & (bean_centres[:, 2] + vertical <= rim_z + limit))
    boundary_violations = []
    for reason, distances in (('bottom', bean_centres[:, 2] - vertical - floor_z),
                              ('rim', rim_z - bean_centres[:, 2] - vertical)):
        for i in np.flatnonzero(distances < -limit):
            boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason=reason,
                                            distance_m=float(distances[i])))
    for wall in walls:
        rotation = d.geom_xmat[wall].reshape(3, 3)
        local = (bean_centres - d.geom_xpos[wall]) @ rotation
        directions = np.einsum('bij,jk->bik', bean_rotations.transpose(0, 2, 1), rotation)
        radii = np.linalg.norm(directions * axes[:, :, None], axis=1)
        size = m.geom_size[wall]
        overlaps = np.all(np.abs(local[:, 1:]) <= size[1:] + radii[:, 1:], axis=1)
        outward = overlaps & (local[:, 0] > -size[0] + limit)
        inside[outward] = False
        for i in np.flatnonzero(outward):
            boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason='wall_outer_side',
                                            geom_name=m.geom(int(wall)).name,
                                            distance_m=float(-size[0] - local[i, 0])))
        candidates = np.flatnonzero(overlaps & ~outward & (local[:, 0] + radii[:, 0] > -size[0] + limit))
        for i in candidates:
            # Projection overlap is only a broad phase; the lower edge is finite.
            distance = mujoco.mj_geomDistance(m, d, int(bean_geoms[i]), int(wall), .0005, None)
            if distance < -limit:
                inside[i] = False
                boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason='wall_penetration',
                                                geom_name=m.geom(int(wall)).name, distance_m=float(distance)))
    outer_radius = max(np.linalg.norm(d.geom_xpos[g, :2] - d.geom_xpos[bottom, :2])
                       + np.linalg.norm(np.abs(d.geom_xmat[g].reshape(3, 3)[:2]) @ m.geom_size[g]) for g in walls)
    radial_clearance = outer_radius - np.linalg.norm(bean_centres[:, :2] - d.geom_xpos[bottom, :2], axis=1)
    inside &= radial_clearance >= 0
    for i in np.flatnonzero(radial_clearance < 0):
        boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason='outer_radius',
                                        distance_m=float(radial_clearance[i])))
    physical = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
    rotations = d.geom_xmat.reshape(-1, 3, 3)
    centres = d.geom_xpos + np.einsum('gij,gj->gi', rotations, m.geom_aabb[:, :3])
    extents = np.einsum('gij,gj->gi', np.abs(rotations), m.geom_aabb[:, 3:])
    penetration = 0.
    penetration_peak = None
    penetration_ids = set()
    planes = set(physical[m.geom_type[physical] == mujoco.mjtGeom.mjGEOM_PLANE])
    for geom in idx.bean_collision_geoms:
        candidates = physical[np.all(np.abs(centres[physical] - centres[geom]) <= extents[physical] + extents[geom] + .0005, axis=1)]
        candidates = set(candidates) | planes
        for other in candidates:
            if other == geom or (other in idx.bean_id_by_geom and other < geom):
                continue
            distance = mujoco.mj_geomDistance(m, d, int(geom), int(other), .0005, None)
            if -distance > penetration:
                penetration = -distance
                penetration_peak = dict(geom_names=[m.geom(int(g)).name for g in (geom, other)],
                                        bean_ids=[idx.bean_id_by_geom[g] for g in (geom, other)
                                                  if g in idx.bean_id_by_geom],
                                        distance_m=float(distance), source='geometry')
            if distance < -limit:
                penetration_ids.add(idx.bean_id_by_geom[geom])
                if other in idx.bean_id_by_geom:
                    penetration_ids.add(idx.bean_id_by_geom[other])
    for source, rows in (('boundary_contact', task.contacts),
                         ('applied_contact', getattr(task, 'applied_contacts', []))):
        for row in rows:
            if row.get('bean1_id') or row.get('bean2_id'):
                if -row['distance'] > penetration:
                    penetration = -row['distance']
                    penetration_peak = dict(geom_names=[m.geom(row[k]).name for k in ('geom1', 'geom2')],
                                            bean_ids=[b for b in (row['bean1_id'], row['bean2_id']) if b],
                                            distance_m=row['distance'], source=source)
                if row['distance'] < -limit:
                    penetration_ids.update(b for b in (row['bean1_id'], row['bean2_id']) if b)
    roots = {'bowl': set(), 'spoon': set()}
    edges = []
    for row in task.contacts:
        g1, g2 = row['geom1'], row['geom2']
        for geom, other, sign in ((g1, g2, -1), (g2, g1, 1)):
            bean = idx.bean_id_by_geom.get(geom)
            if not bean or sign * row['force_on_geom2_world'][2] <= task.bean_acceptance['support_min_force_n']:
                continue
            if other == bottom:
                roots['bowl'].add(bean)
            elif other in idx.scoop_geoms:
                rotation = d.site_xmat[idx.tcp].reshape(3, 3)
                if any(mujoco.mj_rayMesh(m, d, g, d.geom_xpos[geom], -rotation[:, 2]) >= 0 for g in idx.scoop_geoms):
                    roots['spoon'].add(bean)
            elif other in idx.bean_id_by_geom and d.geom_xpos[geom, 2] > d.geom_xpos[other, 2]:
                edges.append((idx.bean_id_by_geom[other], bean))
    direct = {key: sorted(value) for key, value in roots.items()}
    for supported in roots.values():
        for _ in range(len(idx.bean_ids)):
            before = len(supported)
            supported.update(upper for lower, upper in edges if lower in supported)
            if len(supported) == before:
                break
    return dict(in_bowl=inside, bowl_supported=np.array([b in roots['bowl'] for b in idx.bean_ids]),
                spoon_supported=np.array([b in roots['spoon'] for b in idx.bean_ids]), support_edges=edges,
                linear_speed_m_s=np.linalg.norm(state['bean_linear_velocities_world'], axis=1),
                angular_speed_rad_s=np.linalg.norm(state['bean_angular_velocities_world'], axis=1),
                max_penetration_m=float(penetration), penetration_ids=sorted(penetration_ids),
                penetration_peak=penetration_peak, boundary_violations=boundary_violations,
                direct_support=direct)
