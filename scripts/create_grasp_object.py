"""Create a graspable compound object (base + neck + flange) as a USD file.

Shape:
    ┌──────┐       ← flange (hook/rim, prevents gripper from slipping off)
    └──┐┌──┘
       ││          ← neck (narrow, gripper closes around this)
       ││
  ┌────┴┴────┐     ← base (wide, stable on ground)
  └──────────┘

All dimensions in meters. Total mass: 0.2kg.
"""

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# Dimensions (meters)
BASE_SIZE = (0.12, 0.12, 0.04)       # wide base: 12x12x4cm
NECK_RADIUS = 0.015                   # neck: 3cm diameter (gripper can easily close around)
NECK_HEIGHT = 0.06                    # neck: 6cm tall
FLANGE_SIZE = (0.06, 0.06, 0.02)     # flange: 6x6x2cm (wider than neck, prevents slip)

# Vertical layout
BASE_Z = BASE_SIZE[2] / 2                              # base center: 2cm
NECK_Z = BASE_SIZE[2] + NECK_HEIGHT / 2                # neck center: 7cm
FLANGE_Z = BASE_SIZE[2] + NECK_HEIGHT + FLANGE_SIZE[2] / 2  # flange center: 11cm
TOTAL_HEIGHT = BASE_SIZE[2] + NECK_HEIGHT + FLANGE_SIZE[2]   # 12cm total

MASS = 0.2  # kg

OUTPUT_PATH = "assets/grasp_object.usda"


def create_grasp_object():
    stage = Usd.Stage.CreateNew(OUTPUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root xform
    root = UsdGeom.Xform.Define(stage, "/GraspObject")
    root.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

    # Physics: rigid body on root
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

    # Mass
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(MASS)

    # --- Base (wide cuboid) ---
    base = UsdGeom.Cube.Define(stage, "/GraspObject/base")
    base.AddTranslateOp().Set(Gf.Vec3d(0, 0, BASE_Z))
    base.AddScaleOp().Set(Gf.Vec3f(
        BASE_SIZE[0] / 2, BASE_SIZE[1] / 2, BASE_SIZE[2] / 2
    ))
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    # Physics material for base
    mat_path = "/GraspObject/physics_material"
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(mat_path, "Material"))
    mat_prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI(mat_prim)
    phys_mat.CreateStaticFrictionAttr(2.0)
    phys_mat.CreateDynamicFrictionAttr(2.0)
    phys_mat.CreateRestitutionAttr(0.0)

    # Bind material
    for part_path in ["/GraspObject/base", "/GraspObject/neck", "/GraspObject/flange"]:
        prim = stage.GetPrimAtPath(part_path)
        if prim:
            binding = UsdPhysics.MaterialAPI.Apply(prim)

    # --- Neck (narrow cylinder) ---
    neck = UsdGeom.Cylinder.Define(stage, "/GraspObject/neck")
    neck.AddTranslateOp().Set(Gf.Vec3d(0, 0, NECK_Z))
    neck.CreateRadiusAttr(NECK_RADIUS)
    neck.CreateHeightAttr(NECK_HEIGHT)
    neck.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(neck.GetPrim())

    # --- Flange (wide top, hook/rim) ---
    flange = UsdGeom.Cube.Define(stage, "/GraspObject/flange")
    flange.AddTranslateOp().Set(Gf.Vec3d(0, 0, FLANGE_Z))
    flange.AddScaleOp().Set(Gf.Vec3f(
        FLANGE_SIZE[0] / 2, FLANGE_SIZE[1] / 2, FLANGE_SIZE[2] / 2
    ))
    UsdPhysics.CollisionAPI.Apply(flange.GetPrim())

    # --- Visual color (red) ---
    for path in ["/GraspObject/base", "/GraspObject/neck", "/GraspObject/flange"]:
        prim = stage.GetPrimAtPath(path)
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([(0.8, 0.2, 0.2)])

    stage.GetRootLayer().Save()
    print(f"Created grasp object: {OUTPUT_PATH}")
    print(f"  Base: {BASE_SIZE[0]*100:.0f}x{BASE_SIZE[1]*100:.0f}x{BASE_SIZE[2]*100:.0f}cm")
    print(f"  Neck: {NECK_RADIUS*200:.0f}cm diameter x {NECK_HEIGHT*100:.0f}cm")
    print(f"  Flange: {FLANGE_SIZE[0]*100:.0f}x{FLANGE_SIZE[1]*100:.0f}x{FLANGE_SIZE[2]*100:.0f}cm")
    print(f"  Total height: {TOTAL_HEIGHT*100:.0f}cm")
    print(f"  Mass: {MASS}kg")
    print(f"  Neck center at z={NECK_Z*100:.0f}cm (gripper target)")


if __name__ == "__main__":
    create_grasp_object()
