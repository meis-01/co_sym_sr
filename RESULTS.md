# Executed CFD results

Only directories with a final dataset and report appear in the completed table.
Errors below are numerical measurements, not claims of physiological fidelity.

| Run | Cells | Cycles | Periodic velocity | Final scaled divergence | Max analytical velocity L2 error |
|---|---:|---:|---:|---:|---:|
| `results/expansion` | 337 | 11 | 0.001179 | 0.5784 | n/a |
| `results/expansion_stabilized` | 764 | 8 | 0.001047 | 0.1016 | n/a |
| `results/expansion_strict` | 764 | 4 | 0.001662 | 0.08656 | n/a |
| `results/poiseuille` | 337 | 6 | 0.0005678 | 0.1657 | 4.687% |
| `results/womersley` | 337 | 5 | 0.001666 | 0.4329 | 18.216% |
| `results/womersley_low_alpha` | 337 | 4 | 0.0005922 | 0.3326 | 13.357% |
| `studies/convergence/base` | 337 | 8 | 0.0003818 | 0.433 | 18.195% |
| `studies/convergence/dt_fine` | 337 | 8 | 0.0003829 | 0.4344 | 18.200% |
| `studies/convergence/dt_medium` | 337 | 8 | 0.0003825 | 0.4339 | 18.199% |
| `studies/convergence/mesh_fine` | 1250 | 5 | 0.0004849 | 0.3706 | 12.376% |
| `studies/convergence/mesh_medium` | 764 | 5 | 0.0004323 | 0.3618 | 13.868% |
| `studies/rotation/orientation_0` | 337 | 7 | 0.001895 | 0.7035 | n/a |
| `studies/rotation/orientation_90` | 337 | 8 | 0.0009067 | 0.7151 | n/a |

The initial unstabilized expansion is a coarse diagnostic dataset. Its local divergence
requires refinement/stabilization despite excellent global flux balance.
All datasets retain `accuracy_certified: false`; use study results and application-specific tolerances.
The ratio-1000 expansion missed the 0.1 local-divergence check (0.1016). The ratio-10000
follow-up passed (0.0866), but changes velocity by 11.4% and pressure by 30.5% relative to
ratio 1000 on the interior comparison nodes. This sensitivity requires finer aortic meshes;
passing the divergence threshold alone does not make these resolution-independent training fields.

## Convergence study

```json
{
  "mesh_medium": {
    "relative_velocity_l2_nodes": 0.0773392274693205,
    "relative_pressure_l2_nodes": 0.06236272887867229,
    "points": 234,
    "angle_radians": 0.0,
    "note": "Interior node-weighted norm; excludes outer 15% of radius and end 5% of length",
    "analytical_velocity_error": 0.13868107939415597
  },
  "mesh_fine": {
    "relative_velocity_l2_nodes": 0.08039999119810223,
    "relative_pressure_l2_nodes": 0.09039656337232808,
    "points": 234,
    "angle_radians": 0.0,
    "note": "Interior node-weighted norm; excludes outer 15% of radius and end 5% of length",
    "analytical_velocity_error": 0.1237625080601945
  },
  "dt_medium": {
    "relative_velocity_l2_nodes": 0.00027839796690821967,
    "relative_pressure_l2_nodes": 0.010358225437858429,
    "points": 234,
    "angle_radians": 0.0,
    "note": "Interior node-weighted norm; excludes outer 15% of radius and end 5% of length",
    "analytical_velocity_error": 0.1819862169727491
  },
  "dt_fine": {
    "relative_velocity_l2_nodes": 0.0004184960202694475,
    "relative_pressure_l2_nodes": 0.01553715286381722,
    "points": 234,
    "angle_radians": 0.0,
    "note": "Interior node-weighted norm; excludes outer 15% of radius and end 5% of length",
    "analytical_velocity_error": 0.18200340225825115
  },
  "base": {
    "analytical_velocity_error": 0.18195234257446294
  },
  "mesh_error_decreases": true,
  "temporal_to_finest": {
    "coarse": 0.000418492434310668,
    "medium": 0.0001400979089286064,
    "error_decreases": true,
    "pressure_coarse": 0.015543679199666765,
    "pressure_medium": 0.005181292507093642,
    "pressure_error_decreases": true
  },
  "observed_mesh_error_slope": 0.8311107916943457,
  "passed": true
}
```

## Rotation study

```json
{
  "relative_velocity_l2_nodes": 0.13831066292893082,
  "relative_pressure_l2_nodes": 0.0118566979012658,
  "points": 234,
  "angle_radians": 1.5707963267948966,
  "note": "Interior node-weighted norm; excludes outer 15% of radius and end 5% of length",
  "passed": true
}
```

## Stabilization study

```json
{
  "relative_velocity_l2_nodes": 0.11366038669342907,
  "relative_pressure_l2_nodes": 0.3047496374943173,
  "points": 598,
  "angle_radians": 0.0,
  "note": "Interior node-weighted norm; excludes outer 15% of radius and end 5% of length"
}

```

## Interrupted or failed runs

- `results/poiseuille_initial_aborted`: aborted: Stopped to replace expensive high-order reference kernel evaluation with independent vectorized quadrature; no final CFD result claimed.
- `results/womersley_initial_aborted`: aborted: Stopped to replace expensive high-order reference kernel evaluation with independent vectorized quadrature; no final CFD result claimed.
