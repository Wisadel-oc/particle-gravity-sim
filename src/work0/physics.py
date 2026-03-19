import taichi as ti
from .config import (
    N_PARTICLES, DT, SOFTENING, MOUSE_GRAVITY_STRENGTH,
    PARTICLE_REPULSION, PARTICLE_ATTRACTION, PARTICLE_INTERACT_RADIUS
)

ti.init(arch=ti.cuda, device_memory_GB=1.0)

pos = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)

@ti.kernel
def init_particles():
    for i in range(N_PARTICLES):
        pos[i] = ti.Vector([ti.random() * 0.9 + 0.05, ti.random() * 0.9 + 0.05])
        vel[i] = ti.Vector([ti.random() * 0.2 - 0.1, ti.random() * 0.2 - 0.1])

@ti.kernel
def apply_particle_interactions():
    ti.loop_config(serialize=True)
    for i in range(N_PARTICLES):
        for j in range(N_PARTICLES):
            if i != j:
                r = pos[j] - pos[i]
                dist = r.norm()
                if dist < PARTICLE_INTERACT_RADIUS:
                    dir = r / (dist + SOFTENING)
                    if dist < 0.05:
                        force = dir * PARTICLE_REPULSION / (dist * dist)
                        vel[i] += force * DT
                    elif dist < PARTICLE_INTERACT_RADIUS:
                        force = dir * PARTICLE_ATTRACTION * dist
                        vel[i] += force * DT

@ti.kernel
def update_physics_base():
    for i in range(N_PARTICLES):
        vel[i] *= 0.8
        pos[i] += vel[i] * DT
        for d in ti.static(range(2)):
            if pos[i][d] < 0.05:
                pos[i][d] = 0.05
                vel[i][d] *= -0.8
            elif pos[i][d] > 0.95:
                pos[i][d] = 0.95
                vel[i][d] *= -0.8

@ti.kernel
def apply_mouse_force(mouse_x: float, mouse_y: float, strength: float):
    for i in range(N_PARTICLES):
        r = ti.Vector([mouse_x, mouse_y]) - pos[i]
        dist = r.norm() + SOFTENING
        force = r * strength / (dist * dist)
        vel[i] += force * DT

init_particles()